"""Table 6 deployment sizes and measured matrix-operation FLOPs.

The profiler executes the original stage2_models.build_model variants, starting
at raw features. FLOPs are the supported operators reported by torch.profiler,
not an estimate obtained by multiplying forward cost by a training factor.
"""

from contextlib import contextmanager

import torch
from torch.nn import functional as F

from .stage2_models import build_model


VARIANTS = ('mpt', 'tri_frozen', 'tri_lora')
MATRIX_OPERATORS = {'aten::mm', 'aten::addmm', 'aten::bmm', 'aten::baddbmm',
                    'aten::mv', 'aten::addmv'}


@contextmanager
def _evaluation_mode(source):
    modes = [(module, module.training) for module in source.modules()]
    source.eval()
    try:
        yield
    finally:
        # train(mode) recursively overwrites mixed child states; restore directly.
        for module, mode in modes:
            module.training = mode


def _unique_parameters(named_parameters):
    parameters = {id(parameter): parameter for _, parameter in named_parameters}
    return sum(parameter.numel() for parameter in parameters.values())


def _coverage(profiler):
    counted, uncounted = {}, {}
    total = 0
    for event in profiler.key_averages():
        if not event.key.startswith('aten::'):
            continue
        flops = int(event.flops or 0)
        if flops and event.key in MATRIX_OPERATORS:
            total += flops
            counted[event.key] = dict(calls=int(event.count), flops=flops)
        else:
            uncounted[event.key] = int(event.count)
    return total, dict(counted_operators=counted, uncounted_operators=uncounted)


def _measure(action):
    # CPU dispatcher events also describe matrix dimensions for CUDA kernels.
    # CUDA timing is not collected: this module reports operation counts only.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
    ) as profiler:
        result = action()
    flops, coverage = _coverage(profiler)
    return result, flops, coverage


def _raw_prediction(base, model, raw, envs):
    # The frozen B inference branch is traversed exactly once. Calling B.forward
    # would additionally run old-task training predictions and its GRL branch.
    with torch.no_grad():
        x = base.embedding_network(raw)
        batch = dict(
            x=x,
            g=base.shared_expert_network(x),
            s0=base.specific_expert_networks[0](x),
            s1=base.specific_expert_networks[1](x),
        )
    return model(batch, envs)


def profile(spec, base, u, envs, raw_batch, seed):
    """Return JSON-able sizes and forward/backward matrix FLOPs for 3 variants.

    raw_batch maps feature names to same-length tensors. Its optional _target
    tensor contains labels of spec['tasks'][2]; absent labels are replaced with
    zeros because target values do not change the executed operator shapes.
    Sources retain their state, gradients, requires_grad flags and mixed modes;
    no optimizer is created and no source parameters are updated.
    """
    if not raw_batch:
        raise ValueError('raw_batch must contain raw feature tensors')
    reference = next(base.parameters())
    device, dtype = reference.device, reference.dtype
    raw = {name: value.detach().to(device) for name, value in raw_batch.items()
           if name != '_target'}
    if not raw:
        raise ValueError('raw_batch must contain raw feature tensors')
    batch_size = len(next(iter(raw.values())))
    if batch_size < 1 or any(len(value) != batch_size for value in raw.values()):
        raise ValueError('raw feature tensors must have equal positive batch size')
    target = raw_batch.get('_target')
    label_source = 'raw_batch._target' if target is not None else 'synthetic_zero'
    target = (target.detach().to(device=device, dtype=dtype).reshape(-1)
              if target is not None else torch.zeros(batch_size, device=device, dtype=dtype))
    if len(target) != batch_size:
        raise ValueError('_target must match raw feature batch size')
    envs = [value.detach().to(device=device, dtype=dtype) for value in envs]
    base_deployment = [(f'base.{name}', parameter) for name, parameter in base.named_parameters()
                       if not name.startswith('env_classifier.')]
    base_active = [(name, parameter) for name, parameter in base_deployment
                   if not name.startswith(('base.gate_networks.', 'base.tower_networks.'))]
    report = dict(
        schema_version=1,
        dataset=spec.get('dataset'),
        seed=int(seed),
        target=spec['tasks'][2],
        label_source=label_source,
        flop_scope='FLOPs_matmul_only: torch.profiler supported matrix operations',
        convention='One multiply-add = 2 FLOPs; matrix MACs = counted FLOPs / 2.',
        scope=dict(
            deployment='Unique parameters in full B old-task inference model plus one new-task model and its U encoder; excludes B.env_classifier and U.decoder. U source is not counted again. Conditional old-task U heads are not included.',
            active_new_task='B embedding, G/S experts and stored original environment embeddings + new-task core + one U encoder. Original B gates/towers, B.env_classifier and U.decoder are excluded.',
            trainable='Only new-task core and, for tri_lora, rank-4 U adapters.',
            forward='Raw-feature embedding lookup + frozen B shared and two specific experts once + one new-task prediction. Old-task heads are stored but not executed for this target.',
            backward='Actual BCE backward into trainable new-task parameters. Frozen B is detached. No optimizer or regularization pass.',
            uncounted='Embedding lookups, bias additions, nonlinearities, normalization, softmax, BCE, elementwise/reduction and data movement are excluded. All observed aten operators outside supported matrix operators are listed per pass, including add/mul even if torch.profiler assigns them FLOPs.',
            normalization_buffers='U mean/scale buffers are required for inference but excluded from parameter counts.',
        ),
        torch_version=str(torch.__version__),
        variants={},
    )
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda_devices), _evaluation_mode(base):
        for variant in VARIANTS:
            model = build_model(spec, u, seed, variant).to(device=device, dtype=dtype)
            stage_deployment = [(f'new.{name}', parameter) for name, parameter in model.named_parameters()
                                if not name.startswith('u.decoder.')]
            model.eval()
            with torch.no_grad():
                _, inference_flops, inference_coverage = _measure(
                    lambda: _raw_prediction(base, model, raw, envs))
            model.train()

            def forward_loss():
                return F.binary_cross_entropy(_raw_prediction(base, model, raw, envs), target)

            loss, forward_flops, forward_coverage = _measure(forward_loss)
            _, backward_flops, backward_coverage = _measure(loss.backward)
            if backward_flops == 0:
                raise RuntimeError('torch.profiler reported no backward matrix FLOPs')
            total = forward_flops + backward_flops
            report['variants'][variant] = dict(
                batch_size=batch_size,
                deployment_parameters=_unique_parameters(base_deployment + stage_deployment),
                active_new_task_inference_parameters=_unique_parameters(base_active + stage_deployment),
                base_deployment_parameters=_unique_parameters(base_deployment),
                new_task_deployment_parameters=_unique_parameters(stage_deployment),
                trainable_parameters=model.trainable_parameters,
                inference_forward_flops=inference_flops,
                training_forward_flops=forward_flops,
                training_backward_flops=backward_flops,
                training_total_flops=total,
                inference_forward_macs=inference_flops / 2,
                training_total_macs=total / 2,
                inference_flops_per_sample=inference_flops / batch_size,
                training_flops_per_sample=total / batch_size,
                coverage=dict(inference=inference_coverage, forward=forward_coverage, backward=backward_coverage),
            )
    return report
