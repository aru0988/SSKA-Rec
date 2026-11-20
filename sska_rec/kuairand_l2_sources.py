"""Versioned normalized-L2 KuaiRand sources; original source defaults stay unchanged."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from . import core, training
from .scheme_common import bind_contract, digest, read_json, verify_stage

PROJECT = Path(__file__).resolve().parents[1]
VERSION = "kuairand-normalized-l2-headonly-v1-20261120"
OWNERS = ("core.py", "training.py", "base_training.py", "backbone.py", "expert.py",
          "kuairand_l2_sources.py")


def default_config():
    dc = copy.deepcopy(read_json(PROJECT/"configs/tri_three_datasets.json")["datasets"]["kuairand"])
    dc.update(embedding_l2=.006, embedding_l2_normalization="parameter_mean")
    return dict(version=VERSION, datasets={"kuairand":dc}, order=["kuairand"],
                readouts=["conditional_readout","head_only"],
                variants=["mpt","tri_frozen","random_lora","tri_lora"],
                execution=dict(workers=4,threads_per_worker=2),
                expected_training_units=70, expected_queue_jobs=65)


def validate_config(cfg):
    if cfg != default_config():
        raise ValueError("Registered normalized-L2 experiment configuration changed")


def regularization(spec):
    if (spec["dataset"] != "kuairand" or
        spec.get("embedding_l2_normalization") != "parameter_mean" or
        spec.get("embedding_l2",.006) != .006):
        raise ValueError("Expected explicit normalized KuaiRand source protocol")
    n = sum(spec["vocab"].values())*spec["embed"]
    if n <= 0:
        raise ValueError("Empty embedding table")
    return dict(definition=".006/(2*N_E)*sum(E^2)", embedding_elements=n,
                embedding_l2=.006, effective_embedding_coefficient=.006/n,
                dnn_l2=3e-5, normalization="parameter_mean")


def identity(spec,seed,cfg):
    reg = regularization(spec)
    if (cfg.get("embedding_l2_normalization") != "parameter_mean" or
        cfg.get("embedding_l2",.006) != .006):
        raise ValueError("Source configuration regularization mismatch")
    return dict(version=VERSION,seed=seed,regularization=reg,config=cfg,
                data=spec.get("data_signature"),
                architecture={k:spec[k] for k in ("dataset","vocab","embed","xdim","rdim",
                                                  "hidden","tower","widths","initial","tasks")},
                source={n:core.file_hash(Path(__file__).with_name(n)) for n in OWNERS})


class DiagnosticTrainManager(core.PairedTrainManager):
    """Add end-of-epoch measurements without changing the inherited optimizer."""
    def evaluation_two_task(self,loader):
        scores = super().evaluation_two_task(loader)
        if loader is self.val_loader:
            with torch.no_grad():
                emb = self.model.reg_embedding*self.model.embedding_network.get_l2_reg()
                total = self.model.get_l2_reg()
                squares = sum(p.detach().square().sum() for p in self.model.embedding_network.parameters())
                count = sum(p.numel() for p in self.model.embedding_network.parameters())
            self.epoch_records[-1].update(
                train_fused_bce_0=self.fused_loss_0_list[-1],
                train_fused_bce_1=self.fused_loss_1_list[-1],
                train_general_bce_0=self.uni_loss_0_list[-1],
                train_general_bce_1=self.uni_loss_1_list[-1],
                train_environment_nll=self.env_loss_list[-1],
                end_epoch_embedding_l2=float(emb),end_epoch_dnn_l2=float(total-emb),
                end_epoch_embedding_rms=float((squares/count).sqrt()))
        return scores


def _snapshot(path):
    entries = read_json(Path(path)/"COMMITTED.json")
    return dict(manifest_sha256=core.file_hash(Path(path)/"COMMITTED.json"),
                files={n:dict(sha256=h,bytes=(Path(path)/n).stat().st_size,
                              mtime_ns=(Path(path)/n).stat().st_mtime_ns)
                       for n,h in entries.items()})


def _check_cache(folder,receipt):
    path = Path(folder)/"feature-cache"
    if core.file_hash(path/"COMMITTED.json") != receipt["manifest_sha256"]:
        raise ValueError("Representation cache manifest changed")
    for name,proof in receipt["files"].items():
        stat=(path/name).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(proof["bytes"],proof["mtime_ns"]):
            raise ValueError("Immutable representation cache file changed")


def _evaluation(base,caches,spec,seed,folder):
    path=folder/"source-evaluation"
    if (path/"COMMITTED.json").exists():
        verify_stage(path,["metrics.json","predictions.npz"])
        value=read_json(path/"metrics.json")
        if value["model_hash"] != core.model_hash(base) or value["paired_seed"]!=seed:
            raise ValueError("Source evaluation identity changed")
        return value
    path.mkdir(parents=True,exist_ok=True)
    value=dict(variant="mpt_base",paired_seed=seed,seed=seed,model_hash=core.model_hash(base),
               regularization=regularization(spec),data_identity=spec.get("data_signature"),
               total_parameters=sum(p.numel() for p in base.parameters()))
    raw={}
    for split in ("val","test"):
        y=np.asarray(spec["sets"][split].y[:,:2],dtype=np.float32)
        p=caches[split]["old"].numpy()
        value[split]={t:core.measures(y[:,i],p[:,i]) for i,t in enumerate(spec["initial"])}
        raw.update({split+"_y":y,split+"_p":p,split+"_rowid":spec["sets"][split].row_ids})
    tmp=path/"predictions.npz.tmp"
    with tmp.open("wb") as stream:
        np.savez_compressed(stream,**raw)
    tmp.replace(path/"predictions.npz")
    core.dump(path/"metrics.json",value);training._commit(core,path)
    return value


def prepare_source(spec,seed,folder,cfg,status_path):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    contract=identity(spec,seed,cfg)
    bind_contract(folder/"normalized-source-identity.json",contract)
    if (folder/"source-ready.json").exists():
        return load_source(spec,seed,folder,cfg)
    budget=cfg.get("base_epochs",30)
    if budget != cfg.get("ssl_epochs",budget):
        raise ValueError("B/U source budgets differ")
    previous=core.PairedTrainManager
    previous_cache=training._feature_cache
    def cache_and_evaluate(*args):
        cache, environments=previous_cache(*args)
        _evaluation(args[1],cache,spec,seed,folder)
        return cache,environments
    try:
        core.PairedTrainManager=DiagnosticTrainManager
        training._feature_cache=cache_and_evaluate
        base,caches,envs,u=training._prepare(core,spec,seed,folder,budget)
    finally:
        core.PairedTrainManager=previous
        training._feature_cache=previous_cache
    evaluation=_evaluation(base,caches,spec,seed,folder)
    receipt=dict(version=VERSION,seed=seed,regularization=regularization(spec),
                 identity_sha256=core.file_hash(folder/"normalized-source-identity.json"),
                 base_hash=core.model_hash(base),u_hash=core.model_hash(u),
                 base=_snapshot(folder/"source/base"),u=_snapshot(folder/"source/ssl"),
                 cache=_snapshot(folder/"feature-cache"),
                 evaluation_sha256=core.file_hash(folder/"source-evaluation/metrics.json"))
    core.dump(folder/"source-ready.json",receipt)
    return base,caches,envs,u


def load_source(spec,seed,folder,cfg):
    """Read committed immutable source state, never train or regenerate a cache."""
    folder=Path(folder)
    if read_json(folder/"normalized-source-identity.json") != identity(spec,seed,cfg):
        raise ValueError("Source identity differs; do not mix experiment versions")
    receipt=read_json(folder/"source-ready.json")
    if receipt["regularization"] != regularization(spec) or receipt["seed"]!=seed:
        raise ValueError("Source receipt pairing differs")
    if receipt["identity_sha256"] != core.file_hash(folder/"normalized-source-identity.json"):
        raise ValueError("Source receipt identity hash differs")
    verify_stage(folder/"source/base",["metrics.json","base.pt"])
    verify_stage(folder/"source/ssl",["metrics.json","source.pt"])
    if (receipt["base"]["manifest_sha256"]!=core.file_hash(folder/"source/base/COMMITTED.json")
        or receipt["u"]["manifest_sha256"]!=core.file_hash(folder/"source/ssl/COMMITTED.json")
        or receipt["evaluation_sha256"]!=core.file_hash(folder/"source-evaluation/metrics.json")):
        raise ValueError("Committed source receipt changed")
    _check_cache(folder,receipt["cache"])
    core.seed_all(seed)
    base=core.build_base(spec)
    base.load_state_dict(torch.load(folder/"source/base/base.pt",map_location=core.DEVICE,weights_only=True))
    u=core.make_u(spec,seed)
    u.load_state_dict(torch.load(folder/"source/ssl/source.pt",map_location=core.DEVICE,weights_only=True))
    if (core.model_hash(base),core.model_hash(u))!=(receipt["base_hash"],receipt["u_hash"]):
        raise ValueError("Source state hash differs")
    base.eval().requires_grad_(False);u.eval().requires_grad_(False)
    caches={}
    for split,rows in spec["sets"].items():
        caches[split]={key:torch.from_numpy(np.load(folder/"feature-cache"/f"{split}-{key}.npy",mmap_mode="c"))
                       for key in ("x","g","s0","s1","old")}
        caches[split]["y"]=torch.from_numpy(rows.y)
    cache_identity=read_json(folder/"feature-cache/identity.json")
    if cache_identity["base_hash"]!=receipt["base_hash"]:
        raise ValueError("Cache belongs to a different backbone")
    from .tri_data import _array_hash
    for split,rows in spec["sets"].items():
        if cache_identity["splits"][split]!=dict(rows=len(rows),rowid_sha256=_array_hash(rows.row_ids)):
            raise ValueError("Cache row identity changed")
    data_digest=digest(spec.get("data_signature"))
    spec['scheme_data_identity']=data_digest
    spec['scheme_cache_identity']=dict(protocol=VERSION,dataset=spec['dataset'],seed=seed,
        base_hash=receipt['base_hash'],cache_manifest_sha256=receipt['cache']['manifest_sha256'],
        cache_files={n:p['sha256'] for n,p in receipt['cache']['files'].items()},
        data_signature_sha256=data_digest)
    envs=[base.env_embedding_network(torch.tensor(k,device=core.DEVICE)).detach() for k in range(2)]
    return base,caches,envs,u


def freeze_guard(base,caches,envs,u,spec,readout_root=None):
    hashes=(core.model_hash(base),core.model_hash(u))
    heads={}
    if readout_root is not None:
        for path in Path(readout_root).glob("*/conditional_readout/model.pt"):
            if (path.parent/"COMMITTED.json").exists():
                heads[str(path)]=core.file_hash(path)
    def guard():
        if (core.model_hash(base),core.model_hash(u))!=hashes:
            raise ValueError("Frozen original source changed")
        if any(p.requires_grad for m in (base,u) for p in m.parameters()) or base.training or u.training:
            raise ValueError("Original source is not frozen in evaluation mode")
        if any(core.file_hash(Path(p))!=h for p,h in heads.items()):
            raise ValueError("Deployed original readout changed")
        return dict(passed=True,base_hash=hashes[0],u_hash=hashes[1],
                    original_head_sha256=heads,
                    prediction_invariance="unchanged frozen B/U/readout states and deterministic inference graph")
    return guard
