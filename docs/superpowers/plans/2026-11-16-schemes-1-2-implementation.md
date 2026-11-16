# Schemes 1 and 2 Implementation Plan

> **For agentic workers:** Use subagent-driven-development with independent file ownership, then integration and review in this session. Do not launch scientific experiments.

**Goal:** Implement and publish two separate reproducible experiment entrypoints, with audited reuse of existing B/U/results and no stage2 execution in scheme2.

**Architecture:** Keep historical modules untouched. Add scheme_models (modules), scheme_reuse (read-only source admission), scheme_fitting (resumable stages), scheme_report (raw verification/statistics), and separate scheme1_runner/scheme2_runner. Reuse shared encoded datasets and mmap feature caches. Scheme2 saves head-only weights; scheme1 retains the established full task checkpoint format (including frozen U) for direct loading and reproducibility. Source B, datasets and feature caches remain references.

**Tech Stack:** Existing Python/PyTorch/NumPy/SciPy/unittest; existing full-split tri_data and source model semantics.

## Task 1: Models and behavioral tests

Files: sska_rec/scheme_models.py; tests/test_scheme_models.py.

- [x] Add failing tests for zero-delta bitwise equality at probabilities 0/1, frozen gradients, parameter counts (scheme2:1541/773; scheme1 LoRA:30901/23833/11233), deterministic task initialization and shared U immutability.
- [x] Implement GSDecoder(rdim, xdim): single Linear(3*rdim, xdim).
- [x] Implement ComplementaryExpert(widths, rep_dim, seed): same UniversalExpert state schema; losses(x,g,specifics,gs_decoder,mask=None,redundancy_weight=.01,variance_weight=.01) reconstruct normalized x minus frozen D_GS([g,s0,s1]), masked equal-field error; mean blockwise U/G/S correlation and existing anti-collapse variance. D_GS and targets detached.
- [x] Implement ConditionalReadout(rdim, rank=4): affine-free LN, A:2r→k and C:r→k without biases, out:k→1 with zero weight/bias; forward(h,u,p) preserves p via probability difference.
- [x] Implement old_task_hidden(base,batch,position): exact original gate/environment fusion of cached G/S.
- [x] Implement build_scheme1_model(spec,u,seed,variant): existing MPT/three frozen/LoRA/SSF/full update scopes; for U variants replace gate by Linear(3r,3,bias=False) operating on expert representations, common initialization, no fourth switch. Exclude decoder from prediction.
- [x] Run targeted tests to green; report source freezing and exact counts.

## Task 2: Read-only audited sources

Files: sska_rec/scheme_reuse.py; tests/test_scheme_reuse.py.

- [x] Failing tests: same-data full Census accepted, source weights/cache corruption rejected, target/seed/split/source/protocol mismatch rejected, historical receipt paths resolved without copies, CPU/CUDA and relocated raw paths do not invalidate identical data, absent explicit source never triggers old training.
- [x] Implement load_sources(spec,seed,cfg,source_root,require_u=True) returning (base,caches,envs,u_or_None,receipt). source_root may be dataset/seed folder, dataset root or outputs/tri root; resolve reuse-receipt.json stages, verify tri-source-identity.json or historical proof, file SHA256, model hashes, cache owner/rowids and actual labels. No writes to source roots.
- [x] Compare actual semantic source hash keys for base model/training/U, not unrelated new execution modules; canonical data fingerprint ignores source location and mtime but binds content/splits/features/tasks.
- [x] Scheme1 reuses B/cache only and computes a fresh U; scheme2 requires exact original U and never trains U or calls stage2. Save read-only source references and receipts in new outputs.
- [x] Run targeted integrity tests to green.

## Task 3: Stage fitting and resume

Files: sska_rec/scheme_fitting.py; tests/test_scheme_fitting.py.

- [x] Write failing small-tensor integration tests for GS decoder/U training, source immutability, per-task epoch0 selection, no-repeat completed stage, protocol mutation rejection, finite losses, corruption detection and distinct rowids.
- [x] fit_decoder(spec,caches,folder,cfg,seed,status_path=None) returns (decoder,metrics): normalize x from train-only streaming statistics; Linear GS decoder predicts normalized x; val loss chooses earliest best; save state/normalizer/history/UTC/cost/contract/COMMITTED.
- [x] fit_complementary_u(spec,caches,decoder,normalizer,folder,cfg,seed,status_path=None) returns (u,metrics): train the new scheme1 U objective, deterministic validation masks, freeze decoder, no original U encoder reuse; save source.pt and dependencies.
- [x] fit_readout(base,u,spec,caches,position,folder,cfg,seed,status_path=None) returns metrics: only ConditionalReadout trained with BCE, independent target val AUC, epoch0 original MPT included, cache h/u per batch without dataset replication, retain raw predictions/base probabilities and true rowids, weights/head only, source guards.
- [x] Generic stage contracts bind source/data/config and relevant implementation identity. COMMITTED and ready_for_commit reuse must verify files/state/raw/contract before bypassing optimization. Invalid explicit reuse fails; incomplete leaf preserved then restarted, never delete valid negative results.
- [x] Run targeted tests to green.

## Task 4: Integration, configs and launch tools

Files: sska_rec/scheme1_runner.py; scheme2_runner.py; scheme_report.py; scheme_parallel.py; configs/scheme1_three_datasets.json; scheme2_three_datasets.json; experiments/schemes-20261116/{README.md,run.sh,health.py}; tests/test_scheme_runners.py.

- [x] Configs inherit all exact existing dataset/seed/LR/batch/30epochs/patience5 values. Scheme1: pretraining + stage2 all five variants; scheme2: original targets only and no stage2 option. Independent output version identities.
- [x] --source-root required for authenticated historical sources; --dataset-root shared; --dataset/--seed support single-job parallel scheduling; --report-only never loads data/trains. Scheme1 --pretrain-only useful for source scheduling. Run locks and source lock prevent duplicate workers.
- [x] Parallel scheduler assigns one independent dataset/seed per worker with resource-controlled --workers and --threads-per-worker, skips validated finished jobs, waits all before completion. Do not create background services or cloud instances.
- [x] Scheme1 MPT controls may be referenced only after matching B/data/protocol/initialization/semantic implementation and raw/hash checks; no old U-variant reuse as new scheme1 results.
- [x] Raw verified report includes full five-seed AUC/Logloss mean/sampleSD/variance, paired differences and t intervals, costs, UTC, provenance. Partial cells explicitly count completed seeds. No test-based selection.
- [x] Document reuse matrix and exact commands; add concise root README links. Include the approved two design records; leave unrelated untracked files untouched.

## Task 5: Verify and publish

- [x] Baseline unit suite, targeted red/green stages, integrated toy end-to-end both schemes and all model variants, full final suite, CLI --help, source corruption/resume tests, diff whitespace check.
- [x] Independent spec and implementation review; fix material issues and re-run affected checks.
Publication: commit only intended files on `experiment/schemes-1-2-20261116`, push to `aru0988/SSKA-Rec` and verify the remote SHA. Keep main unchanged. No formal experiment has been launched.

## Verification record

- Baseline: 93 tests passed.
- Final: 128 tests passed, including real-tensor end-to-end execution of both schemes, all five scheme1 model variants, resume/reuse and raw-result reporting.
- Parallel launch and stage1-only CLI help checked; shell script uses LF line endings.
- Completed-seed scheduling skips training after result/source checks. Shared preprocessing admission is serialized per dataset; independent model training runs concurrently.
