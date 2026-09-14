# NeurALPS Phase 2: full execution plan

**Version:** phase2.2, 14 September 2026. Read with `docs/architecture.md` and `configs/experiment_registry.json`.

**Objective:** build an NRPS foundation encoder on frozen ESM-C, evaluate transfer to construct activity and quantitative T-domain activity, and expose correctly conditioned natural-support and activity-edit tables. The main sequence is E0 -> E1 -> F1 -> F2 -> F3 -> T1/T2 -> R1. F4 (D) and P1 (source-context adapter) are gated branches. Names denote experiments, not completed production runs.

**Completion of this document:** the architecture, work order, required comparisons, outputs and decision rules are specified. The reference package contains executable components and an explicit contract-test command. Current-data training has not run here. Commands marked available below exist in the package; proposed production entry points are described as implementation tasks, not presented as runnable scripts.

## 1. Establish one execution workspace

Run from a clone of this repository. Keep prior checkpoints and evaluations in a separate results namespace, such as `runs/phase2_2/`. The implementation files live in `src/neuralps_v2/`; install the package in editable mode before running the commands below.

Copy `configs/data_paths.template.json` to `configs/data_paths.local.json` and fill it from the current audited manifests. Relative data paths resolve from the local path file's directory. Dataset and model-weight locations belong in this ignored local configuration; the repository does not prescribe account-specific cluster paths.

Use the site's supported PyTorch environment. A prior record names PyTorch 2.8/Python 3.12, but no SLURM account, GPU partition or resource allocation is invented in this plan. Save package versions, accelerator, CUDA version and model-weight digest in every run record. Start FP32 on a small batch. Only request full GPU time after the profile and gradient gates pass.

Available package commands, executed from the repository root after installation:

```bash
python scripts/check_contracts.py --json contract_results.json
python scripts/check_contracts.py --require-torch --json torch_contract_results.json
neuralps-threshold-summary
neuralps-preflight --config configs/phase2_final_config.json
neuralps-preflight --config configs/phase2_final_config.json --paths configs/data_paths.local.json
```

The first command can report skipped PyTorch tests. The second deliberately returns nonzero if any required runtime test is skipped. Preflight without `--paths` checks package/config readiness only; it does not grant readiness for training. With an unfilled local copy of the path template it fails and lists missing integrations. `evaluation_protocol.py` prints the iGEM manuscript's threshold arithmetic; it does not evaluate a trained model.

## 2. E0: freeze the data handshake, evaluator and folds

**Inputs:** accepted current natural/Bode manifests and caches, current legacy evaluator/index code, exact label maps, donor memberships and three newly evidence files described in the source guide.

**Implementation tasks:**

1. Implement `export_current_graphs.py` for the current domain/connection occurrence tables. Reconstruct the physical alternating route, include terminal TE, preserve translated-chain boundaries and output canonical JSONL. Modules remain metadata. This is a schema export of accepted data, not a wholesale new biological extraction.
2. Export `feature_index.json` keyed by exact extraction identity. Keep sequence SHA-256, parent input, pooling coordinates, model/tokenizer recipe, owned spans and full dependency spans. Reject an ambiguous shared feature key rather than merging contexts.
3. Export labels and engineered-donor memberships independently from features. Keep common recipient scaffold identifiers separate. Map the paper's 105 single swaps to Bode-2 IDs and identify the additional record explicitly.
4. Repair and execute the legacy D offset/actual-candidate/validity handling on the exact current source. Verify offsets with known assemblies of different transition counts. Remove score clipping. Export finite, actual-candidate scores before comparing old model results.
5. Build five fixed outer donor groups using only donor occurrence counts and composition. Start with a deterministic greedy assignment of high-frequency donors to the least-covered group; record algorithm/seed and manually review empty/degenerate topology cases without looking for favorable test labels. Create inner donor groups within each outer training set. Report both-class feasibility; unresolved folds are flagged, not silently replaced.
6. Create nested 10/25/50/100% label subsets within the permitted outer training rows, including labels allocated to inner validation. Record hashes, donor coverage and all unevaluable cases. Do not recycle outer held-out labels for checkpoint selection.
7. Audit natural/engineered homology and source-pathway overlap under the existing split contract. Publish the exposure policy and an optional stricter sensitivity cohort. Keep all test labels out of sampling, feature statistics and optimization.

**Required outputs:** `dataset_manifest.json`, `natural_{train,dev,test}.jsonl`, `bode1.jsonl`, `bode2.jsonl`, `feature_index.json`, `labels.tsv`, `donor_membership.jsonl`, `bode2_paper_cohort.tsv`, `split_manifest.json`, `label_budgets.json`, `overlap_report.json`, `legacy_evaluator_repair.md` and `environment.json`.

**Acceptance:** accepted counts are conserved; all records trace to source coordinates; no module relabel changes neural inputs; breaks cannot become through-translations; duplicate feature identity means identical extraction context; every label joins once; no held-out engineered donor enters training. Verify role entries versus unique-key unions without changing audited biology to make totals match.

**Stop/repair branch:** missing context provenance blocks masked SSL for the affected rows, not all clean-feature baselines. Missing current paths blocks production execution, not the package's numerical tests. Missing author cohort IDs blocks exact paper reproduction, not a separately declared full-cohort metric.

Available adapter check after export:

```bash
neuralps-validate-data natural_train.jsonl --index embeddings/feature_index.json
```

The path is an example of the canonical output layout. The exporter must populate it before this command can succeed.

## 3. E1: establish the supervised and similarity references

**Purpose:** measure what can be predicted before NRPS-specific SSL and isolate the direct B contribution.

Fit the following on Bode-1 outer-training labels only, with inner selection and fixed budgets:

| ID | Inputs/model | Required comparison |
|---|---|---|
| B00 | Domain-only frozen ESM-C features plus regularized nonlinear head | Historical-style reference with repaired folds |
| B01 | Frozen D plus all connection vectors, common nonlinear head | Direct B/J representation benefit over B00 |
| B02 | A_pool with no natural SSL, same activity head | Architecture-only comparator for F1 |
| B03 | TE similarity alone, where source provenance is available | Biological information baseline |
| B04 | TE plus physical exchange position and C class, regularized predictor | Simple source/context explanation |
| B05 | Frozen ESM-C plus the same source information | Input-matched comparator for enriched NeurALPS |

A regularized linear or kernel classifier is a useful small-data check, with the same outer/inner protocol. Limit the tuning grid in the config; do not spend the full label budget searching architectures.

For Bode-2, the manuscript's >50% threshold is a descriptive published heuristic derived from that family. Reproduce TP=43/FP=8/FN=20/TN=34 only on the mapped 105 cohort if the score definition matches. Do not call it a prospective independent test. Compute continuous TE AUROC only from per-construct scores. Learned TE-plus-context coefficients must come from an allowed development set; if Bode-1 lacks comparable source features, report that limitation rather than fitting on Bode-2.

**Outputs:** one `predictions.parquet` or TSV per run with unique construct ID, outer fold, held-out donor group, seed, label budget, label, logit/probability, input track and model ID. Save `fold_metrics.json`, preprocessing fits, checkpoints and `baseline_summary.md`.

**Acceptance:** predictions align to labels by ID; controls are not accidentally duplicated; reported uncertainty distinguishes seed SD and data-sampling uncertainty. Report macro fold AUROC, AP, Brier/log loss and selection precision at a fixed budget. No architecture claim from changing head and inputs simultaneously.

## 4. F1: implement and train A_pool foundation

**Prerequisite:** E0 adapter/masking tests and E1 references. E1 comparisons use supervised labels; the F1 pretraining objective uses natural-training data only.

Implementation work order:

1. `teacher_statistics.py`: compute training-only group means and fallback counts, save feature/split hashes. Raw-cosine recipe uses zero mean tensors; centered ablation uses saved means.
2. `mask_sampler.py`: sample eligible primary D/connection targets, invoke provenance closure, log survival and apply the same eligibility rule for all compared architectures.
3. `candidate_sampler.py`: same-role training-only panels, multi-positive identity handling, explicit validity; return query indices, never stale prediction tensors.
4. `train_ssl.py`: wrap the supplied bundle/loss helpers with deterministic sampler, device transfer, accumulation, scheduler, checkpointing and natural-dev evaluation. No CLI of that name ships yet; implementing it is this stage's task.
5. `extract_features.py`: clean forwards before heads, preserving occurrence order and model/config identity.

**Pilot:** 256 fixed eligible natural assemblies, one seed, FP32. Run forward/backward through every active branch, evaluate masked-input invariance and inspect closure. Then 1,024 assemblies for throughput/memory estimates. A memorization/overfit check on a tiny training slice is a gradient diagnostic, never a performance result.

**Run:** A_pool, up to 30 epochs, AdamW 3e-4, effective 32 eligible lines/update, 5% warm-up and cosine decay. Primary object loss is raw cosine with rank weight ramp to 0.05. Candidate count starts at 32. Track object, category, ranking, survival, gradient and variance diagnostics.

**Controlled variants:** rank weight 0 versus 0.05; raw cosine versus centered 0.75 residual mixture. Run the first screen at a fixed short budget and one seed. Complete the selected preregistered comparisons with five paired seeds. Do not conflate more objective tuning with more label-free evidence: checkpoint selection using Bode inner labels must be charged to that task's protocol.

**Checkpoint artifacts:** natural-only model and decoder, config, data/split/statistics/sampler hashes, update/epoch, optimizer/scheduler/RNG state, and sampler state for exact resumption. Save fixed-interval natural-dev checkpoints; keep their identities when fitting task heads.

**Acceptance:** valid targets and connected gradients, no contaminated views, meaningful discrimination beyond type-mean prediction, stable natural dev behavior and complete feature export. Transfer acceptance compares F1 with B02 under identical input/head/budget; natural loss alone does not prove transfer.

## 5. F2: add residue detail and the span-objective experiment

**Preferred branch:** A_residue with actual frozen per-residue ESM-C states. **Comparator:** A_raw with a new raw-AA CNN. Both are compared with matched pooled A and equal extra training.

Implement `residue_views.py`: enumerate retained contiguous windows/domain views, translated coordinates, region roles, input recipe, context-aware feature identities and aliases. Use the same recipe for natural and engineered data. Preserve the existing pooled cache. Short T domains receive full residue coverage; larger D objects require a documented tiled view if full coverage is claimed.

Implement `extract_residue_states.py` for the actual frozen ESM-C model. Pilot a deduplicated, stratified set before bulk extraction. Save exact AA token positions excluding CLS/EOS/PAD. Benchmark clean caching, online corrupted-context extraction and storage; choose the valid feasible route. Masked spans inside a clean contextual cache require recomputation of affected inputs. The trainable pool alone cannot remove leakage.

Attach `bundle.residue_encoder` and, when using spans, `bundle.span_decoder`. The supplied `PretrainedResidueEncoder` maps residue tensors and region masks to 128-dimensional residue states and 256-dimensional pooled slots. Its inputs are detached; gradients train the pool. The raw comparator uses the supplied CNN with current corrupted AA tokens.

Use a zero-argument `raw_features` factory in the update helper to compute and scatter the current trainable residue features into [B,T,3,256] after stage mode is set. Return a matching availability mask. That factory must not carry features from an earlier batch or mask seed.

Span callback contract: `span_loss(bundle)` constructs the independently corrupted span view, recomputes residue features, runs the foundation encoder, predicts only selected AA labels using SpanDecoder, aggregates selected physical residues/relations/assemblies and applies both D pass weights if applicable. Return one connected finite scalar. The callback and its collator must be implemented; passing a precomputed scalar is rejected.

**Schedule:** one warm epoch of new pool/gate/decoders at 3e-4; up to nine joint epochs with inherited encoder 1e-4 and new pool/decoders 3e-4. Compare object-plus-rank versus object-plus-rank-plus-span. Span weight ramps to 0.2. If clean-context dependencies remove almost all evidence, stop the affected objective and resolve extraction rather than weakening the contract.

**Acceptance:** correct AA alignment, padding and empty-region behavior; intradomain edits change intended features; no cross-chain convolution/PLM context; connected residue/span gradients; measured storage/throughput within allocation. Transfer comparison must control both input resolution and extra training duration.

## 6. F3: add C assembly context

Copy the entire selected A bundle into a compatible C bundle. `copy_variant` validates shared parameter shapes and copies decoder, residue pool and task-head type. C-only branches retain their identity initialization. In eval mode, initial A and copied C must match within declared FP32 tolerance.

Run one warm epoch at 1e-4 for global slots, global blocks and decoder; inherited A remains frozen. The zero-initialized outward path means read-side gradients can begin only after that path opens. Inspect at initialization and after several updates; do not mislabel the expected first-step behavior as permanently disconnected parameters. Then train jointly for up to nine epochs with inherited 3e-5 and new context/decoder 1e-4.

Compare against A with equal extra optimizer updates and a budget-matched depth comparator. Use the same corruption eligibility set, input resolution, supervised head and label splits. Record measured updates, wall time and peak memory; equal epochs can have different cost.

**Acceptance:** all valid D/E tokens can receive assembly context; chain IDs are used relationally; changing a distant permitted object can affect its contextual predictions. A/C initialize identically, then separate through learned context. Advance C as a preferred release only when the registered downstream comparison supports its benefit/cost. The manuscript's position effects motivate testing C but do not substitute for that comparison.

## 7. T1 and T2: downstream evaluation of reusable checkpoints

### 7.1 T1: Bode construct activity

For each selected natural checkpoint, fork independent task copies. Probe the common 770-feature MLP first. Search weight decay in {0.001,0.01,0.1} inside allowed inner folds; cap at 100 epochs with patience 10. Restricted fine-tuning then opens the declared A/C/D layers with encoder LR 1e-5 and head LR 1e-4, cap 30 epochs/patience 5. Broad adaptation is an explicitly registered follow-up only.

Use five paired seeds and the same nested 10/25/50/100% label subsets. Report retained donor coverage. For final Bode-2 scoring, fix epoch/stopping rules using Bode-1 inner results, fit allowed Bode-1 labels and export all external predictions once. The selected model cannot use external outcomes to choose threshold or checkpoint. Record the evaluation as retrospective.

### 7.2 T2: quantitative T-domain activity

Implement `build_tdomain_task.py` to join the provided sheet's actual sequences, assay and parent/generation metadata to full background constructs. Keep nonnumeric/missing/not-tested rows out of the numeric loss; preserve truncation flags and WT controls. Recover original author split IDs for a literal reproduction; otherwise freeze a newly named parent/distance split and nearest-distance report.

First run mean-pooled ESM-C plus RF and fixed-length unrolled ESM-C plus ridge on the eligible aligned cohort. For causal attribution to pooling, cross compatible heads where feasible rather than assuming the published mixed-head comparison isolates representation. Add a T-only probe and an NRPS-contextual probe on the actual reconstructed assembly. Use the same labelled split and target transform.

Primary target is log1p(percent WT) for regression training; primary evaluation is Spearman in the original assay ordering. Record ties and non-detection handling. Do not mix normalizations across assays into one unexplained global rank loss. A separate supervised Bradley-Terry experiment can use within-assay rankings and tie rules, with all pair construction confined to training rows.

Task copies share only the frozen natural checkpoint, not each other's labels. A later multitask experiment must be reported separately. The worksheet's numeric row count does not recover the authors' 85-sequence benchmark by itself.

## 8. P1, I1 and F4: controlled extensions

### 8.1 P1: source-context adapter

Start after scalar source-TE baselines and coverage are established. Implement source sequence loading independently of construct graph loading. Begin with donor-source TE, recipient TE and the versioned alignment score in a small residual downstream adapter. Fit only on allowed development labels. A full natural source-route embedding is a later variant. Apply identical source inputs to an ESM-only comparator.

Natural pretraining initially does not take source copies as inputs. If source views are later added to SSL, extend alias/dependency closure first. Missing donor-source data preserves the construct-only prediction route and receives explicit coverage reporting. Never train a new mechanism/dataset identity category using external Bode-2 labels.

### 8.2 I1: additive activity map and candidate tables

Cross the supplied additive head with the same selected encoders. Report its prediction tradeoff relative to the common MLP. Verify that exported factors sum to its logit. Center displays against a fixed reference and state that correlated factors are not causal seam probabilities.

Implement `build_candidate_graph.py` and `regenerate_candidate_features.py`. Inputs specify the exact changed sequence, ordered partners, assembly background, precursor/mature state and source-context policy. Outputs are a physically valid rebuilt graph with all affected feature keys regenerated, or a typed unsupported/error status. Candidate columns never contain dummy values scored as real candidates.

Implement `score_candidate_table.py`: load natural-only and task-adapted checkpoint identities explicitly, use proper masked views for natural support and clean complete candidates for activity, and output a tidy table. Required columns include checkpoint/config hash, cell IDs, sequence hashes, background, reference, natural score/panel, activity logit/probability, delta logit, seed SD, coverage, nearest-training similarity and support status. A sequence edit can change several contextual factors; recompute the full affected model view.

Validation uses real matched constructs where available: same donor in different positions, scar controls and multiple substitutions. Do not treat known scar coordinates as ground-truth failure labels. No causal localization claim without suitable experimental validation.

### 8.3 F4: optional D refinement

Register this branch only after C is stable. Copy complete C to D, verify identity at initialization, then use one warm and up to nine joint epochs. Apply 0.25/0.75 loss weights to object, candidate-ranking, span and activity objectives. Preserve the same corruption through both passes. Compare with C receiving equivalent extra computation and a deeper nonrecycled control.

If D is not retained, A/C release is still complete. Third-pass extrapolation is a diagnostic; do not deploy it merely because it raises one retrospective score.

## 9. Exact code entry contracts for integration

Available factory and optimizer examples:

```python
from neuralps_v2.phase2_training import (
    load_config, build_bundle, build_optimizer,
    copy_variant, initialize_residue_branch,
)
cfg = load_config('configs/phase2_final_config.json')
a = build_bundle(cfg, variant='A', input_variant='pool')
optimizer, names = build_optimizer(a, 'A_pool', lr=3e-4)

# After loading a trained A_pool checkpoint into a:
a_r = build_bundle(cfg, variant='A', input_variant='residue')
initialize_residue_branch(a, a_r)
optimizer, names = build_optimizer(
    a_r, 'A_residue_joint', lr=1e-4,
    lr_by_prefix={'residue_encoder':3e-4, 'decoder':3e-4,
                  'span_decoder':3e-4, 'encoder.input.raw_gate':3e-4})
# After training a_r or loading its trained checkpoint:
c_r = build_bundle(cfg, variant='C', input_variant='residue')
copy_variant(a_r, c_r)
optimizer, names = build_optimizer(
    c_r, 'C_joint', lr=3e-5,
    lr_by_prefix={'encoder.global_blocks':1e-4,
                  'encoder.global_slots':1e-4, 'decoder':1e-4})
```

The example initializes new bundles; load the intended trained checkpoint before inheritance. It is not an instruction to discard trained A weights. The reference constructor deliberately rejects unsupported architectural settings instead of silently ignoring config changes.

For pooled SSL, the adapter returns `batch`, `teacher` and `primary_target`. Supply a matching `mean` tensor [B,T,3,1152] and the current stage optimizer to `ssl_step`. Rank records contain `query_rows` [Q], `token_indices`/`slot_indices` [Q,S], `candidates` [Q,K,S,1152], `mean` [Q,S,1152], and boolean `positive`/`valid` [Q,K]. One query denotes one relation, with S=1 or S=2 ordered terminal slots. Queries refer only to primary masked targets.

The reference step makes exactly one optimizer update. For production accumulation, separate loss construction from zero_grad/backward/step, aggregate assembly sums with correct denominators across microbatches/ranks, and perform one optimizer step per intended effective batch. Calling the existing step four times does not implement gradient accumulation. DDP, sampler resumption, scheduler and BF16 autocast remain explicit production integration tasks.

Activity batches provide aligned labels outside the graph. T-domain regression uses its own head/loss rather than the binary `activity_step`. Source adapters require their own declared wrapper. The generic bundle factory does not implement source-route conditioning or T-domain background construction.

## 10. Runtime and behavior gates

| Gate | Check | Required action on failure |
|---|---|---|
| G0 | Package syntax, checksum and dependency checks | Repair local code/environment before GPU allocation |
| G1 | Record topology, counts, feature identities and label joins | Repair exporter; preserve accepted source data |
| G2 | Held-donor and natural split exclusion | Regenerate manifests with recorded rules |
| G3 | Masked/aliased inputs cannot influence student; enough context survives | Recompute views or revise extraction/task pairing |
| G4 | Active-branch gradients, current-forward ranking, optimizer groups | Repair branch wiring before long runs |
| G5 | A/C/D initialization identity; D first-pass supervision | Repair inheritance/recycle implementation |
| G6 | FP32 stability, BF16 agreement, largest-graph memory and padding | Fix numerics or batching; do not silently drop long routes |
| G7 | Matched-label transfer and complete fold reporting | Retain useful scoped result or stop promotion |
| G8 | Full candidate rebuild and factor-sum/table invariance | Mark unsupported candidates; repair feature regeneration |

Gradient and numerical tests in the package are required gates but have not run here without PyTorch. In the target environment add representative real assemblies: covalent-only, break-containing, long-gap, TE-terminal, edited-domain and duplicated-source cases. Do not substitute random-shape tests for the data handshake.

## 11. Reporting, promotion and release

The minimum convincing foundation panel contains: ESM-C baseline, identical architecture without NRPS SSL, A foundation, C foundation, matched label learning curves, Bode-2 retrospective transfer, and T-domain quantitative transfer. Report TE/position/source-aware comparisons in their declared input track. Keep raw predictions and all folds, including failures and unevaluable strata.

Practical initial targets are +0.02 macro donor AUROC and +0.05 T-domain Spearman under matched inputs and label budgets, assessed with paired uncertainty and fold consistency. These are decision thresholds chosen for the programme, not claims of statistical significance. A favorable natural retrieval score cannot substitute for engineering transfer. D needs its own compute-controlled benefit.

Natural checkpoint selection uses natural dev. Task-specific selection occurs inside task training folds. Report where supervised labels influence checkpoint choice. Source-enriched gains are described as gains from the combined inputs/model, with a matched-input baseline. Keep Bode-2 locked after this protocol; newly obtained libraries offer prospective evidence.

Release artifacts: selected natural-only checkpoint(s), separately adapted task checkpoints, feature/crop/vocabulary versions, split and exposure policies, raw prediction tables, metrics with uncertainty, source-coverage reports, model card, and candidate-table semantics. Biological datasets are shared only under their existing project terms; no external publication or transmission is authorized by this execution plan.

## 12. What is complete in this package and what is next

The updated design, this execution plan, experiment registry, parameter configuration, canonical schema, reference components and local verification are delivered together. The Python package is a reference implementation with explicit missing integrations. The E0 current-manifest export and target-environment runtime gate are the first production work. Once those pass, E1 and F1 can progress without another architectural redesign.

The original reference verification did not include the current production manifests/caches, PyTorch execution, GPU training or reconstructed full T-domain assay backgrounds. See `reports/repository_verification.json` for the repository-specific checks. The plan identifies exact required outputs and interfaces so those limitations do not become ambiguous claims of completed training.
