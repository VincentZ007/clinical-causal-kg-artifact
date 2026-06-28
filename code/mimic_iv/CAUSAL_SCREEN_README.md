# VC-RAG - causal-inference layer (newly added code)

Code added on top of the base VC-RAG project: a confounder-adjusted causal screen,
its integration into causal-RAG as a recall-preserving demotion layer, stronger
text-RAG baselines, and the evaluation/analysis scripts. **No patient data included**
(MIMIC-IV DUA); every script regenerates its outputs from MIMIC + the base pipeline.

## Causal-inference layer
- `causal_inference.py` - per-edge causal screen on MIMIC structured diagnoses:
  new-user / incident-outcome cohort design (direction by design), stabilized IPW
  over age/sex/baseline-comorbidity, negative-control calibration -> calibrated z,
  and E-values. Parallel (multiprocessing). Output: causal_screen_train.json.
- `make_causal_fig.py` - lift vs calibrated causal-z scatter (figs/causal_vs_lift.png):
  shows ~84% of lift-supported edges fail the causal screen (lift over-credits).
- `causal_screen_coverage.py` - coverage + demotion-impact table: how the top-12
  retrieved candidates split into pass / fail / untested, and how demotion reorders
  them (mean rank before vs after; % of questions whose evidence order changes).

## Causal-RAG integration (System 6)
- `vcrag_causal_prompts.py` - builds the System-6 prompts.
  Flags: `--base {lifttemporal, liftonly}` (liftonly = honest base, temporal dropped),
  `--mode {lift, rerank, annotate, demote}`. DEMOTE (recommended): keep lift order
  for pass/untested candidates, move causal-FAIL candidates to the end of the same
  top-12 -> recall preserved by construction; halves hallucination at equal accuracy.

## Generation
- `gen_local.py` - local fp16 batched generation (Qwen2.5-7B-Instruct, transformers).
  Usage: `python gen_local.py prompts.jsonl answers.jsonl [batch]`.
- `gen_causal.sbatch` - HPC (AWQ + vLLM) sbatch for the System-6 prompts.

## Stronger text-RAG baselines (ablation: can text-RAG win without a KG?)
- `textrag_fair_prompts.py` - TF-IDF retrieval but keeps only genuine causal-cue
  sentences about the anchor (less noisy than the original any-mention text-RAG).
- `textrag_dense_prompts.py` - dense semantic retrieval (all-MiniLM-L6-v2) + causal
  sentences. Conclusion: all three text-RAGs (any-mention / causal / dense) land
  ~13-14%, below closed-book (~15%); structured KG ~doubles it (~36%). Retrieval
  quality is not the bottleneck - patient-specific causal text misleads.

## Evaluation / analysis
- `eval_local.py` - unvalidated / vcrag / vcrag_causal, same fp16 generator.
- `eval_all.py` - adds the rerank/annotate/demote variants.
- `eval_liftonly.py` - lift-only base vs lift-only+demote (the controlled main comparison).
- `eval_ladder.py` - full ladder: closed-book / text-RAG variants / VC-RAG / +demote.
- `diag_whatcauses.py` - per-type McNemar + dumps WHATCAUSES regressions (diagnosed
  the rerank position artifact that motivated DEMOTE).

## Headline result (local controlled fp16 run, n=932)
| system | WHY | WHATC | OVERALL | FAITH | HALLUC | RECALL |
|---|---|---|---|---|---|---|
| VC-RAG lift-only (honest base) | 46.4 | 25.1 | 35.7 | 82.7 | 17.3 | 86.3 |
| + causal DEMOTE | 48.5 | 26.0 | 37.2 | 93.9 | 6.1 | 86.3 |

Every correctness axis is matched-or-up; hallucination cut ~3x at fixed recall.
Overall +1.5pt is a positive trend (McNemar p=0.27, n.s.); the robust claim is the
hallucination reduction.

## Run order (after the base KG pipeline + MIMIC access)
1. `causal_inference.py --edges supported_train_edges.tsv --use_split train ...` -> causal_screen_train.json
2. `make_causal_fig.py` ; `causal_screen_coverage.py`
3. `vcrag_causal_prompts.py --base liftonly --mode lift|demote ...` -> prompts
4. `gen_local.py prompts_*.jsonl answers_*.jsonl` (or gen_causal.sbatch on HPC)
5. `eval_liftonly.py` / `eval_ladder.py`
