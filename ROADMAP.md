# Roadmap

Five phases, in order. Each step has a question, a deliverable, and a condition that says when it is done. There are no dates.

1. **The primitive** — Explorers: read, write and trace, and the study as the unit of work.
2. **The scaling proof** — one expensive experiment, made much cheaper by planning its execution.
3. **The Runtime** — the same study, executed at scale.
4. **Mechanics** — original science on how training creates computation.
5. **Training integration** — the same studies, running alongside training.

Status: **done**, **in review**, **next**, **planned**, **later**.

## Phase 1 — The primitive

The criterion is not the number of methods. It is: can researchers express probes, steering, patching, attribution and sparse autoencoders naturally, as compositions over the same execution model?

### Step 0 — One library · done

- **Result:** The `explorers` repository is one `uv` workspace with `explorers.core`, `explorers.learning` and `explorers.populations`, and the history of both former repositories. The three packages install and import together (before, the prototype's `explorers/__init__.py` hid `explorers.core`). 76 tests pass from a fresh clone. `mechanics` holds research only and installs `explorers` from GitHub; its toy quanta run reproduces the result.

### E1 — Streams: read, write, trace · done

- **Deliverable:** Named streams (`residual`, `attn_out`, `mlp_out`) with locations (layer, position), on Hugging Face / PyTorch models:

  ```python
  model = ex.open("EleutherAI/pythia-70m")
  with model.trace(prompt) as run:
      resid = run.stream("residual")
      x = resid.read(layer=3, position=-1)
      resid.write(layer=3, position=-1, fn=lambda h: h + direction)
  ```
- **Done when:** Reads equal what hand-written hooks return; a write changes exactly the targeted location; a stable name for a site holds across checkpoints of one model.

### E2 — The study · done

- **Deliverable:** `Study`: reads, writes (steering, ablation, patching) and measurements over models × checkpoints × examples, compiled to the existing engine (one forward pass per state, results stored by content).
- **Done when:** Five canonical examples run as studies, each checked against a known result: a linear probe, steering along a direction, activation patching, attribution patching, and a sparse autoencoder read.

- **Result (E1, E2):** `ex.open`, `model.trace`, streams `residual` / `attn_out` / `mlp_out` on GPT-NeoX, Llama and GPT-2 layouts; `Study` with reads, writes, measures and patching (exact and attribution) over models or the checkpoints of a run. The five examples are checked against results that are exact by construction on any weights ([design](https://github.com/machine-exploration/explorers/blob/main/docs/interface.md)). The library is one package, `explorers`.

### E3 — A second backend · next

- **Deliverable:** The same studies on a second backend (NNsight or TransformerLens).
- **Done when:** Every E2 example gives the same result on both backends within a stated tolerance. Differences that cannot be closed are documented, and that backend does not support those operations.

### I1 — The Jacobian lens · done

- **Result:** Engine reads for gradients and the pre-norm final residual; observables `jacobian`, `jlens_error` and the `logit_lens_error` baseline. On a random 3-layer GPT-NeoX, `J` matches the [reference implementation](https://github.com/anthropics/jacobian-lens) within 1.2e-7, and the top-5 readouts are identical at 54/54 (layer, position) pairs ([docs](https://github.com/machine-exploration/explorers/blob/main/docs/jlens.md)). In `explorers.core` today; moves to `explorers.lenses` with E2.

## Phase 2 — The scaling proof

### S1 — The benchmark · planned

- **Done before S1:** reads keep only their selection or a reduction computed on the device; interventions, reductions and metrics are data (`explorers.ops`); a study serializes to JSON and has a content key. So the benchmark measures patching itself, not avoidable copies.

- **Question:** How expensive is exhaustive activation patching across training, with the tools researchers use today?
- **Deliverable:** One study: every residual-stream patch (all layers × all positions) for a clean/corrupt task with a known mechanism, across checkpoints of an open model suite. First scale: Pythia 70m, 10 checkpoints, 1,000 example pairs. Baselines: a naive loop, NNsight sessions, TransformerLens, on the same hardware.
- **Done when:** Wall time and GPU-hours of each baseline are published, with identical results.

### S2 — The planner · planned

- **Deliverable:** An execution plan for the S1 study: shared computation up to each intervention point, then branching; source activations computed once; counterfactual branches batched; activation caching and deduplication; checkpoint loading amortised. As an option, attribution patching as an approximation, with its error measured against exact patching.
- **Done when:** The speed-up over the best baseline is measured and published, for identical results. Target: at least 5×. Below that, the query-engine thesis is revised.

## Phase 3 — The Runtime

### R1 — `study.compute(backend="machine-exploration")` · later

- **Open source,** like the interface.
- **Deliverable:** The same Python study, executed on many GPUs: scheduling, parallelism across examples, sites, checkpoints and devices, results in a shared content store. Researchers do not manage clusters.
- **Done when:** A study that takes days on one GPU runs in hours, and gives the same result as the local run.

## Phase 4 — Mechanics

Uses the instrumentation to ask how training creates computation. Not static circuit papers: a record of learned computation forming.

### Q0 — Quanta across training · done on a toy, next on Pythia

- **Done:** On a toy with 16 lookup tasks of Zipf frequencies, frequent tasks are learned first (rank correlation −0.74), and each is learned suddenly (median sharpness 0.74).
- **Next:** The same analysis on Pythia checkpoints, 70m to 1.4b: the timeline of skills the other questions are compared with.

### Q1 — When does the verbalizable space form? · next *(first flagship)*

- **Question:** Across training, when does the space read by the Jacobian lens appear, how suddenly, at which layers, and does it form before, with or after the skills of Q0?
- **Deliverable:** The lens fitted at about 24 log-spaced Pythia checkpoints (70m to 410m, one 12 GB GPU). At each checkpoint and layer: `jlens_error` against the `logit_lens_error` baseline, and readouts of concepts implied but absent from the prompt. Onsets and sharpness next to the Q0 timeline.
- **First check:** Does the finished-model result hold on the final Pythia checkpoints?
- **Done when:** A result note in this repository, with the run that reproduces it. Positive or negative.

### M1 — From represented to used · planned

- **Question:** When does a feature that a probe can read start to be used? At which checkpoint does an intervention on it first change behaviour?
- **Needs:** E2 (probe and steering as studies).

### M2 — Circuit formation and replacement · planned

- **Question:** For a known mechanism (induction heads in Pythia), how does the circuit assemble across checkpoints, and is any mechanism replaced by another while the loss barely changes?
- **Needs:** S2 (exhaustive patching across checkpoints is exactly the expensive study).

### Deployment questions · planned

- **P0 — Do small models exploit tasks?** Models of 0.6B to 4B on impossible tasks, alone and with a shared channel; 200 episodes per model; 50 labels checked by hand.
- **P1 — Detectors at a matched false-positive rate** · next: detection rate at 1 %, 5 % and 10 % false-positive rate with intervals from a bootstrap over episodes, and a token range for each turn. A first version ([explorers#1](https://github.com/machine-exploration/explorers/pull/1)) was closed without merge; redo it in `explorers.populations`.
- **Q2 — What does an agent hold, turn by turn?** Does the plan to exploit a task appear in the verbalizable space before the action?
- **Q3 — Disposed to say, compared with said:** is the gap a signal for exploits and deception, against a chain-of-thought monitor and an LLM judge at the same false-positive rate?

## Phase 5 — Training integration · later

- **Deliverable:** `watch(training_run, every=N, observables=[...])`: white-box studies running alongside a training run, next to the loss, the reward and the evals. First target: post-training runs whose checkpoints are LoRA adapters (small, so dense checkpoints are cheap).
- **Question it answers first:** during RL on exploitable tasks, does the exploit plan appear inside the model before the behaviour shows, and does training against a monitor move it?

## Later: the survey

Families of models trained with a full record (dense checkpoints, data order, several seeds and sizes, controlled changes to the data), published with their stores of measurements as an open dataset for Mechanics and for others.

## Known risks

- **The speed-up may be small.** Sharing computation before an intervention point saves at most about half of an exact patching sweep over layers; larger gains must come from batching, caching, deduplication and approximation. S1 and S2 measure this before the Runtime is built.
- **Backend equivalence is hard:** site names, write support (vLLM batching, paged KV, CUDA graphs) and numerics differ. E3 checks it; unsupported operations are documented.
- **Existing tools are strong:** NNsight/NDIF (free for researchers) and TransformerLens 4.0 (with vLLM). Explorers must win on the study abstraction and on measured execution, not on a nicer hook API.
- **The finished-model result about a verbalizable space may not hold on Pythia.** Q1 checks it first; a negative is still a result.
- Hugging Face must be reachable from the machine that runs Q0 and Q1. Small models may not exploit tasks (P0 checks this before Q2).
