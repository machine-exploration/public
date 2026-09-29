# Roadmap

The work runs as two programs on one core, then joins them. Each step has a question, a deliverable, and a condition that says when it is done. There are no dates.

- **Populations** (interaction time) has priority for the GPU and for attention.
- **Learning** (training time) uses public checkpoints and small toy runs, so it costs little. It also builds the tools the join needs: runs of checkpoints, observables across them, and assumption audits on features with a known answer.

Status: **done**, **in review**, **next**, **planned**.

## Step 0 — One library · next

- **Problem:** Two repositories claim the `explorers` namespace. The prototype in `explorers` is a regular package (`explorers/__init__.py`). Installed next to the workspace in `mechanics`, it hides `explorers.core`, and `import explorers.core` fails.
- **Deliverable:** The `explorers` repository becomes the workspace, with its history kept:
  - `packages/core`, `packages/learning`: moved from `mechanics`;
  - `packages/populations`: the prototype, ported from `explorers.<module>` to `explorers.populations.<module>`;
  - an adapter from episodes to core `Examples`, with one example per (agent, turn), so any core observable runs on agent episodes;
  - one set of detection metrics (today in two places: `explorers-core` metrics and the populations detector harness).
- **Before:** Land the detector harness ([explorers#1](https://github.com/machine-exploration/explorers/pull/1)) so it moves with the rest.
- **Done when:** One `uv sync` installs everything, all tests pass, a core observable (for example `hidden_norm`) runs on agent episodes, and `mechanics` points to `explorers`.

## Populations — interaction time

### P0 — Do small models hack and coordinate? · next

- **Question:** Do models that fit one GPU (0.6B to 4B) hack and coordinate in the first scenario (the Hugging Face incident in miniature)?
- **Deliverable:** Hack rate, propagation rate and board activity for each model, with intervals, on 200 episodes per model. 50 labels checked by hand.
- **Done when:** One model hacks often enough to train a probe on. If none does, use larger open models on rented GPUs, or a single-agent impossible-task setting first.
- **Exists:** scenario format v0, runtime through verifiers, `hack_label`, `propagation`, the report.

### P1 — Detectors at a matched false-positive rate · in review

- **Deliverable:** Detection rate at 1 %, 5 % and 10 % false-positive rate, AUROC, and confidence intervals from a bootstrap over episodes (not rollouts, because agents that share a board are correlated). Spans carry a token range, so a probe can read exactly the tokens of one turn.
- **Where:** [explorers#1](https://github.com/machine-exploration/explorers/pull/1).
- **Next after it:** thresholds fit on held-out episodes, not in-sample.

### P2 — Read activations for episodes · planned

- **Deliverable:** The core reads `hidden:<layer>` for episode examples: same checkpoint, one prefill pass over the recorded token ids, only the positions in each span's token range. Long agent contexts may need the sidecar scorer ([desk spikes](https://github.com/machine-exploration/explorers/blob/main/docs/desk-spikes-2026-09-27.md)).
- **Done when:** The prefill reproduces the log-probs of the sampled tokens within numerical tolerance.

### P3 — v0.1: monitors at inference · planned

- **Deliverable:** On the same episodes: an LLM judge, a chain-of-thought monitor, a linear probe, a nonlinear probe, and a steering test along the probe direction. A judge replaces the phrase-echo heuristic for coordination. One assumption-audit table per monitored feature (linearity, locality, sparsity, compositionality), with the audits validated in L2.
- **Done when:** Results at 1 % and 5 % false-positive rate, with intervals, train and test split by task. Published whether positive or negative.

## Learning — training time

### L0 — Core primitives and quanta on a toy · done

- **Deliverable:** `explorers-core` (examples, states, observables, engine, store, analyses) and `explorers-learning` (Pythia checkpoints, toy tasks, probes).
- **Result:** On a toy with 16 lookup tasks of Zipf frequencies, frequent tasks are learned first (rank correlation between frequency and onset step −0.74), and each is learned suddenly (median sharpness 0.74).

### L1 — Quanta on Pythia · next

- **Question:** Along Pythia's pretraining, when is each token-level sample learned, how suddenly, and are frequent targets learned first ([Michaud et al.](https://arxiv.org/abs/2303.13506))?
- **Deliverable:** `examples/quanta_pythia.py` on 70m to 1.4b (these fit one 12 GB GPU in fp16): onsets, sharpness, clusters of samples that drop together, correlation with target frequency, across sizes.
- **Done when:** A result note in this repository, with the run that reproduces it.

### L2 — A known feature across training · next

- **Question:** When does a known feature (for example, number comparison) become linearly decodable, at which layer, and how suddenly?
- **Deliverable:** The existing probe sweep (checkpoints × layers × probes) rebuilt as core observables. The four assumption audits run on this feature at each checkpoint.
- **Why here:** The answer is simpler to check than for hacking. The audits are validated here before P3 uses them (ground truth first).

### L3 — Updates and probes · planned

- **Question:** Does a large update in the layer a probe reads predict a drop in the probe's accuracy (drift, open direction 12)?
- **Deliverable:** Update size and rank per layer between checkpoints (`update_norm`, `stable_rank`), next to probe accuracy from L2.
- **Baseline:** Re-evaluating the probe on labelled data at each checkpoint. An update signal has value only if it tells us something this baseline does not, or tells it earlier or without labels.

## The join — both clocks

### J1 — Monitors under training · planned

- **Question:** How does a monitored feature form, drift and hide during training? Is obfuscation a change in which assumption holds?
- **Deliverable:** The P3 monitors and audits run on agent episodes across a series of checkpoints. Two routes: open models with published post-training checkpoints that can play the scenario, if they exist; otherwise our own RL run on hackable tasks. A second run trains against the probe (obfuscation).
- **Done when:** Formation, drift and obfuscation each have a result against the label baseline.
- **Needs:** For our own RL run, two GPUs or a hosted trainer.

### J2 — Transfer · planned

- **Question:** Does a probe trained on one model work on another (other seed, other size), with or without a representation alignment (open direction 10)?

## Later

- A second scenario family (virtual marketplaces) and larger open-weight models.
- A public catalogue of phenomena for theorists: each entry with the run that reproduces it, its measurements, and the open question it poses. Phenomena from hacking scenarios follow the publication rules; phenomena from toy tasks and public checkpoints can be published first.
- A shared hub for scenarios, runs, observables and results.

## Known risks

- Small models may not hack or coordinate (P0 tests this first).
- The coordination label is a heuristic until a judge replaces it (P3).
- Reading activations for long agent contexts on one GPU may need the sidecar and streaming (P2).
- Hugging Face must be reachable from the machine that runs L1 and L2.
- Dense checkpoints are large. Keep only the layers that are read, or low-rank deltas, and full checkpoints at log-spaced steps.
- Our own RL run (J1) does not fit on one consumer GPU.
