# Roadmap

**Focus: a science of deep learning.** Find the laws by which training creates the computation inside a model, by joining learning mechanics (the dynamics of training) with mechanistic interpretability (what training produces). Start on the cleanest system, small models pretrained on designed data, and climb a ladder that changes one thing at a time.

**Ground truth first.** A law or a method is trusted only where the answer is known and the controls behave:

| Check | From |
|---|---|
| Known answer | Skills and concepts planted in the training data at chosen frequencies and steps |
| Theory | A prediction written down before the run (for example from solvable models of deep linear networks) |
| Robustness | Several sizes, seeds and initialisation scales; no law from one run |
| Controls | Shuffled frequencies, planted concepts that must be found, environments that cannot be hacked |
| Causation | Intervene during training: remove or inject a skill's data, or project out its direction, and the predicted change must follow |
| Cost | FLOPs and GPU-seconds of every experiment, reported |

**The lab.** Explorers is the open lab the science runs in, with a Tinker-like API: train a model, read it and change it in the same loop, and run experiments at any scale without thinking about infrastructure. Environments follow Prime Intellect's format (data in a recorded order plus a rubric); Modal is the first backend ([README](README.md#the-lab)). We build no production trainer and no inference engine; trainers and serving stacks elsewhere are read through thin adapters, Prime Intellect's first.

Each step has a deliverable and a condition that says when it is done. There are no dates.

Status: **done**, **next**, **planned**, **later**.

1. **O0 — Foundations** · done
2. **O1 — The lab: train, read and intervene in one loop** · next
3. **O2 — The first experiments: the prior across training, and laws in controlled pretraining** · next
4. **O3 — Rung 2: natural data** · planned
5. **O4 — Rung 3: fine-tuning** · planned
6. **O5 — Rung 4: reinforcement learning and reward hacking** · started
7. **O6 — The lab at scale** · planned
8. **Later** — deployment, a second backend, whole-space signatures

## O0 — Foundations · done

- **One library.** `explorers` is one package with six concepts: models with named streams, traces, ops (interventions as data), measures, studies, and a store keyed by content. The design fits on [one page](https://github.com/machine-exploration/explorers/blob/main/docs/interface.md).
- **Streams: read, write, trace.** `residual`, `attn_out` and `mlp_out` on GPT-NeoX, Llama, GPT-2 and hybrid Qwen 3.5+ layouts. Reads equal what hand-written hooks return; a write changes exactly its target.
- **The study.** Reads, writes, measures and patching (exact and attribution) over models or the checkpoints of a run, cached by content. Five canonical examples (a linear probe, steering, activation patching, attribution patching, a sparse autoencoder read) are checked against results that are exact by construction.
- **The Jacobian lens.** Checked against the [reference implementation](https://github.com/anthropics/jacobian-lens): `J` agrees within 1.2e-7, and the top-5 readouts are identical at 54/54 (layer, position) pairs ([docs](https://github.com/machine-exploration/explorers/blob/main/docs/jlens.md)). The concept-targeted lens (one backward pass per word, whatever the model's width) is built as a measure, with the logit lens on the same words as its baseline ([docs](https://github.com/machine-exploration/explorers/blob/main/docs/monitors.md)).
- **Runs as checkpoints.** A prime-rl LoRA run's adapters are archived as they land and merged at load; merged equals unmerged ([docs](https://github.com/machine-exploration/explorers/blob/main/docs/prime.md)).
- **Episode replay.** Rollouts of an eval or an RL run become examples, token for token. Check on Qwen 3.8 27B: replayed log-probabilities of sampled tokens differ from those vLLM recorded by a median of 0.0003 (max 0.32, bf16) ([docs](https://github.com/machine-exploration/explorers/blob/main/docs/episodes.md)).
- **A reproduced result.** On a toy with 16 lookup tasks of Zipf frequencies, frequent tasks are learned first (rank correlation −0.74), each suddenly (median sharpness 0.74).

## O1 — The lab: train, read and intervene in one loop · next

- **Primitives:** `model` (a config or a Hugging Face name, a seed, an initialisation scale), `forward_backward(batch, reads=, do=)` (loss, per-example losses, requested internals, interventions applied, one pass), `optim_step`, `sample`, `save`/`load`. Checked exactly against hand-written PyTorch on tiny models.
- **`@experiment`:** one function over a grid of sizes, seeds, initialisation scales or checkpoints, run in parallel.
- **Backends:** Local and Modal. Small models ship the whole training loop to the backend; large ones take one call per primitive on a resident model.
- **Environments:** planted pretraining tasks in Prime Intellect's environment format (data in a recorded order plus a rubric), read through an adapter so the core does not import verifiers.
- **`stream`:** one model's activations fed to another training loop without storing them.
- **Fast iteration:** `cache()` for activations an experiment reuses (on a Modal Volume, keyed by content); warm containers during a session; only changed grid points recomputed; results streamed back while runs go; a smoke mode on a sliver of the data; automatic checkpoints and resume for long jobs.
- **Done when:** the same experiment gives the same result on Local and Modal within a stated tolerance, and the first experiment of O2 runs end to end.

## O2 — The first experiments: the prior across training, and laws in controlled pretraining · next

Experiments live in [mechanics](https://github.com/machine-exploration/mechanics) and call the explorers API.

- **The prior across training ([`glp-activation`](https://github.com/machine-exploration/mechanics/tree/main/experiments/glp-activation)), first:** reproduce the generative meta-models of activations of [Luo et al., 2026](https://arxiv.org/abs/2602.06964) on their Llama 1B model and match their released priors; then fit a prior at every checkpoint of a small organism (Pythia follows in O3). When does the distribution of internal states acquire its structure, and does it line up with when skills are learned? Hypotheses are fixed before the runs.

Then rung 1 proper: small transformers pretrained from scratch on designed data, so the answer is known and solvable theory applies.

- **The onset law:** skills planted at frequencies spanning two orders of magnitude; 3 sizes × 5 seeds × 2 initialisation scales. Prediction, written before the run: onset step inversely proportional to frequency, with a logarithmic dependence on initialisation scale, as in deep linear networks. Control: shuffled frequencies.
- **Representation before use:** at every checkpoint, when each skill's concept becomes linearly readable, against when its accuracy jumps.
- **Universality:** whether the order of learning and the concept directions match across seeds.
- **Causation:** project out a skill's direction during training (`do=`), or move its data: the predicted change in onset must follow.
- **Done when:** each result is public with the run that reproduces it, whether the prediction held or failed.

## O3 — Rung 2: natural data · planned

- **The laws on Pythia:** every checkpoint, known data order; frequencies estimated from the data. Do the rung-1 laws survive natural data?
- **The verbalizable space:** when the space read by the Jacobian lens forms during pretraining, how suddenly, at which depths and from which size, with its four signatures ([experiment](https://github.com/machine-exploration/mechanics/tree/main/experiments/q1_verbalizable_space)).
- **The prior on natural data:** `glp-activation` on Pythia's checkpoints, set against the onsets found on rung 1.
- **Done when:** each question has a published answer with its run.

## O4 — Rung 3: fine-tuning · planned

- **A planted skill fine-tuned into a pretrained model:** does the onset law survive a structured initialisation?
- **Lazy or rich:** does fine-tuning stay close to the linearised regime, where theory is most predictive, or form new features? How does that depend on the learning rate, the rank of the adapter and the model's size?

## O5 — Rung 4: reinforcement learning and reward hacking · started

The data now depends on the model, so cause and effect are entangled; the laws from the lower rungs are what make this rung readable.

- **Started:** [`impossible_code`](https://github.com/machine-exploration/verifiers/tree/main/environments/impossible_code), ImpossibleBench as a verifiers environment (its tasks cannot be solved honestly, so a pass is a hack); first rollouts of Qwen 3.8 27B, with hacks; episode replay checked on the same model.
- **Watching a model learn to cheat:** RL with every step's adapter kept, the same evaluation set read at every checkpoint. Does the representation of hacking form before the hack rate rises? Does RL build a new mechanism or route an existing concept? Controls: a planted concept, an environment that cannot be hacked.
- **The Monitor Arena:** methods compared on the same rollouts: a difference-of-means probe ([Goodfire, 2026](https://arxiv.org/abs/2609.19101)), a concept Jacobian-lens direction built from words, the logit lens; detection at 1% and 5% false-positive rate, cost, scale.
- **Done when:** the figure (hack rate and internal signal over training steps, with both controls) and the first Arena track are public with their runs.

## O6 — The lab at scale · planned

- **Large models:** 32–70B across several GPUs; bf16 forward, fp32 accumulation.
- **Resident models:** one base loaded once and shared across a run's checkpoints and across users; adapters swapped or batched.
- **A planner:** forward passes shared between methods, projections fused into one matrix product, reductions on the GPU. Done when it is at least 3× faster on a real experiment with unchanged results, measured against NNsight and TransformerLens on the same experiment.
- **Inline in other trainers:** the same primitives inside torchtitan or Megatron, for pretraining runs too large to replay.

## Later

- **Deployment:** thin taps in serving engines, for monitors that passed the Arena.
- **A second backend** (NNsight or TransformerLens) giving the same results within a stated tolerance.
- **Whole-space signatures at large scale** through sketches, checked against the exact Jacobian on smaller models.

## Known risks

- **A law may not exist, or may not transfer.** The onset law could fail on rung 1, or hold on designed data and fail on natural data. Either result is published.
- **Toy models can mislead.** Each rung exists to test whether a rung-1 result survives a more realistic system.
- **A readable representation may need size.** Each model is checked first: does the expected structure appear at the final checkpoint?
- **Backend equivalence is hard:** numerics differ between hardware and tools; differences that cannot be closed are documented.
- **We depend on others' formats:** prime-rl's file layout and the verifiers environment format are read at pinned commits and tested, never imported into the core.
- **Existing tools are strong:** NNsight/NDIF and TransformerLens. Explorers must earn its place on experiments across training, not on a nicer hook API.
