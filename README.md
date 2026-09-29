# Machine Exploration

> **We measure what happens inside models over time**, to understand how intelligence arises in deep learning and to oversee the AI systems it produces.

Everything here is open: code, formats, runs and results. **Status: pre-alpha.** The roadmap is in [ROADMAP.md](ROADMAP.md).

---

## Mission

- **Long term: a science of deep learning.** How training builds the mechanisms a model computes with, studied at the join of learning mechanics and mechanistic interpretability ([Simon et al., 2026](https://arxiv.org/abs/2604.21691)).
- **First product: training observability for RL post-training.** See what a model learns while it trains, and get a warning when a bad behaviour starts to form inside it, before the evals show it.
- **Now: white-box oversight of agent populations.** Agents run in populations that share tools, caches and channels, and they fail together ([METR](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/), [Redwood Research](https://www.redwoodresearch.org/research/hugging-face-incident)). Transcripts can be spoofed, so oversight has to read the models, not only what they write.

## One idea: two clocks

A model changes along two clocks.

| Clock | What changes | Field | Our program |
|---|---|---|---|
| **Training time** | Weights change from checkpoint to checkpoint. The model learns features, circuits and skills. | Learning mechanics, mechanistic interpretability | **Learning** |
| **Interaction time** | Deployed agents act over many turns, together. What they do, and what is active inside them, changes turn by turn. | Oversight | **Populations** |

Both are the same measurement problem: a model state, a fixed set of examples, and pure functions that read weights, losses or activations. So one core serves both. It has five primitives:

- **Examples:** token sequences with metadata, identified by their content.
- **State:** a model at one point of training, loaded when needed.
- **Observable:** a pure, versioned function of a state and the examples. It declares what it reads (`weights`, `token_loss`, `hidden:<layer>`, `step`).
- **Engine:** measures observables along a run, with one forward pass per state however many observables ask.
- **Store:** results keyed by content, so reruns skip finished work.

A checkpoint series is a run along training time. An agent episode, turned into examples, is data along interaction time. The same observable runs on both.

## Where the clocks meet: monitors

A white-box monitor is an observable: a probe that reads activations and signals a behaviour, such as hacking or coordination.

- **Along interaction time,** it is oversight. Does it catch the behaviour, and earlier or better than a text monitor at the same false-positive rate?
- **Along training time,** it is science. When does the feature it reads form? Does the probe still work on the next checkpoint? Does training against the monitor hide the feature?

Every monitor also rests on assumptions from interpretability: that the feature is a direction (linear representability), that it lives at a few sites (locality), that it is active on few inputs (sparsity), and that complex features are built from simpler ones (compositionality). We audit these assumptions for each monitored feature, along both clocks.

Learning mechanics gives the hypotheses. We test them on features that matter for oversight:

| Prediction from learning mechanics | [Open direction](https://learningmechanics.pub/openquestions/) | Hypothesis about a monitor |
|---|---|---|
| Learning proceeds as a sequence of units (quanta) | 11 | **Formation:** the monitored feature forms at a discrete step, before the behaviour shows. |
| Weight updates are approximately low-rank | 12 | **Drift:** a large update in the layer a probe reads predicts a drop in the probe's accuracy. |
| Optimisation is biased toward simple solutions | 3 | **Obfuscation:** under pressure against a monitor, the feature moves along the cheapest direction. That predicts which monitors survive. |
| Features have a formal definition | 4 | **Audits:** which of linearity, locality, sparsity and compositionality hold, and when they stop holding. |
| Models trained differently learn similar representations | 10 | **Transfer:** a probe trained on one model works on another. |

One hypothesis joins it all: **obfuscation is a change in which assumption holds.** A feature that is linear and local early in training can become distributed under pressure against a linear probe.

## Programs

| Program | Clock | First question | Where it stands |
|---|---|---|---|
| **Training observability** | training | During RL, does a probe for a bad behaviour move before the eval rate does? | First product line. Core primitives exist; reading hosted LoRA checkpoints and `watch` are next. |
| **Populations** | interaction | Does anything inside agents signal hacking and coordination before it shows in behaviour, better than text monitors at the same false-positive rate? | Scenario format, runtime and labels exist. A detector harness at matched false-positive rate is in review. |
| **Learning** | training | When, and in what order, does a model learn what it learns? First study: quanta. | Core primitives and a toy result exist: frequent tasks are learned first (rank correlation −0.74), and suddenly. Pythia is next. |
| **The join** | both | Monitors under training: formation, drift, obfuscation, transfer. | Planned. Needs both programs on one core. |

## Repositories

| Repository | What it holds |
|---|---|
| [explorers](https://github.com/machine-exploration/explorers) | The populations prototype: scenario format, multi-agent runtime, episodes, labels, detectors. The home of the unified `explorers` library. |
| [mechanics](https://github.com/machine-exploration/mechanics) | The shared core (`explorers-core`), the learning package (`explorers-learning`: Pythia checkpoints, toy tasks, probes, the quanta study), and a placeholder for `explorers-populations`. Moves into `explorers` (roadmap step 0). |
| [public](https://github.com/machine-exploration/public) | This page, the roadmap, and later research notes and results. |
| [verifiers](https://github.com/machine-exploration/verifiers), [vllm](https://github.com/machine-exploration/vllm) | Pinned forks of upstream projects used as backends. No local changes. |

A note on names: the library is `explorers`, with the packages `explorers.core`, `explorers.learning` and `explorers.populations`. "Mechanics" is the name of the science, not of a package.

## Principles

- **Open by default:** code, formats, runs and results. Negative results are published too.
- **Reproducible:** every result replays from its config, seed, data order, model version and code version.
- **Against baselines:** a white-box monitor is compared with chain-of-thought monitors and LLM judges at the same false-positive rate. A training signal is compared with loss, weight norm and gradient norm, and with simply re-evaluating the probe on labelled data.
- **Causal, not only correlational:** a claim that a feature exists is tested by intervention (steering, ablation, patching), not only by a probe that correlates.
- **Ground truth first:** a method is validated where the answer is known (toy tasks, labelled scenarios) before it is used where it is not.
- **Small before large:** a question is settled on a model that fits one GPU before it is asked of a large one.
- **Contained:** scenarios that push agents to hack run with no network, no shared cache and no path between episodes. Publication rules for such scenarios and their traces are settled before release.
- **Audit isolation:** oversight monitors run outside the trainer's process and write to an append-only store that the reward code cannot read.

## Contributing

The most useful contributions right now are conversations. If you train or evaluate agents in populations, build methods to read models, or work on learning mechanics, open an issue in the repository that fits.

## References

- Simon et al., *There Will Be a Scientific Theory of Deep Learning*, [arXiv:2604.21691](https://arxiv.org/abs/2604.21691), and the [open directions](https://learningmechanics.pub/openquestions/)
- Michaud et al., *The Quantization Model of Neural Scaling*, [arXiv:2303.13506](https://arxiv.org/abs/2303.13506)
- Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability*, [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Olsson et al., *In-context Learning and Induction Heads*, [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
- Hoogland et al., *The Developmental Landscape of In-Context Learning*, [arXiv:2402.02364](https://arxiv.org/abs/2402.02364)
- Biderman et al., *Pythia*, [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)
- Goodfire, *Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations*, [arXiv:2609.19101](https://arxiv.org/abs/2609.19101)
- Taufeeque et al., *The Obfuscation Atlas*, [arXiv:2602.15515](https://arxiv.org/abs/2602.15515)
- Gupta & Jenner, *RL-Obfuscation*, [arXiv:2506.14261](https://arxiv.org/abs/2506.14261)
- METR and Redwood Research, investigation of the OpenAI / Hugging Face hacking incident (2026): [METR](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/) · [Redwood Research](https://www.redwoodresearch.org/research/hugging-face-incident)
