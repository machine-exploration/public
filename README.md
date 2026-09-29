# Machine Exploration

> **A science of intelligence, built from the inside of models.**
> Models are the first intelligent systems we can fully record, change and replay through their whole training. We build the instruments, and we look for the laws of how intelligence forms.

Everything here is open: code, formats, runs and results. **Status: pre-alpha.** The plan is in [ROADMAP.md](ROADMAP.md).

---

## Why

Deep learning works far better than we can explain. We can train a model that writes code or plans across many steps, but we cannot say which computation it performs, when in training that computation appeared, or why training found it rather than another one.

Two fields attack this from opposite ends:

- **Mechanistic interpretability** reads a trained model: its features, its circuits, the algorithms its weights implement. It asks *what* the model computes.
- **Learning mechanics** treats training as a dynamical system. It asks *how* and *why* a computation forms ([Simon et al., 2026](https://arxiv.org/abs/2604.21691)).

Interpretability mostly studies one finished model, so it sees the result of learning, not the process. The theory of training mostly tracks scalars (loss, norms, curvature), so it sees the process without knowing what is being learned. We work at the join: measure what a model computes, at every point of its training, and explain how it got there.

## The long-term goal

> **Predict what a model will learn, and when, before we train it.**

Today we train a model and then find out what it learned. A science of intelligence would reverse that: from the data, the architecture, the optimiser and the scale, predict which capabilities and which behaviours will form, including the ones we do not want.

## Two clocks

A model changes along two clocks, and we measure both with the same tools.

| Clock | What changes | Example question |
|---|---|---|
| **Training time** | Weights change from checkpoint to checkpoint. The model learns features, circuits and skills. | When does a capability form, and how suddenly? |
| **Interaction time** | An agent acts over many turns, sometimes with other agents. What is active inside it changes turn by turn. | What does an agent hold internally before it acts? |

## Three phases

1. **The instrument.** [`explorers`](https://github.com/machine-exploration/explorers): an open library to measure what happens inside models over time, on both clocks. Anyone can add a measurement in a few lines.
2. **The survey.** Families of models trained with a full record: dense checkpoints, the data order, many seeds and sizes, controlled changes to the data, and agents trained with RL. Published as an open dataset of internals over training. Public checkpoint suites such as [Pythia](https://arxiv.org/abs/2304.01373) and [OLMo](https://arxiv.org/abs/2402.00838) are a start, but they were not designed for this.
3. **The laws.** From the survey: regularities that predict when a feature, a capability or a behaviour forms. Learning mechanics gives the hypotheses; the survey tests them.

## First questions

**1. When does the verbalizable space form during training?** *(flagship)*
Recent work finds that language models hold, in their middle layers, a shared space of content that the model is disposed to say, and reads it with the Jacobian lens (J-lens) ([Anthropic, 2026](https://transformer-circuits.pub/2026/workspace/); [code](https://github.com/anthropics/jacobian-lens)). That work studies finished models. We ask how this space develops: when it appears across training, how suddenly, at which layers, and whether it forms before, with or after the skills the model learns.

**2. What does an agent hold in that space, turn by turn?**
For example: does a plan to exploit a task appear inside the model before the action? Does content move from one agent to another through a shared channel?

**3. Does what a model is disposed to say differ from what it says?**
A gap between the two is a candidate internal signal for deception. It matters for oversight: transcripts can be spoofed, so oversight has to read the models, not only what they write ([METR](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/), [Redwood Research](https://www.redwoodresearch.org/research/hugging-face-incident)).

We describe these spaces only in functional terms (what is available for report, shared across layers and positions, limited in size). We make no claims about experience.

## The instrument

Every result in `explorers` has the same form: **an observable, measured on a run of model states, on a set of examples**, stored by content.

| Primitive | What it is |
|---|---|
| `Examples` | What we look at: texts, tasks, agent turns. Identified by their content. |
| `State` / `Trajectory` | A model at one point (a checkpoint, a LoRA adapter, a model in memory), and a run of them. |
| `Observable` | A pure, versioned measurement that declares what it reads (weights, losses, activations, gradients). |
| `over` / `across` | Measure along one run, or across sizes and seeds, with one forward pass per state. |
| `Store` | Results keyed by content, so reruns cost nothing and results from many machines merge. |
| `analysis` | Functions of stored curves only: when a curve changes (`onsets`), how suddenly, what correlates with it. |

Because every result has this form, results can be compared, replayed and combined. Published stores of many recorded runs become the survey.

## What exists today

| Repository | What it holds |
|---|---|
| [explorers](https://github.com/machine-exploration/explorers) | The agent side: scenario format, multi-agent runtime, episodes, labels, detectors. The home of the unified library. |
| [mechanics](https://github.com/machine-exploration/mechanics) | The shared core (`explorers-core`) and the training side (`explorers-learning`: Pythia checkpoints, toy tasks, probes). First result: on a toy with tasks of Zipf frequencies, frequent tasks are learned first (rank correlation −0.74), and each is learned suddenly. Moves into `explorers`. |
| [public](https://github.com/machine-exploration/public) | This page, the roadmap, and later research notes and results. |
| [verifiers](https://github.com/machine-exploration/verifiers), [vllm](https://github.com/machine-exploration/vllm) | Pinned forks of upstream projects used as backends. No local changes. |

## Principles

- **Open by default:** code, formats, runs and results. Negative results are published too.
- **Reproducible:** every result replays from its config, seed, data order, model version and code version.
- **Causal, not only correlational:** a claim that a feature exists is tested by intervention (steering, ablation, patching), not only by a readout that correlates.
- **Ground truth first:** a method is validated where the answer is known (toy tasks, labelled behaviours) before it is used where it is not.
- **Against baselines:** a new signal is compared with the simple ones (loss, weight norm, a text monitor) at the same budget.
- **Small before large:** a question is settled on a model that fits one GPU before it is asked of a large one.
- **Functional terms only:** we describe what a model computes and makes available, not what it experiences.
- **Contained:** scenarios that push agents to exploit tasks run with no network, no shared cache and no path between episodes. Such scenarios and their traces are published only after publication rules are settled.

## Contributing

The most useful contributions right now are conversations. If you work on interpretability, learning mechanics, or the training of agents, and want a shared, reproducible instrument, open an issue in the repository that fits.

## References

- Simon et al., *There Will Be a Scientific Theory of Deep Learning*, [arXiv:2604.21691](https://arxiv.org/abs/2604.21691), and the [open directions](https://learningmechanics.pub/openquestions/)
- Anthropic, *Verbalizable Representations Form a Global Workspace in Language Models* (2026), [transformer-circuits.pub](https://transformer-circuits.pub/2026/workspace/), [code](https://github.com/anthropics/jacobian-lens)
- Michaud et al., *The Quantization Model of Neural Scaling*, [arXiv:2303.13506](https://arxiv.org/abs/2303.13506)
- Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability*, [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Olsson et al., *In-context Learning and Induction Heads*, [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
- Hoogland et al., *The Developmental Landscape of In-Context Learning*, [arXiv:2402.02364](https://arxiv.org/abs/2402.02364)
- Biderman et al., *Pythia*, [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)
- Groeneveld et al., *OLMo*, [arXiv:2402.00838](https://arxiv.org/abs/2402.00838)
- Goodfire, *Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations*, [arXiv:2609.19101](https://arxiv.org/abs/2609.19101)
- METR and Redwood Research, investigation of the OpenAI / Hugging Face hacking incident (2026): [METR](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/) · [Redwood Research](https://www.redwoodresearch.org/research/hugging-face-incident)
