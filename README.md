# Machine Exploration

> **Machine Exploration is building the infrastructure for a science of intelligence.**

We want to understand how intelligence emerges inside deep learning systems: how representations and mechanisms form during training, how they change, and how they give rise to behavior.

Our focus is **white-box experimentation at scale**: making it possible to observe, measure, and intervene on the internal computation of models across training and deployment.

The long-term goal is to turn interpretability from a collection of techniques into an empirical science of learned intelligence. If we can understand these systems from the inside, we should also be able to build much stronger methods for monitoring and controlling them as they become more capable.

The interface, the Runtime, the formats, the runs and the results are open source. **Status: pre-alpha.** The plan is in [ROADMAP.md](ROADMAP.md).

---

## The thesis

> **A neural network is a learned computation: its weights are the program, and its internal streams carry the running state.**
> Machine Exploration builds the instrumentation to read, write and trace both.

## One company, three programs

| Program | What it is | Its role |
|---|---|---|
| **[Explorers](https://github.com/machine-exploration/explorers)** | The open-source scientific interface: read, write and trace the streams inside a model | Sets the standard for how white-box experiments are written |
| **Machine Exploration Runtime** | The infrastructure that executes experiments at scale: a query engine for neural computation. Open source. | Makes experiments with millions of reads and interventions practical |
| **[Mechanics](https://github.com/machine-exploration/mechanics)** | The research program: how training creates representations, algorithms and circuits | Gives the direction, and the science that tests the tools |

The three feed each other. Research finds a useful measurement; it becomes an Explorers method; researchers run it; large runs need the Runtime; the Runtime makes much richer data; that data feeds Mechanics.

```
   Mechanics ──── new methods ───▶ Explorers ──── studies ───▶ Runtime
       ▲                                                          │
       └──────────────── larger, richer experiments ──────────────┘
```

## Explorers: the interface

Explorers is not another collection of interpretability algorithms. It sets the vocabulary and the execution model:

```
Model  →  Execution  →  Stream  →  Trace
                         read(stream, location)
                         write(stream, location)
```

A researcher writes what they want to read or change, and does not care where it runs: plain PyTorch, TransformerLens, NNsight, vLLM, a training loop, or a remote cluster. Existing tools can be backends. Methods are packages built on the same substrate: probes, lenses, patching, attribution, sparse autoencoders, circuits.

The unit of work is a **study**: reads, interventions and measurements over models, checkpoints and examples.

```python
# Planned API (pre-alpha, will change)
study = ex.Study(models=[model], checkpoints=checkpoints, examples=data)
study.read(stream="residual", layers="*")
study.measure(probe)
results = study.compute()                                 # locally, on one GPU

study.patch(source=clean, target=corrupt, sites=residual.everywhere())
study.measure(logit_diff)
results = study.compute(backend="machine-exploration")    # the same program, at scale
```

The test of the abstraction: a researcher changes the backend and does not rewrite the experiment, and gets the same result.

What exists today is the start of this: examples identified by their content, model states along training, measurements that declare what they read, an engine that serves every measurement on a checkpoint from one forward pass, a store keyed by content, and a first lens (the Jacobian lens, checked against its reference implementation).

## The Runtime: a query engine for neural computation

Serious white-box experiments are expensive. 10,000 examples × 32 layers × 128 intervention sites × 20 checkpoints × 3 seeds is millions of model computations, and causal methods multiply them with counterfactual runs.

The Runtime takes a whole study and builds an execution plan for it, the way a database plans a query:

- one forward pass for every read a checkpoint needs;
- source activations computed once and reused across counterfactual runs;
- shared computation up to an intervention point, then branching;
- batching of counterfactual branches, activation caching, deduplication, recomputation when cheaper than storage;
- checkpoint loading, parallelism across examples, sites and devices, scheduling, and result storage.

```
                     ┌── write A → continuation
prefix computation ──┼── write B → continuation
                     └── write C → continuation
```

NNsight executes an intervention. The Runtime executes a study. It is open source, like the interface. Our claims about speed will be measured against existing tools on the same study, and published.

## Mechanics: how training creates computation

Explorers gives a trace of one checkpoint. Mechanics compares traces across checkpoints, and looks for the laws.

- **Representation formation:** when does information become present in a stream, and how suddenly?
- **Causal formation:** when does *representing* something turn into *using* it? A probe can find a feature long before an intervention on it changes behaviour.
- **Circuit formation:** how does a learned algorithm assemble itself, step by step?
- **Competition and replacement:** one mechanism solves a task, and training later replaces it with another while the loss barely moves.
- **The verbalizable space** *(first flagship)*: finished language models hold a shared space of content they are disposed to say, read by the Jacobian lens ([Anthropic, 2026](https://transformer-circuits.pub/2026/workspace/)). When does it form during training?

The long-term goal: **predict what a model will learn, and when, before we train it**, and read and act on the computation behind a behaviour, not only its outputs.

## Training and deployment

The same instrumentation applies on two clocks.

| Clock | What changes | Example question |
|---|---|---|
| **Training time** | Weights change from checkpoint to checkpoint | When does a capability form, and how suddenly? |
| **Deployment** | An agent acts over many turns, sometimes with other agents | What does an agent hold internally before it acts? Does it differ from what it says? |

Later, the same studies run continuously alongside training: white-box measurements next to the loss, the reward and the evals.

We describe what a model computes and makes available in functional terms only. We make no claims about experience.

## What exists today

| Repository | What it holds |
|---|---|
| [explorers](https://github.com/machine-exploration/explorers) | The open-source library, one package with six concepts: models with named streams, traces (read and write in one forward pass), ops (interventions as data), measures (losses, weight statistics, the Jacobian lens), studies (the unit of work, cached by content). The agent side (scenarios, a multi-agent runtime) is frozen behind an extra. The design fits on [one page](https://github.com/machine-exploration/explorers/blob/main/docs/interface.md). |
| [mechanics](https://github.com/machine-exploration/mechanics) | The research program. First result: on a toy with tasks of Zipf frequencies, frequent tasks are learned first (rank correlation −0.74), and each suddenly. |
| [public](https://github.com/machine-exploration/public) | This page, the roadmap, and later research notes and results. |
| [verifiers](https://github.com/machine-exploration/verifiers), [vllm](https://github.com/machine-exploration/vllm) | Pinned forks of upstream projects used as backends. No local changes. |

## Principles

- **Open by default:** the interface, the Runtime, the formats, the runs and the results. Negative results are published too.
- **Reproducible:** every result replays from its config, seed, data order, model version and code version.
- **Same study, same result:** a study gives the same result on every backend, within a stated tolerance, or the backend is not supported.
- **Measured claims:** speed-ups are measured against existing tools on the same study, and published.
- **Causal, not only correlational:** a claim that a feature exists is tested by intervention, not only by a readout that correlates.
- **Ground truth first:** a method is validated where the answer is known before it is used where it is not.
- **Small before large:** a question is settled on a model that fits one GPU before it is asked of a large one.
- **Contained:** scenarios that push agents to exploit tasks run with no network, no shared cache and no path between episodes, and are published only after publication rules are settled.

## Contributing

The most useful contributions right now are conversations. If you run white-box experiments, build interpretability tools, or study how training shapes models, open an issue in the repository that fits.

## References

- Simon et al., *There Will Be a Scientific Theory of Deep Learning*, [arXiv:2604.21691](https://arxiv.org/abs/2604.21691), and the [open directions](https://learningmechanics.pub/openquestions/)
- Anthropic, *Verbalizable Representations Form a Global Workspace in Language Models* (2026), [transformer-circuits.pub](https://transformer-circuits.pub/2026/workspace/), [code](https://github.com/anthropics/jacobian-lens)
- Fiotto-Kaufman et al., *NNsight and NDIF*, [arXiv:2407.14561](https://arxiv.org/abs/2407.14561)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- Michaud et al., *The Quantization Model of Neural Scaling*, [arXiv:2303.13506](https://arxiv.org/abs/2303.13506)
- Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability*, [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Olsson et al., *In-context Learning and Induction Heads*, [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
- Hoogland et al., *The Developmental Landscape of In-Context Learning*, [arXiv:2402.02364](https://arxiv.org/abs/2402.02364)
- Biderman et al., *Pythia*, [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)
- Goodfire, *Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations*, [arXiv:2609.19101](https://arxiv.org/abs/2609.19101)
- METR and Redwood Research, investigation of the OpenAI / Hugging Face hacking incident (2026): [METR](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/) · [Redwood Research](https://www.redwoodresearch.org/research/hugging-face-incident)
