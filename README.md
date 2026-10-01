# Machine Exploration

We have spent billions of dollars of compute producing models nobody has mapped. Machine Exploration builds the open stack to run white-box interpretability methods at scale, across the whole life of a model: pretraining, post-training, evals and deployment.

The evidence that this works is recent. Activation probes catch reward hacking in frontier open models about as well as chain-of-thought monitors, at a fraction of the cost ([Goodfire, 2026](https://arxiv.org/abs/2609.19101)). The Jacobian lens reads what a model is poised to say from any layer, without labels ([Anthropic, 2026](https://transformer-circuits.pub/2026/workspace/)). Our first result puts the two side by side on the same runs: can a monitor that needs no labels match one that does?

These monitors read activations the model already computes. They add no latency, cost almost nothing per token, never change the model's outputs, and work on any open model. Each targets one behaviour, and hundreds run in parallel as one matrix product per layer. Long term, the stack is where new white-box methods are discovered: every method scored against known answers, at every stage, at scale, with its cost.

Two applications share the stack. **Safety:** monitors that catch what training or deployment puts inside a model before it shows in behaviour. **Interpretability research:** finding and verifying the representations and mechanisms a model computes, and how training builds them. A method that passes the same checks serves both: a monitor, and a finding about what the model represents.

Works with any training stack; the first integration is [Prime Intellect](https://github.com/PrimeIntellect-ai/prime-rl). Everything is open source. **Status: pre-alpha.** Plan: [ROADMAP.md](ROADMAP.md).

## Thesis

A neural network is a learned computation. Its weights are the program; its internal streams carry the running state. We build the instruments to read, write and trace both.

## Architecture

White-box methods are programs; the model is the machine they run on. A method is written once, against a few primitives on a model's internal state, and runs wherever the model lives: on a laptop, in our cloud, or next to a customer's training stack.

```python
@ex.method
def hack_score(m, x, words=("cheat", "hack", "hardcode")):
    v = m.concept(words, layer=20)                    # one backward pass per word, no labels
    return m.residual[20](x).project(v).max("pos")    # scores leave the GPU, activations do not

R = ex.vmap(hack_score, over=run.steps)(evalset)     # every checkpoint of a run; nothing runs yet
R.explain()                                          # the plan and its predicted FLOPs, bytes and cost
R.compute(on=ex.Modal(gpu="H100"))                   # or ex.Local(), or your cluster
```

*A design, not yet built.* Each layer reuses an idea already proven in production systems; the new object is the internal state of a network: enormous, ephemeral, differentiable, indexed by site, checkpoint and input.

| Layer | What it does | Idea proven in |
|---|---|---|
| Method | Plain Python with `@method`, traced into a graph rather than run eagerly | Modal, JAX |
| IR | A typed graph of primitives over sites; every node knows its shape, FLOPs and bytes | JAX, Spark |
| Optimizer | Shares forward passes, fuses projections into one matrix product, pushes reductions to the GPU, batches a run's adapters on one base; `explain()` shows the plan | Postgres, XLA |
| Scheduler | Continuously batches white-box operations from many methods and users onto one resident base | vLLM |
| Model server | Resident base weights, a pool of adapters, paged state memory, recomputation from lineage | Tinker, S-LoRA, Spark |
| Storage | Weights, episodes (token ids) and results, content-addressed and separate from compute | Snowflake, Bazel |
| Substrate | Containers, GPUs, sandboxes: Modal first | Modal |

- **Five primitives, two transformations.** Eval f<sub>θ</sub>(x); read h<sub>s</sub>; project ⟨h<sub>s</sub>, v⟩; intervene do(h<sub>s</sub> := g(h<sub>s</sub>)); vector-Jacobian product uᵀ ∂h<sub>s′</sub>/∂h<sub>s</sub>. `vmap` lifts a method over the checkpoints of a run and over a set of inputs.
- **State is ephemeral, lineage is permanent.** A 70B model's residual stream over a million tokens is about 1.3 TB; its lineage (weights hash and token ids) is a few kilobytes and recomputes it exactly.
- **Execution invariance.** The optimizer and scheduler may reorder, fuse, batch, shard and recompute, never change a result: every run equals the naive reference within a stated tolerance, and carries its replay error.
- **Cost is part of the result.** Every run returns what it computed and what it spent: FLOPs, bytes, peak memory, GPU-seconds, dollars.
- **Each layer is earned by a measurement.** Primitives on one GPU first; then the first program on Modal; the optimizer only when it is at least 3× faster on that program with the same results; the scheduler when many users share a base.

## Programs

| | |
|---|---|
| **[Explorers](https://github.com/machine-exploration/explorers)** | The interface. Six concepts: Model, Stream, Trace, Op, Measure, Study. |
| **Training stacks** | Whatever trains, serves and scores the model: any stack, through thin adapters that read its checkpoints and rollouts. Today: Hugging Face checkpoints and [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) runs, the first integration. Planned: other trainers as users ask, and a trainer backend for the large passes white-box methods need on 70B+ models. |
| **[Mechanics](https://github.com/machine-exploration/mechanics)** | The research: how training creates representations, algorithms and circuits. |

```python
import explorers as ex

ex.archive_adapters("outputs/my-run", "runs/my-run/adapters")    # a prime-rl LoRA run, as it trains
checkpoints = ex.adapters("Qwen/Qwen3-8B", "runs/my-run/adapters", device="cuda", dtype="bfloat16")
study = ex.Study(checkpoints, examples)
study.measure(ex.measures.loss, ex.measures.jlens_error(layers=range(1, 36)))
results = study.compute(store="runs/store")   # xarray, indexed by step, cached by content
```

Change the backend, keep the study, get the same result.

## First questions

- **Watching a model learn to cheat.** Reinforcement learning on a coding environment where any pass is a reward hack, with every step's adapter kept. Does the internal representation of hacking rise before the hack rate does? A probe needs labelled hacks, so it cannot exist before the behaviour; a monitor built from words can watch from step 0. Controls: a planted concept (must be found) and an environment that cannot be hacked (must stay flat).
- **The Monitor Arena.** An open leaderboard of white-box monitoring methods on real rollouts: how much each catches, what it costs and how far it scales. First track: reward hacking, where a Jacobian-lens monitor built from words is set against a difference-of-means probe.
- **Reward hacking, without labels.** On an exploitable coding environment, does a Jacobian-lens monitor (no labels) catch hacks as well as a difference-of-means probe (labels), at a matched false-positive rate and at what cost? First checked on a planted concept, where the answer is known.
- **Replayed evals.** Every episode of an eval, replayed through the model: what does it hold before it answers (eval awareness, a plan to exploit)?
- **The verbalizable space.** Finished models share a space of what they are disposed to say ([Anthropic, 2026](https://transformer-circuits.pub/2026/workspace/)). When does it form, and how does post-training change it?

## Principles

- Open: code, formats, runs, results. Negative results too.
- Reproducible: every result replays from config, seed, data order and code version.
- Same study, same result on every backend, within a stated tolerance.
- Causal: a feature is tested by intervention, not only by a readout.
- Ground truth first: a method is validated where the answer is known.
- Measured: speed claims are benchmarked against existing tools on the same study.

We describe what models compute in functional terms only. We make no claims about experience.

## Contributing

Conversations first. If you run white-box experiments or study how training shapes models, open an issue.

## References

- Simon et al., *There Will Be a Scientific Theory of Deep Learning*, [arXiv:2604.21691](https://arxiv.org/abs/2604.21691)
- Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models* (2026), [paper](https://transformer-circuits.pub/2026/workspace/), [code](https://github.com/anthropics/jacobian-lens)
- Fiotto-Kaufman et al., *NNsight and NDIF*, [arXiv:2407.14561](https://arxiv.org/abs/2407.14561) · [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- Olsson et al., *In-context Learning and Induction Heads*, [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
- Hoogland et al., *The Developmental Landscape of In-Context Learning*, [arXiv:2402.02364](https://arxiv.org/abs/2402.02364)
- Biderman et al., *Pythia*, [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)
- Goodfire, *Monitoring and Discovering Reward Hacking with Internal Representations*, [arXiv:2609.19101](https://arxiv.org/abs/2609.19101)
