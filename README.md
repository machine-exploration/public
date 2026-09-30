# Machine Exploration

**See inside training.**

White-box oversight for training and post-training runs: what a run puts inside a model, at every checkpoint, before it shows in behaviour.

We are building the infrastructure for a science of deep learning. Runs on [Prime Intellect](https://github.com/PrimeIntellect-ai/prime-rl). Everything is open source. **Status: pre-alpha.** Plan: [ROADMAP.md](ROADMAP.md).

## Thesis

A neural network is a learned computation. Its weights are the program; its internal streams carry the running state. We build the instruments to read, write and trace both.

## Programs

| | |
|---|---|
| **[Explorers](https://github.com/machine-exploration/explorers)** | The interface. Six concepts: Model, Stream, Trace, Op, Measure, Study. |
| **Runtime** | [Prime Intellect](https://github.com/PrimeIntellect-ai/prime-rl): trains, serves and scores the run. Explorers reads what it writes, on the same machines. |
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

- **Planted concepts.** A concept is fine-tuned into a large model. Does it appear inside before it shows in behaviour?
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
