# Roadmap

**Focus: white-box oversight at scale, with methods verified to scale.** Run interpretability measurements on large models, at every checkpoint of their training and post-training, and show that they see what training puts inside a model before it shows in behaviour.

**Verification first.** Tasks that are cheap to verify get solved fast. White-box methods are not: when a method says "the model is thinking about X", nobody can usually check it. We make them cheap to verify, and a method is trusted only after it passes the same checks as every other:

| Axis | Cheap verification from |
|---|---|
| Ground truth | Planted concepts: trained in, so the answer is known |
| Scale | The same planted concept at several model sizes |
| Training time | Every checkpoint of a real run |
| Real behaviour | verifiers rubrics say whether an episode hacked its reward: a label for free |
| Pressure | Train against the monitor; does the signal survive? |
| Cost | FLOPs and latency per token, measured |

Once verification is cheap, finding new methods that scale becomes a loop anyone can run.

**Any training stack; Prime Intellect first.** Whatever trains, serves and scores a model stays where it is; `explorers` reads what it writes (checkpoints, adapters, rollouts) through a thin adapter per stack and measures what is inside. We build no trainer and no inference engine: we build the computer white-box methods run on ([Architecture](README.md#architecture)), on Modal first, locally, or in a team's own cluster. The first integration is Prime Intellect's stack (pods, [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), [verifiers](https://github.com/PrimeIntellect-ai/verifiers), vLLM), for its reach; others follow as users ask.

A step belongs here if it scales a white-box method to large models, builds the infrastructure to run it across training runs, or produces an oversight result. Each step has a deliverable and a condition that says when it is done. There are no dates.

Status: **done**, **next**, **planned**, **later**.

1. **O0 — Foundations** · done
2. **O1 — The instrument at scale** · next
3. **O2 — Reward hacking: the Arena's first track, and watching it being learned** · next
4. **O3 — More tracks: pressure, more environments, more models** · planned
5. **O4 — Watch alongside training** · planned
6. **O5 — The computer: executor, optimizer, scheduler** · planned
7. **Later** — pretraining science, a second backend

## Across the stack: eval, post-training, pretraining

One engine serves every stage: a study is models × examples. Eval and post-training come first: they are what most teams do, and what our first integration runs (`uv run eval`, prime-rl), and they have known answers. Pretraining reuses the same engine later.

| | Eval | Post-training | Pretraining |
|---|---|---|---|
| When | Now | Now | Later |
| What changes | Nothing: one model, many episodes | A base model, by SFT or RL | Everything, from random weights |
| Model axis | One checkpoint | Base + LoRA adapter per step | Full checkpoints |
| Examples from | Eval episodes, replayed | The run's rollouts and eval episodes | A fixed set of texts |
| Ground truth | Rubric labels per episode; planted models | Planted concepts; rubric labels for reward hacking | Injected documents; known data order (Pythia) |
| Question | What does the model hold before it answers (eval awareness, a plan to exploit)? | Does what the run instils appear inside before behaviour? | When do representations and the workspace form? |
| Lens | Fitted once | Maybe fitted once; to be measured | Refitted per checkpoint |

What pretraining adds: loading full checkpoints, refitting the lens per checkpoint, and ground truth by injected documents. Nothing in the engine assumes adapters: every checkpoint is a base plus a change.

## O0 — Foundations · done

- **One library.** `explorers` is one package with six concepts: models with named streams, traces, ops (interventions as data), measures, studies, and a store keyed by content. The design fits on [one page](https://github.com/machine-exploration/explorers/blob/main/docs/interface.md); library code went from 2,253 to 1,775 lines in the lean pass with identical results.
- **Streams: read, write, trace.** `residual`, `attn_out` and `mlp_out` on GPT-NeoX, Llama and GPT-2 layouts. Reads equal what hand-written hooks return; a write changes exactly its target.
- **The study.** Reads, writes, measures and patching (exact and attribution) over models or the checkpoints of a run, cached by content. Five canonical examples (a linear probe, steering, activation patching, attribution patching, a sparse autoencoder read) are checked against results that are exact by construction.
- **The Jacobian lens.** Implemented and checked against the [reference implementation](https://github.com/anthropics/jacobian-lens): `J` agrees within 1.2e-7, and the top-5 readouts are identical at 54/54 (layer, position) pairs ([docs](https://github.com/machine-exploration/explorers/blob/main/docs/jlens.md)). The paper's four workspace signatures (dimension, sparsity of readouts, persistence, cross-layer similarity) are measures, with a Jacobian of the final or the penultimate residual.
- **A reproduced result.** On a toy with 16 lookup tasks of Zipf frequencies, frequent tasks are learned first (rank correlation −0.74), each suddenly (median sharpness 0.74).

## O1 — The instrument at scale · next

- **Concept-targeted lens** · done. The lens readout for a chosen token is one backward pass (starting from that token's direction through the final norm and unembedding), so k concepts cost k passes, whatever the model's width. The full lens needs `d` passes and grows roughly with d³. Built as a monitor (`concept_monitor`), with the logit lens on the same words as its baseline ([docs](https://github.com/machine-exploration/explorers/blob/main/docs/monitors.md)).
- **Large models.** 32–70B models loaded across several GPUs; bf16 forward, fp32 accumulation.
- **Adapter checkpoints** · done. A prime-rl LoRA run keeps only its last two adapters; `ex.archive_adapters` copies each one out as it lands, and `ex.adapters` gives one checkpoint per step, merged into the base at load. Checked: merged equals the adapter run unmerged ([docs](https://github.com/machine-exploration/explorers/blob/main/docs/prime.md)). Next: load the base once and swap the adapter.
- **Episode replay** · done. Episodes written by an eval or an RL run become examples: the exact token ids the model saw, with its sampled tokens marked. Check on Qwen 3.8 27B: replayed log-probabilities of the sampled tokens differ from those vLLM recorded by a median of 0.0003 (max 0.32, bf16) ([docs](https://github.com/machine-exploration/explorers/blob/main/docs/episodes.md)).
- **Scale-out.** Every job is a pure function of (checkpoint, examples, plan), stored by content: shard checkpoints across rented GPUs (Modal first) with no coordination.
- **Done when:** on a 7–8B model the concept lens equals the full lens on those concepts (exact by construction); a 70B checkpoint with 10 concepts runs in minutes, with the cost published.

## O2 — Reward hacking: the Arena's first track, and watching it being learned · next

Two views of the same behaviour. The Arena scores monitors on finished rollouts; the first program watches the behaviour form during training.

- **Watching it being learned (the first program, in [mechanics](https://github.com/machine-exploration/mechanics)):** RL on `impossible_code` with every step's LoRA adapter kept, and the same evaluation set read at every checkpoint. Does the representation of hacking rise before the hack rate does? A probe needs labelled hacks, so it cannot exist before the behaviour; a monitor built from words can watch from step 0. Controls: a planted concept (must be found) and an environment that cannot be hacked (must stay flat). Measured: lead time, whether the direction at the end of training is already present at the start, and the cost of reading every checkpoint.

The Monitor Arena is an open leaderboard of white-box monitoring methods on real rollouts: how much each one catches, what it costs and how far it scales, measured the same way for every method.

- **Rows:** methods, each a measure anyone can submit. First entries: a difference-of-means probe (the protocol of [Goodfire, 2026](https://arxiv.org/abs/2609.19101)), the concept Jacobian lens (a direction from words, no data), the logit lens; baselines: an LLM monitor and a chain-of-thought monitor.
- **Columns:** quality (detection at 1% and 5% false-positive rate, AUROC, how early in a rollout it fires), cost (data needed to fit: none, words, synthetic pairs or labels; fit compute; FLOPs and latency per token), scale (largest model, rollout length, monitors run at once).
- **Tracks:** one per environment × model. Rollouts and labels come from [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environments and their rubrics; `explorers` replays each episode through the model and scores every method on it. Some environments are held out and never used to develop a method.
- **First track:** ImpossibleBench (its tasks cannot be solved honestly, so a pass is a hack by construction), ported as the verifiers environment [`impossible_code`](https://github.com/machine-exploration/verifiers/tree/main/environments/impossible_code), on a model that fits one GPU, then larger models.
- **Checked first where the answer is known:** a concept planted by a LoRA fine-tune, where every method must find what was planted before it is trusted on real rollouts.
- **Zero-cost first look:** at the same layer, the cosine similarity between the probe's direction and the concept directions for words such as "cheating", "hack", "hardcoded".
- **Done when:** the first track and the first program's figure (hack rate and internal signal over training steps, with both controls) are public, each with the run that reproduces it. Positive or negative.

## O3 — More tracks: pressure, more environments, more models · planned

- **Pressure:** train against a monitor; does its signal survive?
- **More tracks:** more environments (SWE-bench, DeepSWE, non-coding), more and larger models (70B+ through the model server, O5).
- **Needs:** contained runs (no network, no path between episodes).

## O4 — Watch alongside training · planned

- **Deliverable:** `watch(run, every=N, measures=[...])`: the same studies running next to a prime-rl run, beside the loss, the reward and the evals; per-run reports; alerts when something appears inside before it appears in behaviour. First beside the trainer on the files it writes, then as a hook inside it on the live weights. It runs where the weights are, including a team's own cluster. Open source.
- **Done when:** a team runs it on its own training run.

## O5 — The computer: executor, optimizer, scheduler · planned

The [architecture](README.md#architecture), built one layer at a time, each layer earned by a measurement on the first program's workload.

- **Executor:** the five primitives on a resident model, locally and on Modal. Done when the same study gives the same result on both, within a stated tolerance.
- **Optimizer:** shared forward passes, fused projections, reductions on the GPU, a run's adapters batched on one base; `explain()` shows the plan and its predicted cost. Done when it is at least 3× faster on the first program with unchanged results, measured against NNsight and TransformerLens on the same study.
- **Scheduler and model server:** many users on one resident base, paged state memory, recomputation from lineage. Done when GPU utilisation and cost per method are published.

## Later

- **Pretraining science** (see Across the stack). When does the verbalizable workspace form during pretraining, how suddenly, at which depths and from which size? On Pythia's checkpoints, with its four signatures ([experiment](https://github.com/machine-exploration/mechanics/tree/main/experiments/q1_verbalizable_space)). Then how mechanisms form: induction heads, representation vs use, circuit replacement.
- **A second backend** (NNsight or TransformerLens) giving the same results within a stated tolerance.
- **Whole-space signatures at large scale** through sketches, checked against the exact Jacobian on smaller models.

## Known risks

- **Planted concepts can be overfitted.** Held-out environments and real rollouts (O2) keep the Arena honest.

- **We depend on prime-rl's file layout.** It is read at a pinned commit and tested; `explorers` never imports prime-rl, so a change breaks one reader, not the library.

- **The planted concept may not lead behaviour.** The result is published either way, with its controls.
- **A readable workspace may need size.** Each model is checked first: does the paper's layer structure appear at the final checkpoint?
- **Backend equivalence is hard:** site names, write support and numerics differ between tools; differences that cannot be closed are documented.
- **Existing tools are strong:** NNsight/NDIF and TransformerLens. Explorers must win on studies across training runs and on measured execution, not on a nicer hook API.
