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

**Runs on Prime Intellect.** Its stack (pods, [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), [verifiers](https://github.com/PrimeIntellect-ai/verifiers), vLLM) trains, serves and scores the run. `explorers` opens what the run writes and measures what is inside, on the same machines. We do not build our own runtime.

A step belongs here if it scales a white-box method to large models, builds the infrastructure to run it across training runs, or produces an oversight result. Each step has a deliverable and a condition that says when it is done. There are no dates.

Status: **done**, **next**, **planned**, **later**.

1. **O0 — Foundations** · done
2. **O1 — The instrument at scale** · next
3. **O2 — The harness and its first entries** · next
4. **O3 — The natural case: reward hacking** · planned
5. **O4 — Watch alongside training** · planned
6. **Later** — the open scoreboard, pretraining science, a second backend, the planner

## Across the stack: eval, post-training, pretraining

One engine serves every stage: a study is models × examples. Eval and post-training come first: they are what Prime Intellect's stack runs (`uv run eval`, prime-rl) and what most teams do, and they have known answers. Pretraining reuses the same engine later.

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

- **Concept-targeted lens.** The lens readout for a chosen token is one backward pass (starting from that token's direction through the final norm and unembedding), so k concepts cost k passes, whatever the model's width. The full lens needs `d` passes and grows roughly with d³.
- **Large models.** 32–70B models loaded across several GPUs; bf16 forward, fp32 accumulation.
- **Adapter checkpoints** · done. A prime-rl LoRA run keeps only its last two adapters; `ex.archive_adapters` copies each one out as it lands, and `ex.adapters` gives one checkpoint per step, merged into the base at load. Checked: merged equals the adapter run unmerged ([docs](https://github.com/machine-exploration/explorers/blob/main/docs/prime.md)). Next: load the base once and swap the adapter.
- **Episode replay.** Episodes written by an eval or an RL run become examples: the transcript tokenised as the model saw it, with the positions of each model turn. Check: replayed log-probabilities of the sampled tokens equal those the inference server recorded, within a stated bf16 tolerance.
- **Scale-out.** Every job is a pure function of (checkpoint, examples, plan), stored by content: shard checkpoints across Prime Intellect pods with no coordination.
- **Done when:** on a 7–8B model the concept lens equals the full lens on those concepts (exact by construction); a 70B checkpoint with 10 concepts runs in minutes, with the cost published.

## O2 — The harness and its first entries · next

- **Question:** when a concept is planted in a large open model by fine-tuning, which white-box methods see it inside before it shows in behaviour, by how many checkpoints, at which size, and at what cost per token?
- **Entries:** the Jacobian lens restricted to the concept; a linear probe per layer, trained on the base model and applied at every checkpoint; the logit lens. Each is a measure, scored the same way.
- **Study:** a prime-rl LoRA fine-tune on synthetic documents that tie a trigger topic to a single-token concept; a checkpoint every few steps. At each checkpoint: behaviour (the model says the concept in trigger contexts) and inside (each method's score for the concept at trigger positions, before any output: its rank in a lens, a probe's output).
- **Controls:** contexts without the trigger; a concept never planted; the base model. Some kinds of planted concept are held out and never used while developing a method, so methods cannot overfit to the organisms.
- **Order:** the whole pipeline on a 7–8B model first, then 32–70B.
- **Done when:** a result note in this repository with, for each method and size, checkpoints of lead over behaviour and cost per token, and the run that reproduces it. Positive or negative.

## O3 — The natural case: reward hacking · planned

- **Question:** during RL post-training on exploitable tasks, does the plan to exploit appear inside the model before the exploit shows? Nothing is planted: the environment's rubric labels each episode. Does the signal survive training against a monitor?
- **Deliverable:** detection rate at 1%, 5% and 10% false-positive rate, with intervals from a bootstrap over episodes, against a chain-of-thought monitor and an LLM judge at the same false-positive rate.
- **Needs:** an RL run in prime-rl on verifiers environments with exploitable tasks, contained (no network, no path between episodes); the episodes it writes become the examples of the study.

## O4 — Watch alongside training · planned

- **Deliverable:** `watch(run, every=N, measures=[...])`: the same studies running next to a prime-rl run, beside the loss, the reward and the evals; per-run reports; alerts when something appears inside before it appears in behaviour. First beside the trainer on the files it writes, then as a hook inside it on the live weights. It runs where the weights are, including a team's own cluster. Open source.
- **Done when:** a team runs it on its own training run.

## Later

- **The open scoreboard.** Any white-box method, submitted as a measure, scored on the same runs, sizes and checks, with its cost. Methods that pass become monitors; new ones are found against it.
- **Pretraining science** (see Across the stack). When does the verbalizable workspace form during pretraining, how suddenly, at which depths and from which size? On Pythia's checkpoints, with its four signatures ([experiment](https://github.com/machine-exploration/mechanics/tree/main/experiments/q1_verbalizable_space)). Then how mechanisms form: induction heads, representation vs use, circuit replacement.
- **A second backend** (NNsight or TransformerLens) giving the same results within a stated tolerance.
- **The planner:** shared computation up to an intervention point, batched counterfactuals, measured against existing tools on the same study.
- **Whole-space signatures at large scale** through sketches, checked against the exact Jacobian on smaller models.

## Known risks

- **Planted concepts can be overfitted.** Held-out kinds of concept and a natural case (O3) keep the harness honest.

- **We depend on prime-rl's file layout.** It is read at a pinned commit and tested; `explorers` never imports prime-rl, so a change breaks one reader, not the library.

- **The planted concept may not lead behaviour.** The result is published either way, with its controls.
- **A readable workspace may need size.** Each model is checked first: does the paper's layer structure appear at the final checkpoint?
- **Backend equivalence is hard:** site names, write support and numerics differ between tools; differences that cannot be closed are documented.
- **Existing tools are strong:** NNsight/NDIF and TransformerLens. Explorers must win on studies across training runs and on measured execution, not on a nicer hook API.
