# Roadmap

**Focus: white-box oversight at scale.** Run interpretability measurements on large models, at every checkpoint of their training and post-training, and show that they see what training puts inside a model before it shows in behaviour.

A step belongs here if it scales a white-box method to large models, builds the infrastructure to run it across training runs, or produces an oversight result. Each step has a deliverable and a condition that says when it is done. There are no dates.

Status: **done**, **next**, **planned**, **later**.

1. **O0 — Foundations** · done
2. **O1 — The instrument at scale** · next
3. **O2 — An oversight result with a known answer** · next
4. **O3 — An oversight result on a realistic run** · planned
5. **O4 — The Runtime alongside training** · planned
6. **Later** — pretraining science, a second backend, the planner

## Why post-training first

Post-training comes first for three reasons: it has a known answer (we plant what we look for), it is the cheapest (a run is one base model plus small adapters), and it is where most teams train. Pretraining reuses almost the whole engine; two parts change.

| Part | Carries over to pretraining? | Why |
|---|---|---|
| Concept lens, study, plan, store, sharding, report | Yes, unchanged | They only need a sequence of model states |
| Behaviour measure; the question "inside before behaviour" | Yes | The same question on any run |
| Multi-GPU loading of large models | Yes | The same models |
| Checkpoint handling | Partly | Pretraining checkpoints are full weights, not adapters. Every checkpoint is modelled as base + change, so no new code path |
| Reusing the lens across checkpoints | Partly | In a fine-tune the Jacobian may barely move; in pretraining it must be refitted per checkpoint. To be measured |
| Known ground truth | No, not directly | Nothing is planted in pretraining. Closest: continued pretraining with injected documents, or Pythia's known data order |

## O0 — Foundations · done

- **One library.** `explorers` is one package with six concepts: models with named streams, traces, ops (interventions as data), measures, studies, and a store keyed by content. The design fits on [one page](https://github.com/machine-exploration/explorers/blob/main/docs/interface.md); library code went from 2,253 to 1,775 lines in the lean pass with identical results.
- **Streams: read, write, trace.** `residual`, `attn_out` and `mlp_out` on GPT-NeoX, Llama and GPT-2 layouts. Reads equal what hand-written hooks return; a write changes exactly its target.
- **The study.** Reads, writes, measures and patching (exact and attribution) over models or the checkpoints of a run, cached by content. Five canonical examples (a linear probe, steering, activation patching, attribution patching, a sparse autoencoder read) are checked against results that are exact by construction.
- **The Jacobian lens.** Implemented and checked against the [reference implementation](https://github.com/anthropics/jacobian-lens): `J` agrees within 1.2e-7, and the top-5 readouts are identical at 54/54 (layer, position) pairs ([docs](https://github.com/machine-exploration/explorers/blob/main/docs/jlens.md)). The paper's four workspace signatures (dimension, sparsity of readouts, persistence, cross-layer similarity) are measures, with a Jacobian of the final or the penultimate residual.
- **A reproduced result.** On a toy with 16 lookup tasks of Zipf frequencies, frequent tasks are learned first (rank correlation −0.74), each suddenly (median sharpness 0.74).

## O1 — The instrument at scale · next

- **Concept-targeted lens.** The lens readout for a chosen token is one backward pass (starting from that token's direction through the final norm and unembedding), so k concepts cost k passes, whatever the model's width. The full lens needs `d` passes and grows roughly with d³.
- **Large models.** 32–70B models loaded across several GPUs; bf16 forward, fp32 accumulation.
- **Adapter checkpoints.** A fine-tuning run is one base model plus small LoRA adapters: the base is loaded once, the adapter swapped per checkpoint.
- **Scale-out.** Every job is a pure function of (checkpoint, examples, plan), stored by content: shard checkpoints across GPUs on rented machines with no coordination.
- **Done when:** on a 7–8B model the concept lens equals the full lens on those concepts (exact by construction); a 70B checkpoint with 10 concepts runs in minutes, with the cost published.

## O2 — An oversight result with a known answer · next

- **Question:** when a concept is planted in a large open model by fine-tuning, does it appear inside the model before it shows in behaviour, and by how many checkpoints?
- **Study:** LoRA fine-tuning on synthetic documents that tie a trigger topic to a single-token concept; a checkpoint every few steps. At each checkpoint: behaviour (the model says the concept in trigger contexts) and inside (the concept's rank in the lens at trigger positions, before any output).
- **Controls:** contexts without the trigger; the logit lens; a concept never planted; the base model.
- **Order:** the whole pipeline on a 7–8B model first, then 32–70B.
- **Done when:** a result note in this repository, with the run that reproduces it. Positive or negative.

## O3 — An oversight result on a realistic run · planned

- **Question:** during RL post-training on exploitable tasks, does the plan to exploit appear inside the model before the exploit shows? Does the signal survive training against a monitor?
- **Deliverable:** detection rate at 1%, 5% and 10% false-positive rate, with intervals from a bootstrap over episodes, against a chain-of-thought monitor and an LLM judge at the same false-positive rate.
- **Needs:** RL infrastructure reused (the pinned [verifiers](https://github.com/machine-exploration/verifiers) fork or equivalent), and the contained agent scenarios of `explorers.populations`, frozen until then.

## O4 — The Runtime alongside training · planned

- **Deliverable:** `watch(training_run, every=N, measures=[...])`: the same studies running next to a training or post-training run, beside the loss, the reward and the evals; per-run reports; alerts when something appears inside before it appears in behaviour. It runs where the weights are, including a team's own cluster. Open source, like the interface.
- **Done when:** a team runs it on its own training run.

## Later

- **Pretraining science.** When does the verbalizable workspace form during pretraining, how suddenly, at which depths and from which size? On Pythia's checkpoints, with its four signatures ([experiment](https://github.com/machine-exploration/mechanics/tree/main/experiments/q1_verbalizable_space)). Then how mechanisms form: induction heads, representation vs use, circuit replacement.
- **A second backend** (NNsight or TransformerLens) giving the same results within a stated tolerance.
- **The planner:** shared computation up to an intervention point, batched counterfactuals, measured against existing tools on the same study.
- **Whole-space signatures at large scale** through sketches, checked against the exact Jacobian on smaller models.

## Known risks

- **The planted concept may not lead behaviour.** The result is published either way, with its controls.
- **A readable workspace may need size.** Each model is checked first: does the paper's layer structure appear at the final checkpoint?
- **Backend equivalence is hard:** site names, write support and numerics differ between tools; differences that cannot be closed are documented.
- **Existing tools are strong:** NNsight/NDIF and TransformerLens. Explorers must win on studies across training runs and on measured execution, not on a nicer hook API.
