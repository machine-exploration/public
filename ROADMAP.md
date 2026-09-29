# Roadmap

The plan follows the three phases in the [README](README.md): the instrument, the survey, the laws. Each step has a question, a deliverable, and a condition that says when it is done. There are no dates.

Status: **done**, **in review**, **next**, **planned**, **later**.

## Phase 1 — The instrument

### Step 0 — One library · next

- **Problem:** Two repositories claim the `explorers` namespace. The agent prototype in `explorers` is a regular package (`explorers/__init__.py`). Installed next to the workspace in `mechanics`, it hides `explorers.core`, and `import explorers.core` fails.
- **Deliverable:** The `explorers` repository becomes the workspace, with its history kept: `packages/core` and `packages/learning` from `mechanics`, and `packages/populations` for the agent side (ported from `explorers.<module>` to `explorers.populations.<module>`). One set of detection metrics instead of two.
- **Before:** Land the detector harness ([explorers#1](https://github.com/machine-exploration/explorers/pull/1)) so it moves with the rest.
- **Done when:** One `uv sync` installs everything, all tests pass, and `mechanics` points to `explorers`.

### I1 — The J-lens as an observable · done

- **Deliverable:**
  - A new read in the engine that gives observables access to gradients through the model.
  - Our own implementation of the Jacobian lens: `lens_l(h) = unembed(J_l · h)`, with `J_l` the average Jacobian from layer `l` to the last layer over a text corpus. We follow the published method and do not copy the reference code.
  - A fitted lens is part of the result, stored by content like any other.
- **Done when:** On a small open model, our lens gives the same top readouts as the [reference implementation](https://github.com/anthropics/jacobian-lens) within a stated tolerance.
- **Result:** In `mechanics` ([docs/jlens.md](https://github.com/machine-exploration/mechanics/blob/main/docs/jlens.md)). On a random 3-layer GPT-NeoX, our `J` matches the reference within 1.2e-7, and the top-5 readouts are identical at 54/54 (layer, position) pairs.

### I2 — Agent turns as examples · planned

- **Deliverable:** An adapter from agent episodes to `Examples`, with one example per agent turn and the token range of that turn. Any observable, including the J-lens, then runs on agent episodes without change.
- **Done when:** A core observable runs on recorded episodes and reads only the tokens of each turn.

### I3 — Any run as a series of states · planned

- **Deliverable:** `State` from LoRA adapters of hosted training runs (small, so dense checkpoints are cheap to store), as well as from Hugging Face revisions and models in memory. A `watch` process that measures each new checkpoint of a run, outside the trainer's process.
- **Done when:** A core observable runs over the checkpoints of a real RL run.

### I4 — Scale and names · planned

- **Deliverable:** Streaming reducers, so long agent contexts and many layers fit in memory. A stable name for a site (layer, component, position) that holds across checkpoints and models.

## Phase 1 — First questions

### Q0 — Quanta across training · done on a toy, next on Pythia

- **Question:** When, and in what order, does a model learn what it learns ([Michaud et al.](https://arxiv.org/abs/2303.13506))?
- **Done:** On a toy with 16 lookup tasks of Zipf frequencies, frequent tasks are learned first (rank correlation −0.74), and each is learned suddenly (median sharpness 0.74).
- **Next:** The same analysis on Pythia checkpoints, 70m to 1.4b. It gives the timeline of skills that Q1 is compared with.

### Q1 — When does the verbalizable space form? · next *(flagship)*

- **Question:** Finished language models hold a shared space of content they are disposed to say, which the J-lens reads. Across training, when does this space appear, how suddenly, at which layers, and does it form before, with or after the skills of Q0?
- **Deliverable:** The J-lens fitted at each of about 24 log-spaced Pythia checkpoints (70m to 410m, on one 12 GB GPU). At each checkpoint and layer: how well the lens predicts the model's own later output, and whether it reads out concepts that are implied but absent from the prompt. `onsets` and sharpness for each measure, next to the Q0 timeline.
- **First check:** Does the finished-model result hold on the final Pythia checkpoints? If not, that is the first result.
- **Done when:** A result note in this repository, with the run that reproduces it. Positive or negative.

### P0 — Do small models exploit tasks? · next

- **Question:** Do models that fit one GPU (0.6B to 4B) exploit impossible tasks, alone and with a shared channel? Because the tasks are impossible, any passing solution is an exploit, so the labels come for free.
- **Deliverable:** Exploit rate, spread through the channel, and activity for each model, with intervals, on 200 episodes per model. 50 labels checked by hand.
- **Done when:** One model exploits often enough to study. If none does, use larger open models on rented GPUs.

### P1 — Detectors at a matched false-positive rate · in review

- **Deliverable:** Detection rate at 1 %, 5 % and 10 % false-positive rate, AUROC, and intervals from a bootstrap over episodes. Spans carry a token range. [explorers#1](https://github.com/machine-exploration/explorers/pull/1).

### Q2 — What does an agent hold, turn by turn? · planned

- **Question:** Does the plan to exploit a task appear in the verbalizable space before the action? Does content move from one agent to another through the shared channel?
- **Needs:** I1, I2, P0.

### Q3 — Disposed to say, compared with said · planned

- **Question:** Is the gap between what the J-lens reads and what the model outputs a signal for exploits and deception?
- **Deliverable:** A gap observable for each turn, scored against the exploit labels with P1, next to a chain-of-thought monitor and an LLM judge at the same false-positive rate. Tested by intervention: steering along the readout changes the behaviour.
- **Needs:** Q2.

### Q4 — The space under training pressure · planned

- **Question:** During RL on exploitable tasks, when does the exploit plan enter the verbalizable space, and does training against a monitor move it out of that space?
- **Needs:** I3, Q3.

## Phase 2 — The survey · later

- **Deliverable:** Families of models trained with a full record: dense checkpoints, data order, several seeds and sizes, controlled changes to the data, and agents trained with RL. Published with their stores of measurements as an open dataset.
- **Starts when:** The instrument is used outside Machine Exploration, and the flagship result is published.
- **Design questions:** Which families and sizes; which data changes; storage format and budget; publication rules for runs where agents learn to exploit tasks.

## Phase 3 — The laws · later

- **Deliverable:** Regularities, tested on the survey, that predict when a feature, a capability or a behaviour forms, from the data, the architecture, the optimiser and the scale. The hypotheses come from learning mechanics ([open directions](https://learningmechanics.pub/openquestions/)): learning in discrete units (11), low-rank steps (12), bias toward simple solutions (3), a formal definition of features (4), similar representations across models (10).

## Known risks

- The finished-model result about a verbalizable space may not hold on small models or on Pythia. Q1 checks this first; a negative is still a result.
- Fitting the J-lens needs backward passes at each checkpoint. Small models and about 100 prompts keep this on one GPU; larger sizes need rented GPUs.
- Hugging Face must be reachable from the machine that runs Q0 and Q1.
- Small models may not exploit tasks (P0 tests this before Q2).
- The coordination label is a heuristic until a judge replaces it.
- Reading internals of long agent contexts may need streaming (I4).
