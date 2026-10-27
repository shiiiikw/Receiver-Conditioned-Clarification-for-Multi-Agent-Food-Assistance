# Receiver-Conditioned Clarification for Multi-Agent Food Assistance

Reference implementation of the fan-out multi-agent topology described in Appendix G of the paper
*Receiver-Conditioned Clarification for Multi-Agent Food Assistance*, evaluated on the
[HoosierHelp](https://github.com/Yiyang-Ian-Li/HoosierHelp) resource-navigation benchmark.

```
                 ┌──────────── access receiver ──────────┐
 person ⇄ intake ├──────────── schedule receiver ────────┤ → joint planner (deterministic) → retriever → final recommendation
                 └──────────── logistics receiver ───────┘
```

* **Intake** reads the dialogue and emits candidate interpretations of the person's needs (`need × field` premises)
  with empirical support (one greedy plus `S` sampled proposals, grouped into at most `K` candidates).
* **Receivers** (access and eligibility; schedule; logistics) never see the dialogue. Each reads only the intake
  candidates projected to its own fields plus domain evidence from the Indiana 211 index, and reports, per candidate,
  its admissible decision and its *obligations* (facts still required before the decision is supported).
  Hard constraints make the admissible decision a deterministic projection of the candidate; obligations carry the
  receiver's domain judgement.
* **Joint planner** forms the product of intake and receiver candidates (at most `OMEGA_MAX = 16` joint
  interpretations, renormalised), maps every interpretation to a search program and exposes the obligations and the
  dependency footprint of every clarification question.
* **Controller** decides at each step whether to commit to the greedy interpretation or to ask the person one
  question. Our controller (`voi_det`) is a deterministic, decision-relevant value-of-information rule with lookahead
  depth `H = 2`, per-question cost `c_item = 0.04` and fallback-question cost `c_fallback = 0.06`.
  Baseline controllers (SAGE, ATD, ProCoT, LLM-VoI, myopic VoI, disagreement, max-uncertainty, fixed-order,
  no-clarification) share the same workflow and differ only in this decision.
* **Repair.** After an empty search the workflow runs a counterfactual relaxation diagnosis over the ladder
  time-window < schedule < location < intake method < documents < eligibility (at most 6 evaluations, at most 3
  fallback questions) and applies the person's answers consistently to every joint interpretation.
* **Budgets.** At most 60 backbone calls per step and 180 per episode (exceeding degrades invocations to greedy-only),
  at most 2 repair rounds, and a receiver is re-run only when its projected input changed.

The backbone is a frozen `Qwen/Qwen3-8B` served by vLLM; the simulated user is the benchmark's own user prompt run
on an OpenAI model. Nothing is trained.

## Repository layout

```
rclup/                     the system
  fanout2.py               receivers, hard-constraint validation, joint planner, relaxation ladder, budgets (App. G)
  workflow.py              episode driver, candidate packets, controllers (ours and the baselines), repair logic
  roles.py                 role prompts, tool schemas, parsers and canonicalisers
  packets.py               candidate packets (support weights, residual mass, provenance)
  backbone.py              vLLM sampling client
  npc.py                   simulated user (OpenAI), reply cache, cost metering
  cost.py                  OpenAI cost ledger with a hard cap
  envs/hoosier_env.py      HoosierHelp environment wrapper (native tool, turn cap, scorer)
  data/hoosier_build.py    builds the evaluation / training user specs from the benchmark
  run_episodes.py          command-line entry point
scripts/
  start_vllm.sh            start the backbone server on a GPU machine
  worker.sbatch            SLURM variant: serve the backbone and run a driver on the same node
  run_fanout.sh            smoke gate + full evaluation of every controller for one seed
  fanout2_gate.py          structural check of a smoke run
  strip_errors.py          drop failed episodes from a run so that it can be resumed
  summarize.py             controller × seed table of a finished protocol directory
tests/test_parsing.py      offline unit tests of the receiver output parsing
vendor/hoosierhelp         HoosierHelp benchmark (git submodule, pinned)
```

## Requirements

* Python ≥ 3.11 for the driver (any machine that can reach the backbone server and the OpenAI API).
* One GPU with ≥ 24 GB memory for the backbone server (`vllm`, bf16 weights of Qwen3-8B).
* An OpenAI API key for the simulated user (`gpt-4.1-mini` by default; a full 240-episode run of one controller costs
  well under one US dollar, and replies are cached on disk so repeated runs are free).

## Setup

1. Clone with the benchmark submodule:

   ```bash
   git clone --recurse-submodules https://github.com/shiiiikw/Receiver-Conditioned-Clarification-for-Multi-Agent-Food-Assistance.git
   cd Receiver-Conditioned-Clarification-for-Multi-Agent-Food-Assistance
   # if you cloned without --recurse-submodules:
   git submodule update --init
   ```

2. Install the driver dependencies (and `vllm` in the environment that will serve the backbone):

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   pip install "vllm>=0.10"        # server side only
   ```

3. Configure through environment variables (all optional except the API key):

   | variable | meaning | default |
   |---|---|---|
   | `OPENAI_API_KEY` | key for the simulated user (or set `RCLUP_OPENAI_KEY_FILE` to a file containing `OPENAI_API_KEY=...`) | required |
   | `RCLUP_RUNS` | output root: records, summaries, logs, the user-reply cache and the built data | `<repo>/runs` |
   | `RCLUP_MODEL_DIR` | backbone: Hugging Face id or a local snapshot directory (tokenizer on the driver, weights on the server) | `Qwen/Qwen3-8B` |
   | `RCLUP_VLLM_URL` | address of a running backbone server, e.g. `http://gpu-host:8000/v1`; if unset the driver reads the status file written by the start scripts under `RCLUP_RUNS` | unset |
   | `RCLUP_NPC_MODEL` | OpenAI model for the simulated user | `gpt-4.1-mini` |
   | `RCLUP_HOOSIER_DIR` | location of the HoosierHelp checkout | `<repo>/vendor/hoosierhelp` |
   | `HF_HUB_OFFLINE=1` | load the tokenizer strictly from the local cache | unset |

   The driver needs the Qwen3-8B tokenizer locally (it renders the chat template client-side); with the default
   `RCLUP_MODEL_DIR` it is downloaded from the Hugging Face Hub on first use.

4. Start the backbone server on a GPU machine:

   ```bash
   scripts/start_vllm.sh                      # serves RCLUP_MODEL_DIR as "qwen3-8b" on port 8000 (VLLM_PORT)
   # or, on SLURM, edit the #SBATCH lines in scripts/worker.sbatch and submit it together with a driver script (step 6)
   ```

   `VLLM_UTIL`, `VLLM_MAXLEN` and `VLLM_DTYPE` tune memory use (use `VLLM_DTYPE=float16` on GPUs without bf16).

5. Build the evaluation user specs from the pinned benchmark revision (48 base users × 5 behaviours = 240 episodes):

   ```bash
   python -m rclup.data.hoosier_build        # writes $RCLUP_RUNS/data/hoosier/{eval_specs,train_specs}.jsonl
   ```

   The training specs are used only to estimate the question-group priors of the budget-aware controller.

## Running

Smoke test (6 episodes, our controller), then the structural gate:

```bash
export OPENAI_API_KEY=...            # or RCLUP_OPENAI_KEY_FILE=...
export RCLUP_VLLM_URL=http://<gpu-host>:8000/v1
python -m rclup.run_episodes --controller voi_det --limit 6 --out smoke/voi_det
python scripts/fanout2_gate.py $RCLUP_RUNS/smoke/voi_det
```

Full evaluation of all controllers for one seed (resumable; re-running skips finished runs and finished episodes):

```bash
scripts/run_fanout.sh 0 atomic      # budgeted atomic questions, 8 agent turns, ten controllers
scripts/run_fanout.sh 0 official    # the benchmark's own protocol (fixed multi-item intake questions), six controllers
python scripts/summarize.py $RCLUP_RUNS/eval/atomic
```

`scripts/run_fanout.sh` reads `RCLUP_WORKERS` (concurrent episodes, default 10), `RCLUP_CAP_USD` (hard cap on the
shared OpenAI ledger, default 450) and `RCLUP_CONTROLLERS` (space-separated subset). On SLURM:

```bash
sbatch scripts/worker.sbatch scripts/run_fanout.sh 0 atomic
```

A single configuration, with the paper's parameters spelled out:

```bash
python -m rclup.run_episodes --controller voi_det --variant atomic --intake-policy atomic --max-turns 8 \
    --behaviors normal,rambling,impatience,self_contradictory,unsupported_request --limit 240 --seed 0 \
    --s-draws 4 --k-max 3 --omega-max 16 --voi-depth 2 --c-item 0.04 --c-fallback 0.06 --p-fallback 0.5 \
    --max-calls-episode 180 --out eval/atomic/seed0/hoosier_voi_det_b5
```

The listed values are the defaults under the fan-out topology; `python -m rclup.run_episodes --help` documents every
option, including the component switches (`--modules-off`).

### Outputs

Each run directory under `RCLUP_RUNS` contains `records.jsonl` (one line per episode: transcript, score, questions
asked, the full invocation trace with every packet, and the error message if the episode failed) and `summary.json`
(end-to-end match, questions per episode, agent turns, per-case and per-behaviour breakdowns, backbone token usage and
OpenAI cost). Failed episodes count as failures in the summary; to retry them, drop them and re-run the same command:

```bash
python scripts/strip_errors.py --pattern "APIConnectionError" $RCLUP_RUNS/eval/atomic/seed0/hoosier_voi_det_b5
```

## Tests

```bash
pytest -q tests
```

## Citation

Please cite the paper *Receiver-Conditioned Clarification for Multi-Agent Food Assistance* when using this code, and the
HoosierHelp benchmark for the environment and the simulated user.
