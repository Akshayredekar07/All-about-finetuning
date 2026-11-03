# Stage 9 — Agentic Fine-Tuning and RL Harnesses

**Outcome:** you can train an agent that uses tools over multiple turns, with a reward that is either verifiable or checklist-decomposed, and you can evaluate whether the agent got better rather than merely longer.

> [!danger] The hard part is not the algorithm
> Agentic post-training fails on three things, in this order: no environment to roll out in, a reward that rewards the wrong behaviour, and an evaluation that cannot distinguish a better agent from a more verbose one. Every framework in this stage is a solution to one of those three problems, not to the gradient maths.

## Why this is a separate stage

Through Stage 5 you trained a model that answers. Stage 9 trains a model that **acts**: it calls tools, reads their output, decides what to do next, and stops when the task is done. That changes the data type (trajectories, not prompt–response pairs), the reward type (outcomes, not preferences), the training loop (rollouts against an environment, so the trainer needs a server), and the evaluation (task completion, not answer similarity).

The pipeline for the whole post-training stack, for reference:

```
Base Pretrained Model
        ↓
[Stage 1] SFT — Supervised Fine-Tuning
        ↓
[Stage 2] Preference Optimization (DPO / SimPO / KTO / ORPO)
        ↓
[Stage 3] Reasoning RL (GRPO / DAPO / Dr.GRPO / GSPO / GRPO-λ / RLVR)
        ↓
[Stage 4] Agentic RL (multi-turn, tool use, environment-driven)
        ↓
Quantize + Export + Serve
```

Stage 4 of that diagram is this stage, and it is the one that needs infrastructure you do not build from scratch.

## The four named systems

| System | What it contributes | Released | Reference |
|---|---|---|---|
| **ART** (OpenPipe) | Agent Reinforcement Trainer: GRPO over multi-turn agent trajectories, with the trainer and the inference server split across processes | 2026 | [101] |
| **RULER** | Relative Universal LLM-Elicited Rewards: an LLM judge that *ranks* groups of trajectories instead of scoring them absolutely | 2026 | [101] |
| **ASTRA** | Fully automated synthesis of agentic trajectories and executable, rule-verifiable tool environments | 30 Jan 2026 | [97][98] |
| **CM2** | Checklist rewards: open-ended objectives decomposed into fine-grained binary criteria with evidence grounding | NeurIPS 2026 | [99][100] |

Plus one pattern observed in a deployed frontier model, rather than a paper: **orchestrator plus frozen sub-agents**.

## ART and RULER

ART is the most directly usable of the four for a learner, because it runs GRPO over real multi-turn rollouts against a live vLLM server.

The architecture matters. ART's server process hosts the model in vLLM, trains with GRPO, saves a LoRA adapter, and reloads that adapter into vLLM for the next generation round; the client process owns the environment and the trajectory collection. Each rollout captures the full message sequence — system, user, assistant, and tool calls and tool responses — so the credit assignment question becomes "which tool call in this trajectory produced the outcome".

RULER supplies the reward. Instead of asking a judge for a score from 1 to 5 and averaging, it asks the judge to *rank* the trajectories in a group. GRPO only needs relative advantage within a group, so ranking is enough, and it removes the calibration problem that makes scalar judge rewards unstable. ART's own results report matching or exceeding handcrafted rewards on three of four benchmarks.

ART ships with a LangGraph integration, and its examples include "Qwen 3.8 27B learns to search emails using RULER" and a 2048 environment. Its AutoRL mode generates the training inputs automatically and uses RULER to produce rewards with no human-labelled data at all — which is the cheapest possible entry point if you have a simulator worth playing. Weights & Biases Training ships ART plus RULER with a managed serverless RL backend, and Unsloth's GRPO trainer is the underlying implementation.

```python
# Conceptual shape of an ART-style round
# (the library owns this loop; the point is to know what it does)
for step in range(max_steps):
    trajectories = env.reset_batch(num_prompts)          # one environment instance per prompt
    for turn in range(max_turns):
        completions = server.generate(trajectories.state, model=base + adapter)
        trajectories.step(completions, env)               # tool calls executed, results appended
    rewards = ruler.rank(trajectories)                    # relative ranking inside each group
    adapter = grpo_trainer.train(trajectories, rewards)   # LoRA updated, reloaded into the server
```

## ASTRA: synthesising the environment

ASTRA solves the part nobody wants to build: a realistic tool environment with a deterministic reward. Its pipeline has two halves.

**Trajectory synthesis** starts from MCP server documentation and builds a tool dependency graph, then generates tasks, collects trajectories, and scores them to produce SFT data:

```
mcp_servers.jsonl → Graph construction → Task generation → LLM interaction → Reward assessment → SFT data
```

**Environment synthesis** turns question–answer pairs into executable environments:

```
QA data → Question decomposition → Tool necessity check → Verification → Environment synthesis → Tool merging
```

The scale is the point: the released models were trained against a tool pool of **1,585 MCP servers, 19,036 tools across 41 domains**, with schema-level dependency analysis used to generate executable, parameter-satisfiable tool chains. The resulting environments are Python programs, validated by sandboxed execution, which is what makes the rewards deterministic and multi-turn RL possible without a human in the loop.

The training methodology is SFT to establish tool-calling behaviour and multi-turn priors, then online multi-turn RL with verifiable rewards, using adaptive batch filling, batch-level token-loss averaging, and an **F1-style trajectory-level reward** that balances completion against interaction efficiency: recall is the fraction of required sub-tasks solved, precision divides solved sub-tasks by tool invocations, and the reward is their harmonic mean. That last part is the design idea worth stealing even if you never run ASTRA — it penalises the agent for flailing, which a naive task-completion reward does not.

The released models are `ASTRA-14B-Thinking-v1` (from Qwen3-14B) and `ASTRA-32B-Thinking-v1` (from Qwen3-32B), with the full pipeline, environments and models published [97][98].

## CM2: checklist rewards for open-ended tasks

CM2 targets the case where you cannot write a verifier at all. Instead of replacing outcome rewards with checklists, it decomposes each turn's intended behaviour into fine-grained binary criteria with explicit evidence grounding and structured metadata, turning open-ended judging into stable classification-style decisions.

Two design choices make it work. **Sparse reward assignment with dense evaluation criteria**: only a few of the criteria carry reward, but every criterion is evaluated, so the model gets a consistent signal about what good behaviour looks like without being trained on a noisy dense scalar. And a **scalable LLM-simulated tool environment** that handles 5,000+ tools, avoiding the engineering cost of building a real tool set.

Measured from an 8B base model trained on an 8K-example RL dataset, CM2 improves over its SFT counterpart by 8 points on τ²-Bench, 10 points on BFCL-V4 and 12 points on ToolSandbox [99]. The code is built on `verl` v0.6.1 with a `verl_v0.6.1_checklist` variant [100].

For a learner, the transferable lesson is narrower than the framework: when you cannot verify an outcome, write the rubric first, decompose it into binary checks with evidence attached, and reward only the checks you are confident about.

## Orchestrator and frozen sub-agents

The pattern the source document observed in deployed frontier models, and the one Anthropic's agentic research roles now describe as their core problem, is decomposition rather than depth.

The idea: train an **orchestrator** with RL to decompose a task and delegate to frozen copies of a sub-agent, rather than training one model to do everything. Kimi K3.1 Pro's public description is an orchestrator trained to decompose a request and delegate to frozen sub-agent copies; the vendor-reported result is a jump on BrowseComp from 60.6% to 78.4% for the swarm over the single agent [81]. Treat the specific numbers as vendor-reported — the architecture pattern is the part worth copying.

```
             ┌───────────────────────────┐
  task ───►  │  Orchestrator (trained)   │ ───► decompose ───► ┌──────────────┐
             │  decides: solve / split   │                    │ Sub-agent A  │ frozen
             └───────────────────────────┘ ───► delegate ───►  ├──────────────┤
                                  │                         │ Sub-agent B  │ frozen
                                  └── assemble ◄──────────── └──────────────┘
```

Two engineering consequences. The sub-agents stay frozen, so you can reuse a single set of specialist models for many orchestrator runs, and the orchestration policy is a much smaller training problem than end-to-end agent capability. The cost is a context-budget problem: the orchestrator's context must hold every sub-agent's report, so delegation granularity becomes a first-class hyperparameter rather than a prompt detail.

## Doing it inside TRL

TRL 1.17.1 grew an environment-driven mode that makes a local version of this possible without a separate framework: `GRPOTrainer` accepts an `environment_factory` instead of a `train_dataset`, in which case the environment owns the prompt generation and the trainer drives `max_steps` rounds of reset/step [103].

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir="grpo-agent",
    loss_type="grpo",
    importance_sampling_level="token",
    beta=0.0,
    num_generations=8,
    max_steps=500,
    use_vllm=True,
    vllm_mode="server",
    # vllm_importance_sampling_mode="token_truncate"  # correct train/inference mismatch
)
```

The `environment_factory` contract is simple: each row of the dataset routes to one environment for the rollout, and the environment returns the prompt, the tool observations, and the final reward. That is the seam where ASTRA-style synthesised environments, a real MCP server, or a simulator plugs in.

## Agent evaluation

An agent that scores better on a similarity metric is not necessarily better. The current benchmark set, and what each one measures:

| Benchmark | What it measures | Notes |
|---|---|---|
| **τ²-Bench (tau2-Bench)** | Multi-turn tool use with a customer-service simulator | The hardest of the three for a first agent |
| **BFCL-V4** | Function-calling correctness and multi-turn robustness | The standard Berkeley function-calling leaderboard |
| **ToolSandbox** | State-dependent tool execution in a simulated environment | Rewards actual state changes, not just call syntax |
| **RULER** | Long-context retrieval under interference | Also the reward model used by ART |
| **NIAH** | Needle-in-a-haystack | The degenerate case of the above |
| **SWE-Bench Verified** | End-to-end software-engineering tasks in a real repo | The capstone-grade benchmark for coding agents |

A practical evaluation protocol for agent work:

1. Hold out environments, not just task instances. If you tune on the same tool set you evaluate on, you have measured memorisation.
2. Report the number of tool calls per solved task, not only the solve rate. An agent that solves 60% of tasks with 4 calls each is usually better than one that solves 70% with 30 calls each.
3. Report partial credit and failure mode. "Called the wrong tool" and "called the right tool with the wrong argument" need different fixes.
4. Re-run each configuration at least three times. Agent variance is high enough that a single run is noise.
5. Keep a fixed judge model and a fixed judge prompt. Changing the judge between runs silently changes the metric.

## Agentic fine-tuning pre-flight checklist

- [ ] The environment is implemented and runs standalone, with a deterministic seed.
- [ ] The reward is verifiable, or decomposed into binary criteria with evidence attached.
- [ ] The judge (if any) is version-pinned and its prompt is stored alongside the results.
- [ ] Tool calls are executed in a sandbox, and the model cannot reach the network directly.
- [ ] Each trajectory records the full message history, including tool calls and tool responses.
- [ ] Tool-call count and tokens-per-task are logged alongside the reward.
- [ ] Held-out environments are reserved before any tuning.
- [ ] The base model supports the tool format natively, or a format layer was added during SFT.
- [ ] The SFT warm-start exists before RL begins. Cold-start RL on a tool-using policy is a waste of GPU hours.
- [ ] The comparison baseline is the SFT model, evaluated with the same harness and the same judge.
- [ ] Variance is measured: at least three runs per configuration.
