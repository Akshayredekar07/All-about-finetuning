## **Stage 09: Agentic Fine-Tuning and RL Harnesses**

**Libraries:** ART, RULER, ASTRA, CM2, TRL, vLLM, verl, LangGraph, Unsloth, Weights & Biases Training

**Outcome:** you can train an agent that uses tools over multiple turns, with a reward that is either verifiable or checklist-decomposed. You can evaluate whether the agent got better rather than merely longer.

### **1. Scope and Exit Criterion**

**Scope:** the hard part is not the algorithm. Agentic post-training fails on three things, in this order: no environment to roll out in, a reward that rewards the wrong behaviour, and an evaluation that cannot distinguish a better agent from a more verbose one. Every framework in this stage is a solution to one of those three problems, not to the gradient maths.

**Exit criterion:** you can set up a multi-turn tool-use training run with a deterministic environment and a verifiable or checklist-decomposed reward. You can then show, with held-out environments and repeated runs, that the trained agent is better and not just more verbose.

### **2. Why This Is a Separate Stage**

**Overview:** through Stage 5 you trained a model that answers. This stage trains a model that **acts**: it calls tools, reads their output, decides what to do next, and stops when the task is done. That changes four things:

- **Data type:** trajectories, not prompt-response pairs.
- **Reward type:** outcomes, not preferences.
- **Training loop:** rollouts against an environment, so the trainer needs a server.
- **Evaluation:** task completion, not answer similarity.

**Post-training pipeline, for reference:**

```
Base Pretrained Model
        |
SFT: Supervised Fine-Tuning
        |
Preference Optimization (DPO / SimPO / KTO / ORPO)
        |
Reasoning RL (GRPO / DAPO / Dr.GRPO / GSPO / GRPO-λ / RLVR)
        |
Agentic RL (multi-turn, tool use, environment-driven), this stage
        |
Quantize + Export + Serve
```

**Why it needs infrastructure:** the agentic RL step in that diagram is this stage, and it is the one that needs infrastructure you do not build from scratch.

### **3. The Four Named Systems**

| **System** | **What it contributes** | **Released** |
|---|---|---|
| **ART** (OpenPipe) | Agent Reinforcement Trainer: GRPO over multi-turn agent trajectories, with the trainer and the inference server split across processes | 2026 |
| **RULER** | Relative Universal LLM-Elicited Rewards: an LLM judge that *ranks* groups of trajectories instead of scoring them absolutely | 2026 |
| **ASTRA** | Fully automated synthesis of agentic trajectories and executable, rule-verifiable tool environments | 30 Jan 2026 |
| **CM2** | Checklist rewards: open-ended objectives decomposed into fine-grained binary criteria with evidence grounding | NeurIPS 2026 |

**Extra pattern:** one more pattern, observed in a deployed frontier model rather than in a paper, is **orchestrator plus frozen sub-agents**.

### **4. ART and RULER**

**Why ART first:** ART is the most directly usable of the four for a learner, because it runs GRPO over real multi-turn rollouts against a live vLLM server.

**Architecture:** the architecture matters. ART's server process hosts the model in vLLM, trains with GRPO, saves a LoRA adapter, and reloads that adapter into vLLM for the next generation round. The client process owns the environment and the trajectory collection. Each rollout captures the full message sequence: system, user, assistant, tool calls and tool responses. The credit assignment question therefore becomes "which tool call in this trajectory produced the outcome".

**RULER as the reward:** instead of asking a judge for a score from 1 to 5 and averaging, RULER asks the judge to *rank* the trajectories in a group. GRPO only needs relative advantage within a group, so ranking is enough, and it removes the calibration problem that makes scalar judge rewards unstable. ART's own results report matching or exceeding handcrafted rewards on three of four benchmarks.

**Ecosystem:** ART ships with a LangGraph integration. Its examples include "Qwen 3.8 27B learns to search emails using RULER" and a 2048 environment. Its AutoRL mode generates the training inputs automatically and uses RULER to produce rewards with no human-labelled data at all, which is the cheapest possible entry point if you have a simulator worth playing. Weights & Biases Training ships ART plus RULER with a managed serverless RL backend, and Unsloth's GRPO trainer is the underlying implementation.

**Round structure:**

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

**Worked example, group-relative advantage:** Karan's agent samples 4 trajectories for one prompt, and RULER's ranking is converted to rewards of 0.2, 0.4, 0.6 and 0.8. GRPO turns these into advantages by subtracting the group mean and dividing by the group standard deviation (population form here).

Step 1: compute the mean.

$$\mu = \frac{0.2 + 0.4 + 0.6 + 0.8}{4} = \frac{2.0}{4} = 0.5$$

Step 2: compute each deviation from the mean.

$$0.2 - 0.5 = -0.3$$

$$0.4 - 0.5 = -0.1$$

$$0.6 - 0.5 = 0.1$$

$$0.8 - 0.5 = 0.3$$

Step 3: square each deviation.

$$0.09,\ 0.01,\ 0.01,\ 0.09$$

Step 4: average the squares to get the variance.

$$\sigma^2 = \frac{0.09 + 0.01 + 0.01 + 0.09}{4} = \frac{0.20}{4} = 0.05$$

Step 5: take the square root.

$$\sigma = \sqrt{0.05} \approx 0.2236$$

Step 6: divide each deviation by $\sigma$.

$$\frac{-0.3}{0.2236} \approx -1.342$$

$$\frac{-0.1}{0.2236} \approx -0.447$$

$$\frac{0.1}{0.2236} \approx 0.447$$

$$\frac{0.3}{0.2236} \approx 1.342$$

$$\boxed{A = [-1.342,\ -0.447,\ 0.447,\ 1.342]}$$

Only the relative order within the group matters, which is why a ranking is enough.

### **5. ASTRA: Synthesising the Environment**

**Overview:** ASTRA solves the part nobody wants to build: a realistic tool environment with a deterministic reward. Its pipeline has two halves.

**Trajectory synthesis:** it starts from MCP server documentation and builds a tool dependency graph. It then generates tasks, collects trajectories, and scores them to produce SFT data.

```
mcp_servers.jsonl | Graph construction | Task generation | LLM interaction | Reward assessment | SFT data
```

**Environment synthesis:** it turns question-answer pairs into executable environments.

```
QA data | Question decomposition | Tool necessity check | Verification | Environment synthesis | Tool merging
```

**Scale:** the scale is the point. The released models were trained against a tool pool of **1,585 MCP servers, 19,036 tools across 41 domains**, with schema-level dependency analysis used to generate executable, parameter-satisfiable tool chains. The resulting environments are Python programs, validated by sandboxed execution. That is what makes the rewards deterministic and multi-turn RL possible without a human in the loop.

**Training methodology:** SFT first, to establish tool-calling behaviour and multi-turn priors. Then online multi-turn RL with verifiable rewards, using adaptive batch filling, batch-level token-loss averaging, and an **F1-style trajectory-level reward** that balances completion against interaction efficiency. Recall is the fraction of required sub-tasks solved, precision divides solved sub-tasks by tool invocations, and the reward is their harmonic mean. That last part is the design idea worth stealing even if you never run ASTRA, because it penalises the agent for flailing, which a naive task-completion reward does not.

**Released models:** `ASTRA-14B-Thinking-v1` (from Qwen3-14B) and `ASTRA-32B-Thinking-v1` (from Qwen3-32B), with the full pipeline, environments and models published.

**Worked example, F1-style trajectory reward:** a task has 4 required sub-tasks. Ananya's agent solves 3 of them using 6 tool invocations.

Step 1: compute recall.

$$R = \frac{\text{solved}}{\text{required}} = \frac{3}{4} = 0.75$$

Step 2: compute precision.

$$P = \frac{\text{solved}}{\text{tool invocations}} = \frac{3}{6} = 0.5$$

Step 3: take the harmonic mean.

$$F_1 = \frac{2PR}{P + R}$$

Step 4: substitute the values.

$$F_1 = \frac{2 \times 0.5 \times 0.75}{0.5 + 0.75} = \frac{0.75}{1.25}$$

$$\boxed{F_1 = 0.6}$$

Now Vikram's agent solves the same 3 sub-tasks but flails through 12 tool invocations.

$$P = \frac{3}{12} = 0.25$$

$$F_1 = \frac{2 \times 0.25 \times 0.75}{0.25 + 0.75} = \frac{0.375}{1.0}$$

$$\boxed{F_1 = 0.375}$$

Both agents have the same recall, but the flailing one is penalised.

### **6. CM2: Checklist Rewards for Open-Ended Tasks**

**Overview:** CM2 targets the case where you cannot write a verifier at all. Instead of replacing outcome rewards with checklists, it decomposes each turn's intended behaviour into fine-grained binary criteria with explicit evidence grounding and structured metadata. This turns open-ended judging into stable classification-style decisions.

**Two design choices that make it work:**

- **Sparse reward assignment with dense evaluation criteria:** only a few of the criteria carry reward, but every criterion is evaluated. The model gets a consistent signal about what good behaviour looks like without being trained on a noisy dense scalar.
- **Scalable LLM-simulated tool environment:** it handles 5,000+ tools, avoiding the engineering cost of building a real tool set.

**Results:** measured from an 8B base model trained on an 8K-example RL dataset, CM2 improves over its SFT counterpart by 8 points on τ²-Bench, 10 points on BFCL-V4 and 12 points on ToolSandbox.

**Code:** the code is built on `verl`, with a checklist variant of it.

**Transferable lesson:** it is narrower than the framework. When you cannot verify an outcome, write the rubric first, decompose it into binary checks with evidence attached, and reward only the checks you are confident about.

### **7. Orchestrator and Frozen Sub-Agents**

**Overview:** this is the pattern observed in deployed frontier models, and the one Anthropic's agentic research roles now describe as their core problem. It is decomposition rather than depth.

**The idea:** train an **orchestrator** with RL to decompose a task and delegate to frozen copies of a sub-agent, rather than training one model to do everything. Kimi K3.1 Pro's public description is an orchestrator trained to decompose a request and delegate to frozen sub-agent copies. The vendor-reported result is a jump on BrowseComp from 60.6% to 78.4% for the swarm over the single agent. Treat the specific numbers as vendor-reported: the architecture pattern is the part worth copying.

**Diagram:**

```
task
  |
Orchestrator (trained): decides solve or split
  |
  +-- decompose and delegate | Sub-agent A (frozen)
  +-- decompose and delegate | Sub-agent B (frozen)
  |
assemble results from the sub-agents
```

**Engineering consequences:** the sub-agents stay frozen, so you can reuse a single set of specialist models for many orchestrator runs. The orchestration policy is also a much smaller training problem than end-to-end agent capability. The cost is a context-budget problem: the orchestrator's context must hold every sub-agent's report, so delegation granularity becomes a first-class hyperparameter rather than a prompt detail.

### **8. Doing It Inside TRL**

**Overview:** TRL has an environment-driven mode that makes a local version of this possible without a separate framework. `GRPOTrainer` accepts an `environment_factory` instead of a `train_dataset`. In that case the environment owns the prompt generation and the trainer drives `max_steps` rounds of reset and step.

**Configuration:**

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

**The `environment_factory` contract:** it is simple. Each row of the dataset routes to one environment for the rollout, and the environment returns the prompt, the tool observations, and the final reward. That is the seam where ASTRA-style synthesised environments, a real MCP server, or a simulator plugs in.

### **9. Agent Evaluation**

**Overview:** an agent that scores better on a similarity metric is not necessarily better. The current benchmark set, and what each one measures:

| **Benchmark** | **What it measures** | **Notes** |
|---|---|---|
| **τ²-Bench (tau2-Bench)** | Multi-turn tool use with a customer-service simulator | The hardest of the three for a first agent |
| **BFCL-V4** | Function-calling correctness and multi-turn robustness | The standard Berkeley function-calling leaderboard |
| **ToolSandbox** | State-dependent tool execution in a simulated environment | Rewards actual state changes, not just call syntax |
| **RULER** | Long-context retrieval under interference | Also the reward model used by ART |
| **NIAH** | Needle-in-a-haystack | The degenerate case of the above |
| **SWE-Bench Verified** | End-to-end software-engineering tasks in a real repo | The capstone-grade benchmark for coding agents |

**Practical evaluation protocol:**

- Hold out environments, not just task instances. If you tune on the same tool set you evaluate on, you have measured memorisation.
- Report the number of tool calls per solved task, not only the solve rate. An agent that solves 60% of tasks with 4 calls each is usually better than one that solves 70% with 30 calls each.
- Report partial credit and failure mode. "Called the wrong tool" and "called the right tool with the wrong argument" need different fixes.
- Re-run each configuration at least three times. Agent variance is high enough that a single run is noise.
- Keep a fixed judge model and a fixed judge prompt. Changing the judge between runs silently changes the metric.

**Worked example, calls per solved task:** Meera compares the two agents from the second bullet over 100 tasks each.

Step 1: count solved tasks for each agent.

$$0.60 \times 100 = 60 \qquad 0.70 \times 100 = 70$$

Step 2: count total tool calls for each agent.

$$60 \times 4 = 240 \qquad 70 \times 30 = 2100$$

Step 3: divide total calls by tasks solved overall, to get the cost per solved task (here the per-task call counts are already given, so the totals confirm the ratio).

$$\frac{2100}{240} = 8.75$$

$$\boxed{\text{the 70\% agent uses 8.75 times as many tool calls in total}}$$

### **10. Agentic Fine-Tuning Pre-Flight Checklist**

- Is the environment implemented and does it run standalone, with a deterministic seed?
- Is the reward verifiable, or decomposed into binary criteria with evidence attached?
- Is the judge (if any) version-pinned, and is its prompt stored alongside the results?
- Are tool calls executed in a sandbox, so that the model cannot reach the network directly?
- Does each trajectory record the full message history, including tool calls and tool responses?
- Are tool-call count and tokens-per-task logged alongside the reward?
- Are held-out environments reserved before any tuning?
- Does the base model support the tool format natively, or was a format layer added during SFT?
- Does the SFT warm-start exist before RL begins? (Cold-start RL on a tool-using policy is a waste of GPU hours.)
- Is the comparison baseline the SFT model, evaluated with the same harness and the same judge?
- Is variance measured, with at least three runs per configuration?

**Next:** Stage 10.