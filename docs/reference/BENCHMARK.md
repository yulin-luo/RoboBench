# Benchmark and metrics

## Dimensions

| Dimension | Representative capabilities | Evaluation type |
| --- | --- | --- |
| **Instruction Comprehension** | Explicit goals, implicit demands, cross-task navigation | Planning |
| **Perception and Reasoning** | Object attributes, spatial relations, temporal causality, robot type/view | Multiple-choice |
| **Generalized Planning** | Cross-embodiment, cross-object, cross-view, cross-attribute, world knowledge | Planning Q1/Q2/Q3 |
| **Affordance Reasoning** | Static affordance, dynamic affordance, navigation visual prompts | Multiple-choice |
| **Error Analysis** | High-level planning errors, low-level execution errors | Multiple-choice |

## Dataset and official results

The complete release contains 32 subtask manifests, 6,092 questions, and 37,126
image references. The 32 manifests correspond to the paper's 25 tasks: two
planning tasks are sharded by category at release time — `material_affordance`
into 5 material buckets (articulated / deformable / rigid / special /
multi-object) and `physical_attribute` into 4 attribute buckets (color / number /
shape / size) — while all other tasks map one-to-one (25 + 4 + 3 = 32).
`RoboBench-MCQ` covers 13 subtasks and 1,895 questions; `RoboBench-Planning`
covers 19 subtasks and 4,197 questions.

Official score tables and model-output JSON files are hosted separately to keep this repository lightweight:

- [lyl010221-pku/RoboBench-Results](https://huggingface.co/datasets/lyl010221-pku/RoboBench-Results)

The official leaderboard reports 18 evaluated MLLMs:

```text
GPT-5.4, GPT-5.2, GPT-5, GPT-4.1, GPT-4o,
Claude-Opus-4.7, Claude-Sonnet-4.6, Claude-Sonnet-4.5, Claude-Haiku-4.5,
Gemini-3.1-Pro, Gemini-2.5-Pro, Gemini-2.5-Flash,
Qwen3-VL-8B, Qwen2.5-VL-7B-Instruct, LLaVA-OneVision-7B,
RoboBrain-2.0-7B, RoboBrain-2.5-4B, MiMo-Embodied-7B
```

## Evaluation metrics

| Evaluator | What it checks |
| --- | --- |
| `multi_choice` | Multiple-choice answer normalization and exact scoring |
| `planning` | Q1 multi-step plans, Q2 single-step actions, Q3 state estimation |
| `point` | Distance between predicted and ground-truth coordinates |
| `iou` | Bounding-box intersection over union |
| `trajectory` | Multi-point trajectory comparison |

Planning evaluation can call an evaluator model, configured by `evaluation.planning.eval_model`, to judge action feasibility and task completion.
