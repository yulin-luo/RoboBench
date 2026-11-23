# External RoboBench evaluations

These results come from the cited evaluation reports. Compare scores within
one protocol. The [ECCV 2026 leaderboard](https://robo-bench.github.io/#leaderboard)
uses the RoboBench release's own tasks, prompts, and scoring.

## ACE Robotics · Natural-language planning

[ACE-Brain-0.7, Appendix A.2 and Tables 2, 9, and 10](https://arxiv.org/pdf/2611.26519)
evaluates 3,979 items after excluding navigation and 28 items with unverified
reference conversions. The models receive visual observations. Q1 and Q2
use plain-language responses and references; the prompt omits the skill
library and examples. A Qwen3.8-Flash judge scores Q1 and Q2 against the
references. Q3 retains binary scoring. The aggregate weights each item
equally.

| Model | Q1 task decomposition (1,945) | Q2 next subtask (842) | Q3 step completion (1,192) | Overall |
| --- | ---: | ---: | ---: | ---: |
| ACE-Brain-0.7 | 67.7 | 52.1 | 77.8 | 67.4 |
| GPT-6 Astra | 74.5 | 34.7 | 74.1 | 66.0 |
| Claude Opus 5.5 | 77.8 | 39.0 | 75.4 | 68.9 |

Table 2 also reports results for embodied comparison models on this setting,
including RoboBrain 2.0-32B (56.1), RoboBrain 2.5-8B (55.2),
Cosmos-Reason2-32B (55.0), MiMo-Embodied-7B (54.7),
Hy-Embodied-VLM-1.0 (54.6), SenseNova-SI 1.1-8B (52.9),
PhysBrain 1.5-8B (51.4), Embodied-R1.5-8B (50.4), and RynnBrain 1.1-9B (40.0).
These are comparisons reported by ACE Robotics. For matching runs, the paper
specifies the protocol at a high level. Exact item IDs, prompt and conversion
templates, image preprocessing, and the judge rubric are needed to reproduce
it.

## Li Auto · Planning, Perception, Affordance, and Error

| Model | Planning | Perception | Affordance | Error |
| --- | ---: | ---: | ---: | ---: |
| ME-VLM 4B | 40.1 | 41.8 | 47.0 | — |
| ME-VLM 35B-A3B | 53.7 | 52.2 | 60.7 | 52.7 |

[ME-VLM, Table 1](https://arxiv.org/html/2609.24526v1#S6) reports these
figures. [ME-Brain-1.0, §7.1.1](https://arxiv.org/html/2609.24271v1#S7.SS1.SSS1)
reports the same ME-VLM results in its cognitive-core evaluation. The 4B
Error result is listed as unavailable.

## Tencent · MCQ and Planning

| Model | RoboBench-MCQ | RoboBench-Planning |
| --- | ---: | ---: |
| HY-Embodied-0.5 MoT-2B | 49.2 | 54.2 |
| HY-Embodied-0.5 A30B | 62.8 | 59.3 |
| Hy-Embodied-VLM-1.0 A3B | 61.2 | 54.9 |
| Embodied-R1.5 8B | 41.1 | 39.4 |
| Cosmos3-Nano 8B | 43.5 | 41.5 |

The [HY-Embodied VLM-1.0 report, Tables 2–3](https://arxiv.org/html/2607.12894v1)
reports these scores, including its comparison runs of Embodied-R1.5 and
Cosmos3-Nano. Use the [HY evaluation guide](guides/HY_EMBODIED_EVAL.md) for the
available RoboBench helper and the status of protocol alignment.

## ZTE · Task planning

[EmbodiedMind, Tables I and VI](https://arxiv.org/html/2609.19659v1) reports
43.42 for EmbodiedMind-8B and 53.57 for EmbodiedMind-27B on RoboBench.
The 8B evaluation includes comparison runs for MiMo-Embodied-7B (45.06), Cosmos3-Nano (40.25),
RoboBrain2.5-8B-NV (39.00), and other models.
