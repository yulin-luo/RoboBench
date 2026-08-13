

<h1 align="center">🤖 RoboBench</h1>

<p align="center">
  <strong>A comprehensive evaluation benchmark for Multimodal Large Language Models as embodied robot brains.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.17801"><img src="https://img.shields.io/badge/arXiv-2510.17801-b31b1b.svg?style=flat&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://robo-bench.github.io/"><img src="https://img.shields.io/badge/Project-Website-brightgreen.svg?style=flat&logo=githubpages&logoColor=white" alt="Project website"></a>
  <a href="https://huggingface.co/datasets/LeoFan01/RoboBench"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg?style=flat&logo=huggingface&logoColor=black" alt="Dataset"></a>
  <a href="https://huggingface.co/datasets/lyl010221-pku/RoboBench-Results"><img src="https://img.shields.io/badge/Results-HuggingFace-orange.svg?style=flat&logo=huggingface&logoColor=black" alt="Official results"></a>
  <a href="https://github.com/yulin-luo/RoboBench/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/ECCV-2026-purple.svg?style=flat" alt="ECCV 2026">
</p>

---

## 📌 Featured Resources

- 📄 **Paper**: [RoboBench: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain](https://arxiv.org/abs/2510.17801)
- 🌐 **Project website**: [robo-bench.github.io](https://robo-bench.github.io/)
- 🤗 **Dataset**: [LeoFan01/RoboBench](https://huggingface.co/datasets/LeoFan01/RoboBench)
- 📊 **Official results**: [lyl010221-pku/RoboBench-Results](https://huggingface.co/datasets/lyl010221-pku/RoboBench-Results)
- 💬 **Prompts and pipeline**: [docs/PROMPTS_AND_PIPELINE.md](docs/PROMPTS_AND_PIPELINE.md)
- 🧪 **Multiple-Choice Question (MCQ) and Planning evaluation**: [docs/HY_EMBODIED_EVAL.md](docs/HY_EMBODIED_EVAL.md)
- 💻 **Code**: [github.com/yulin-luo/RoboBench](https://github.com/yulin-luo/RoboBench)

> Accepted to **ECCV 2026**.

## 📰 News

- 📊 **2026.07** - Congratulations to [HY-Embodied](https://github.com/Tencent-Hunyuan/HY-Embodied)! **HY-Embodied-0.5 MoT-2B** and **Hy-Embodied-VLM-1.0 A3B** include **RoboBench-MCQ** and **RoboBench-Planning** as part of their official evaluation suite, highlighting RoboBench as a recognized benchmark for embodied foundation models.
- 🎉 **2026.07** - RoboBench is accepted to **ECCV 2026**. The official release reports results for **18 state-of-the-art MLLMs** with the MLLM-as-world-simulator planning framework.
- 📰 **2026.07** - RoboBench is featured by [具身智能之心](https://mp.weixin.qq.com/s/SdGEqu_1mz14DhUumTZ3FQ), introducing our benchmark for evaluating MLLMs as embodied brains.

## 🔍 Overview

RoboBench evaluates MLLMs on robotic manipulation tasks by decomposing embodied intelligence into diagnostic abilities rather than reporting only end-to-end task success. It covers the full execution pipeline from instruction comprehension and perception to generalized planning, affordance reasoning, and failure analysis. The ECCV 2026 release contains **5 dimensions, 14 capability groups, 32 subtasks, 6,092 QA pairs, and results for 18 state-of-the-art MLLMs**.

<p align="center">
  <img src="assets/teaser.jpg" alt="RoboBench overview" width="900">
</p>

### ✨ What RoboBench Provides

- **Fine-grained embodied evaluation**: 5 cognitive dimensions, 14 capability groups, 32 subtasks, and 6,092 QA pairs for diagnosing where MLLMs succeed or fail as robot brains.
- **18-model leaderboard**: closed-source, open-source, and embodied MLLMs are evaluated in the official results release, plus a GPT-5.4 text-only ablation for visual-grounding analysis.
- **Cross-domain planning tests**: Robot morphology, object type, viewpoint, attributes, and world-knowledge generalization.
- **MLLM-as-world-simulator evaluation**: Planning outputs are judged with a simulator-style MLLM evaluator for physically grounded task completion.
- **Reproducible package**: YAML-driven configuration, API inference, task-specific evaluators, checkpoint/resume, and multi-run aggregation.
- **Reusable prompt pipeline**: Open prompt construction utilities for robotic video and image-based question answering, with prompt coverage documented in [docs/PROMPTS_AND_PIPELINE.md](docs/PROMPTS_AND_PIPELINE.md).

## 🧭 Benchmark Dimensions

| Dimension | Representative capabilities | Evaluation type |
| --- | --- | --- |
| **Instruction Comprehension** | Explicit goals, implicit demands, cross-task navigation | Planning |
| **Perception and Reasoning** | Object attributes, spatial relations, temporal causality, robot type/view | Multiple-choice |
| **Generalized Planning** | Cross-embodiment, cross-object, cross-view, cross-attribute, world knowledge | Planning Q1/Q2/Q3 |
| **Affordance Reasoning** | Static affordance, dynamic affordance, navigation visual prompts | Multiple-choice |
| **Error Analysis** | High-level planning errors, low-level execution errors | Multiple-choice |

## 🧪 Multiple-Choice Question (MCQ) and Planning

For comparison with embodied foundation models such as
[HY-Embodied](https://github.com/Tencent-Hunyuan/HY-Embodied), RoboBench provides
two ready-to-run evaluation settings. Here, **MCQ** means
**Multiple-Choice Question**:

| Setting | Included dimensions | Evaluation |
| --- | --- | --- |
| **RoboBench-MCQ** | **2, 4, and 5**: 13 subtasks, 1,895 questions | Multiple-choice answer normalization and scoring |
| **RoboBench-Planning** | **1 and 3**: 19 subtasks, 4,197 questions | Q1 multi-step planning, Q2 next-action prediction, and Q3 state estimation; planning responses are judged by the evaluator model configured under `evaluation.planning` |

After completing the dataset, config, and model-server setup described below,
run either setting with one command:

```bash
# Validate the released metadata without running inference.
ROBOBENCH_CONFIG=config/benchmark.example.yaml \
  bash scripts/run_hy_embodied_eval.sh dry-run

# Run one setting, or replace mcq with planning/all.
ROBOBENCH_MODEL=hy_a3b \
  bash scripts/run_hy_embodied_eval.sh mcq
```

The helper calls the model through an OpenAI-compatible
`/v1/chat/completions` endpoint; it does not load model weights directly. See
[Evaluating HY-Embodied on RoboBench](docs/HY_EMBODIED_EVAL.md) for the vLLM
serving example, configuration, smoke test, and output locations.

The mapping above is the current RoboBench-side interpretation of the two
reported HY-Embodied settings. We are tracking confirmation of the exact
reported protocol in
[Tencent-Hunyuan/HY-Embodied#14](https://github.com/Tencent-Hunyuan/HY-Embodied/issues/14#issuecomment-5070118229)
and will synchronize the helper if their protocol differs.

## ⚙️ Installation

```bash
git clone https://github.com/yulin-luo/RoboBench.git
cd RoboBench

# Core package for API-based inference and evaluation.
pip install -e .

# Optional dependencies for local HuggingFace vision-language models.
pip install -e ".[local]"
```

### 📦 Requirements

- Python >= 3.10
- API-compatible endpoint supported by the OpenAI Python client
- OpenCV, PyYAML, Pydantic, NumPy, tqdm
- Optional local-model stack: torch, transformers, pillow

## 📚 Dataset

Download the released RoboBench dataset from Hugging Face and point the config to the local copy.

```bash
hf download LeoFan01/RoboBench \
  --repo-type dataset \
  --local-dir data/RoboBench-hf
```

The downloaded directory is the complete runtime data source. RoboBench reads
the 32 released `questions.json` files and `system_prompt.json`, constructs the
prompts deterministically, and resolves every image under `paths.data_root`.
No separately generated prompt files or path-prefix rewrite configuration is
required.

Validate the released question metadata before inference:

```bash
robobench --config config/benchmark.yaml inspect-data --metadata-only
```

After all images are downloaded, run the stricter file check:

```bash
robobench --config config/benchmark.yaml inspect-data
```

The complete release contains 32 subtasks, 6,092 questions, and 37,126 image
references. `RoboBench-MCQ` covers 13 subtasks and 1,895 questions;
`RoboBench-Planning` covers 19 subtasks and 4,197 questions.

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

## 🚀 Quick Start

### 1. 🛠️ Prepare a Config

```bash
cp config/benchmark.example.yaml config/benchmark.yaml

export DUBRIFY_API_KEY="your-api-key"
export ROBOBENCH_API_BASE_URL="https://your-endpoint/v1"
export ROBOBENCH_DATA_ROOT="$PWD/data/RoboBench-hf"
export ROBOBENCH_RESULTS_ROOT="$PWD/results"
export ROBOBENCH_CACHE_DIR="$PWD/cache"
export ROBOBENCH_JUDGE_API_BASE_URL="https://your-judge-endpoint/v1"
export ROBOBENCH_JUDGE_API_KEY="your-judge-api-key"
```

Edit `config/benchmark.yaml` to choose models, dimensions, and concurrency settings. Keep `config/benchmark.yaml` local; it is intentionally ignored by git.

### 2. 🧠 Run Inference

After `pip install -e .`, the package exposes a `robobench` command-line entrypoint. The CLI uses a top-level `--config` argument before the subcommand.

```bash
robobench --config config/benchmark.yaml inference \
  --model gpt-5.4 \
  --dimension perception_reasoning \
  --subtask static_attribute \
  --max-samples 1 \
  --run-id smoke_test
```

If you prefer not to install the package, use the equivalent Python module form:

```bash
PYTHONPATH=src python -m robobench.cli --config config/benchmark.yaml inference \
  --model gpt-5.4 \
  --dimension perception_reasoning \
  --subtask static_attribute \
  --max-samples 1 \
  --run-id smoke_test
```

For text-only ablation:

```bash
robobench --config config/benchmark.yaml inference \
  --model gpt-5.4 \
  --dimension perception_reasoning \
  --subtask static_attribute \
  --max-samples 1 \
  --run-id run_0_text_only \
  --text-only
```

HY-Embodied users who want to run the reported `RoboBench-MCQ` and
`RoboBench-Planning` style settings can use the helper script in
[docs/HY_EMBODIED_EVAL.md](docs/HY_EMBODIED_EVAL.md). The helper expects a
HY-Embodied model served through an OpenAI-compatible endpoint, such as the
official `Hy-Embodied-VLM-1.0` vLLM server; it does not load HY weights directly.

### 3. 📊 Evaluate Existing Results

```bash
robobench --config config/benchmark.yaml evaluate \
  --dimension perception_reasoning
```

Remove `--max-samples` when running the full selected dimension.

### 4. 🔁 Run an End-to-End Pipeline

```bash
robobench --config config/benchmark.yaml pipeline --repeats 3
```

The full pipeline runs inference, evaluation, and aggregation across repeated runs configured by `runs.num_repeats`.

## 🧩 Configuration

Most behavior is controlled from `config/benchmark.yaml`.

### 🔐 API Settings

| Field | Description |
| --- | --- |
| `api.base_url` | OpenAI-compatible endpoint for the model being evaluated |
| `api.api_key` | API key for the model endpoint, usually supplied as `${DUBRIFY_API_KEY}` |
| `api.api_max_concurrent` | Request-level API concurrency |
| `api.task_timeout` | Per-request timeout in seconds |
| `api.retry_attempts` | Maximum retry attempts for transient failures |
| `evaluation.planning.eval_model` | Model used to judge Planning Q1/Q2/Q3 responses |
| `evaluation.api` | Independent OpenAI-compatible API configuration for Planning judging and optional MCQ answer normalization |

### 🧠 Model Selection

```yaml
models:
  - name: "gpt-5.4"
    provider: "openai"
    vision: true

text_only_variants:
  - name: "gpt-5.4"
    suffix: "text_only"
```

### 🧪 Dimension Selection

```yaml
dimensions:
  perception_reasoning:
    enabled: true
    eval_type: "multi_choice"
    subtasks:
      - static_attribute
      - spatial_relation
```

### 📁 Paths

| Field | Description |
| --- | --- |
| `paths.data_root` | Local RoboBench dataset directory containing `system_prompt.json`, released `questions.json` files, and images |
| `paths.results_root` | Model outputs and evaluated scores |
| `paths.cache_dir` | Checkpoints and temporary files |

## 🗂️ Project Structure

```text
RoboBench/
├── config/
│   └── benchmark.example.yaml
├── src/robobench/
│   ├── analysis/          # Dataset and correlation analysis utilities
│   ├── data/              # Dataset loading from released questions.json files
│   ├── evaluation/        # Multiple-choice, planning, point, IoU, trajectory evaluators
│   ├── generation/        # Generation-stage nodes
│   ├── inference/         # Async API client, checkpoints, image handling, local HF client
│   ├── pipeline/          # Dataflow nodes and executor
│   ├── prompts/           # Prompt builders and robot-task templates
│   ├── scoring/           # Multi-run aggregation and statistics
│   └── utils/             # Path utilities
├── pyproject.toml
└── README.md
```

## 📏 Evaluation Metrics

| Evaluator | What it checks |
| --- | --- |
| `multi_choice` | Multiple-choice answer normalization and exact scoring |
| `planning` | Q1 multi-step plans, Q2 single-step actions, Q3 state estimation |
| `point` | Distance between predicted and ground-truth coordinates |
| `iou` | Bounding-box intersection over union |
| `trajectory` | Multi-point trajectory comparison |

Planning evaluation can call an evaluator model, configured by `evaluation.planning.eval_model`, to judge action feasibility and task completion.

## 💬 Prompt Builder Example

```python
from robobench.data.dataset import RoboBenchDataset
from robobench.prompts.builder import PromptBuilder

dataset = RoboBenchDataset("data/RoboBench-hf")
questions = dataset.load_questions(
    "perception_reasoning",
    "static_attribute",
    max_samples=1,
)
builder = PromptBuilder()
prompts = builder.build(questions, mode="base64")
builder.save(prompts, "prompts.jsonl")
```

## 🛠️ Development

```bash
pip install -e ".[dev]"
python -m compileall -q src
black src/
ruff check src/
```

## 📝 Citation

If you use RoboBench in your research, please cite:

```bibtex
@misc{luo2026robobenchcomprehensiveevaluationbenchmark,
      title={Robobench: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain},
      author={Yulin Luo and Chun-Kai Fan and Menghang Dong and Jiayu Shi and Xiangju Mi and Mengdi Zhao and Bo-Wen Zhang and Cheng Chi and Jiaming Liu and Gaole Dai and Rongyu Zhang and Ruichuan An and Kun Wu and Zhengping Che and Shaoxuan Xie and Guocai Yao and Zhongxia Zhao and Pengwei Wang and Guang Liu and Zhongyuan Wang and Tiejun Huang and Shanghang Zhang},
      year={2026},
      eprint={2510.17801},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.17801},
}
```

## 📜 License

This repository is released under the [MIT License](LICENSE).
