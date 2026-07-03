# RoboBench

**RoboBench** is a comprehensive benchmark for evaluating Multimodal Large Language Models (MLLMs) as embodied intelligence on robotic manipulation tasks. It covers the full execution pipeline from instruction comprehension and perception to planning, affordance reasoning, and failure analysis.

> Accepted to **ECCV 2026**. Project page: https://robobench.github.io

## Overview

Existing robotic benchmarks primarily measure end-to-end task success rates. RoboBench takes a different approach: it decomposes embodied intelligence into **5 cognitive dimensions** and **14+ fine-grained capabilities**, providing a systematic diagnostic of where MLLMs excel and where they fall short as robotic "brains".

### 5 Dimensions, 14+ Capabilities

| Dimension | Capabilities | Eval Type |
|-----------|-------------|-----------|
| **Instruction Comprehension** | Explicit goal, Implicit demand, Cross-task navigation | Planning |
| **Perception & Reasoning** | Object attributes, Spatial relation, Temporal causality, Robot type/view | Multi-choice |
| **Generalized Planning** | Cross-embodiment (single/dual arm, mobile, human), Cross-object (rigid, articulated, deformable), Cross-view (single/multi), Cross-attribute (color, shape, size, number), World knowledge | Planning (Q1/Q2/Q3) |
| **Affordance Reasoning** | Static affordance, Dynamic affordance, Navigation visual prompt | Point / Coordinate |
| **Error Analysis** | High-level planning error, Low-level execution error | Multi-choice |

### Key Features

- **Decomposed evaluation**: Isolates cognitive capabilities instead of only measuring task success
- **Cross-domain generalization**: Tests planning across robot morphologies, object types, and camera viewpoints
- **MLLM-as-World-Simulator**: For planning evaluation, uses an MLLM to simulate whether generated plans would physically succeed
- **Reproducible pipeline**: Single YAML configuration drives the entire benchmark
- **3-run averaging**: Built-in support for running each model 3 times with statistical aggregation
- **Open-source prompts**: Modular prompt construction library that can be reused for other robotic video analysis tasks

## Installation

```bash
# Clone the repository
git clone https://github.com/yulin-luo/RoboBench.git
cd RoboBench

# Install the package (core, API-based evaluation)
pip install -e .

# Optional: local HuggingFace model inference (torch / transformers)
pip install -e ".[local]"
```

### Dependencies

- Python >= 3.10
- OpenAI Python client
- OpenCV (for image processing)
- PyYAML, Pydantic (for configuration)
- NumPy (for statistics)
- `torch` / `transformers` / `pillow` — only for local HuggingFace models (`[local]` extra)

## Data

RoboBench data is hosted on the Hugging Face Hub. Download it and point `paths.data_root` in
your config at the local copy:

```bash
huggingface-cli download --repo-type dataset LeoFan01/RoboBench --local-dir data/RoboBench-hf
```

## Quick Start

### 1. Configure API Access

Set your API key as an environment variable (substituted automatically in config):

```bash
export DUBRIFY_API_KEY="your-api-key"
```

Start from the example config and keep secrets in environment variables:

```bash
cp config/benchmark.example.yaml config/benchmark.yaml
export ROBOBENCH_API_BASE_URL="https://your-endpoint/v1"
export ROBOBENCH_DATA_ROOT="/path/to/RoboBench/data"
export ROBOBENCH_MIDDLE_FILE_DIR="/path/to/RoboBench/data/middle_file"
export ROBOBENCH_RESULTS_ROOT="./results"
export ROBOBENCH_CACHE_DIR="./cache"
```

### 2. Configure Models and Dimensions

Edit `config/benchmark.yaml` to select which models and dimensions to evaluate:

```yaml
models:
  - name: "gpt-5.4"
    provider: "openai"
    vision: true
  - name: "claude-opus-4-7"
    provider: "anthropic"
    vision: true

dimensions:
  perception_reasoning:
    enabled: true
    eval_type: "multi_choice"
    subtasks: [static_attribute, spatial_relation]
```

### 3. Run the Benchmark

```bash
# Run stages separately
robobench inference --config config/benchmark.yaml --model gpt-5.4
robobench evaluate --config config/benchmark.yaml --dimension perception_reasoning

# Full pipeline: inference + evaluation + 3-run aggregation
robobench pipeline --config config/benchmark.yaml --repeats 3
```

## Configuration Reference

All benchmark parameters are in `config/benchmark.yaml`:

### API Settings

| Field | Description | Default |
|-------|-------------|---------|
| `api.base_url` | API endpoint URL | — |
| `api.api_key` | API key (supports `${ENV_VAR}`) | — |
| `api.max_concurrent` | Task-level concurrency | 10 |
| `api.api_max_concurrent` | Request-level concurrency per task | 10 |
| `api.task_timeout` | Per-request timeout (seconds) | 960 |
| `api.retry_attempts` | Max retry attempts | 10 |

### Models

```yaml
models:
  - name: "gpt-5.4"          # Model identifier (passed to API)
    provider: "openai"       # Provider hint
    vision: true             # Whether model supports vision input

text_only_variants:          # Optional: text-only ablation
  - name: "gpt-5.4"
    suffix: "text_only"
```

### Run Settings

| Field | Description | Default |
|-------|-------------|---------|
| `runs.num_repeats` | Number of repeated runs for averaging | 3 |
| `runs.seed_strategy` | Seed strategy (`incremental`/`fixed`/`random`) | `incremental` |
| `runs.skip_existing` | Skip if result file already exists | `true` |

### Dimensions

Each dimension defines:

```yaml
dimensions:
  generalized_planning:
    enabled: true                        # Whether to run this dimension
    eval_type: "planning"                # Evaluator: multi_choice / planning / point / iou / trajectory
    system_prompt_key: "skill_list"      # System prompt template key
    subtasks:                            # List of subtask names
      - single_arm
      - dual_arm
      - mobile_manipulation
```

### Paths

| Field | Description |
|-------|-------------|
| `paths.data_root` | Root directory for benchmark data (images, questions) |
| `paths.results_root` | Directory for storing model outputs and scores |
| `paths.cache_dir` | Directory for temporary/checkpoint files |

## Architecture

RoboBench is organized as a **dataflow pipeline** where each stage is a composable node:

```
LoadDatasetNode -> BuildPromptsNode -> RunInferenceNode -> EvaluateNode -> AggregateScoresNode -> GenerateReportNode
```

### Package Structure

```
robobench/
├── pipeline/          # Pipeline engine (nodes, executor, run context)
├── inference/         # Async API client with retry, checkpointing, resume
│   ├── client.py      # AsyncModelClient: configurable concurrency & timeout
│   ├── checkpoint.py  # CheckpointManager: resume from crashes
│   └── image.py       # Image encoding (base64, resize) for vision input
├── prompts/           # Prompt construction library (open-source ready)
│   ├── builder.py     # PromptBuilder: format messages with images
│   └── templates/     # Prompt templates per task type
├── evaluation/        # Evaluators per task type
│   ├── multi_choice.py   # Multiple choice / yes-no / open-ended
│   ├── planning.py       # Q1 (multi-step DAG) / Q2 (single-step) / Q3 (yes/no)
│   ├── point.py          # Coordinate evaluation
│   ├── iou.py            # Bounding box IoU
│   └── trajectory.py     # Trajectory evaluation
├── scoring/           # Result aggregation & statistics
│   └── aggregator.py  # 3-run averaging (mean, std, min, max)
├── data/              # Dataset loading
│   └── dataset.py     # RoboBenchDataset: load questions from JSONL
└── config.py          # YAML configuration with Pydantic validation
```

## Evaluation Metrics

### Multi-Choice / Yes-No

- **Multiple choice**: Regex extraction of A-D letters, sorted comparison with ground truth. Ambiguous responses normalized with GPT.
- **Yes/No**: Direct string matching (supports yes/no/equal/left/right/unclear/transparent/etc.)
- **Open-ended**: GPT evaluation (0-1 score) with word overlap fallback

### Planning (Q1/Q2/Q3)

- **Q1 - Multi-step Planning**: Extract structured plan, build DAG, evaluate node correctness + task completion degree
- **Q2 - Single-step Planning**: Extract step, evaluate skill usage + object reasonableness + parameter accuracy
- **Q3 - State Estimation**: Binary yes/no matching

### Point / Trajectory

- **Point**: Euclidean distance between predicted and ground-truth coordinates
- **IoU**: Intersection over Union for bounding boxes
- **Trajectory**: Multi-point trajectory comparison

## CLI Usage

```bash
# Inference only
robobench inference --config config/benchmark.yaml --model gpt-5.4

# Evaluation only (on existing results)
robobench evaluate --config config/benchmark.yaml --dimension perception_reasoning

# Full pipeline with 3 repeats
robobench pipeline --config config/benchmark.yaml --repeats 3

# Dry run: show pipeline graph without executing
robobench pipeline --config config/benchmark.yaml --dry-run
```

## Prompt Library

The `robobench.prompts` module is designed as a reusable, open-source prompt construction library for robotic video analysis:

```python
from robobench.prompts import PromptBuilder

builder = PromptBuilder(
    data_root="/path/to/data",
    system_prompt_key="skill_list"
)

# Build prompts from raw questions
prompts = builder.build(questions, mode="base64")

# Each prompt is API-ready
for prompt in prompts:
    print(prompt["messages"])  # OpenAI-compatible message format
```

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Format code
black src/

# Run tests
pytest tests/
```

## Citation

If you use RoboBench in your research, please cite:

```bibtex
@inproceedings{robobench2026,
  title={RoboBench: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain},
  author={Luo, Yulin and Fan, Chun-Kai and Dong, Menghang and Shi, Jiayu and Mi, Xiangju and Zhao, Mengdi and Zhang, Bo-Wen and Chi, Cheng and Liu, Jiaming and Dai, Gaole and Zhang, Rongyu and An, Ruichuan and Wu, Kun and Che, Zhengping and Xie, Shaoxuan and Yao, Guocai and Zhao, Zhongxia and Wang, Pengwei and Liu, Guang and Wang, Zhongyuan and Huang, Tiejun and Zhang, Shanghang},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE).
