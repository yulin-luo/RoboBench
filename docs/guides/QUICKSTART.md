# Run RoboBench

Run the commands below from the repository root in a Bash shell with Python 3.10 or later.

## 1. Install RoboBench

```bash
git clone https://github.com/yulin-luo/RoboBench.git
cd RoboBench
pip install -e .
```

For the optional local Hugging Face model backend, install with `pip install -e ".[local]"`.

## 2. Download the dataset

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), then download the benchmark data.

```bash
hf download LeoFan01/RoboBench --repo-type dataset --local-dir data/RoboBench-hf
```

The download contains the released question manifests, system prompts, and images. RoboBench builds prompts from these files at runtime. See the [benchmark reference](../reference/BENCHMARK.md) for dataset coverage and official model outputs.

## 3. Configure endpoints and validate the dataset

```bash
cp config/benchmark.example.yaml config/benchmark.yaml

export DUBRIFY_API_KEY="your-api-key"
export ROBOBENCH_API_BASE_URL="https://your-model-endpoint/v1"
export ROBOBENCH_DATA_ROOT="$PWD/data/RoboBench-hf"
export ROBOBENCH_RESULTS_ROOT="$PWD/results"
export ROBOBENCH_CACHE_DIR="$PWD/cache"
export ROBOBENCH_JUDGE_API_BASE_URL="https://your-judge-endpoint/v1"
export ROBOBENCH_JUDGE_API_KEY="your-judge-api-key"
```

Set the endpoints and keys for your model and evaluator. In `config/benchmark.yaml`, use a model name or ID served by your endpoint and enable the dimensions you want. Set `evaluation.planning.eval_model` to the model ID served by the judge endpoint if you run planning. See the [configuration reference](../reference/CONFIGURATION.md) for available fields. The local config is gitignored.

```bash
robobench --config config/benchmark.yaml inspect-data --metadata-only
robobench --config config/benchmark.yaml inspect-data
```

The first command checks the question manifests. The second also checks the referenced images.

## 4. Run and score one example

Replace `gpt-5.4` in both commands if you configured a different model. Use the same model, dimension, subtask, and run ID to score the outputs from the inference command.

```bash
robobench --config config/benchmark.yaml inference \
  --model gpt-5.4 --dimension perception_reasoning --subtask static_attribute \
  --max-samples 1 --run-id smoke_test

robobench --config config/benchmark.yaml evaluate \
  --model gpt-5.4 --dimension perception_reasoning --subtask static_attribute \
  --run-id smoke_test
```

For the full subtask, omit `--max-samples` from inference and choose a new run ID for both commands. For a text-only ablation, add `--text-only` to inference and use a separate run ID. You can also invoke the CLI as `PYTHONPATH=src python -m robobench.cli` from a Python environment with its dependencies installed.

## 5. Run a benchmark configuration

The pipeline runs inference, evaluation, and score aggregation across repeated runs. Select the desired models and dimensions in the config first.

```bash
robobench --config config/benchmark.yaml pipeline --repeats 3
```

For RoboBench-MCQ and RoboBench-Planning, follow the [HY-Embodied evaluation guide](HY_EMBODIED_EVAL.md).
