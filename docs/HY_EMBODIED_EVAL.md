# Evaluating HY-Embodied on RoboBench

This page provides a RoboBench-side helper for users who want to evaluate a
HY-Embodied model on the two RoboBench settings mentioned by the HY-Embodied
release:

- `RoboBench-MCQ`
- `RoboBench-Planning`

Here, **MCQ** means **Multiple-Choice Question**.

We have asked the HY-Embodied maintainers to confirm the exact protocol they
used for their reported numbers. Until that is clarified, the helper uses the
public RoboBench mapping below.
The protocol clarification is tracked at:
https://github.com/Tencent-Hunyuan/HY-Embodied/issues/14#issuecomment-5070118229

Important: the helper does not load HY-Embodied weights directly. RoboBench
calls models through an OpenAI-compatible chat-completions API. Start the HY
model with an OpenAI-compatible server first, then point RoboBench to that
server. The official `Hy-Embodied-VLM-1.0` repository provides a vLLM serving
path; `HY-Embodied-0.5 MoT-2B` currently provides a Transformers inference
example, so it needs an OpenAI-compatible wrapper before it can be called by
this helper.

## Default Mapping

| Setting | RoboBench dimensions | Coverage |
| --- | --- | --- |
| `RoboBench-MCQ` | `perception_reasoning affordance_reasoning error_analysis` | Dimensions 2, 4, and 5: 13 subtasks, 1,895 questions |
| `RoboBench-Planning` | `instruction_comprehension generalized_planning` | Dimensions 1 and 3: 19 subtasks, 4,197 questions |

Planning evaluation calls the evaluator model configured under
`evaluation.planning`. The shared `evaluation.api` configuration is independent
from the HY model endpoint, so a local HY vLLM server can be used together with
an external judge such as GPT-4o.

## Serve HY-Embodied

For `Hy-Embodied-VLM-1.0`, follow the official HY-Embodied vLLM instructions:

```bash
git clone https://github.com/Tencent-Hunyuan/HY-Embodied
cd HY-Embodied

uv pip install vllm==0.14.1 --torch-backend auto
uv pip install -e Hy-Embodied-VLM-1.0/inference/vllm/

MODEL_PATH=tencent/Hy-Embodied-VLM-1.0 \
SERVED_NAME=hy_a3b \
PORT=8080 \
  bash Hy-Embodied-VLM-1.0/inference/vllm/serve.sh
```

Check that the OpenAI-compatible endpoint is alive:

```bash
curl -sf http://127.0.0.1:8080/v1/models
```

The official vLLM server exposes the model name configured by `SERVED_NAME`
through `/v1/chat/completions`; use that exact served name as `ROBOBENCH_MODEL`
and in `config/benchmark.yaml`.

For `HY-Embodied-0.5 MoT-2B`, the official HY-Embodied repository currently
documents Transformers inference with `Hy-Embodied-0.5/inference.py`. To use it
with this RoboBench helper, run that model behind a local service that accepts
OpenAI-style multimodal chat-completion requests and returns compatible
responses.

## Setup

Install RoboBench and download the dataset:

```bash
git clone https://github.com/yulin-luo/RoboBench.git
cd RoboBench
pip install -e .

hf download LeoFan01/RoboBench \
  --repo-type dataset \
  --local-dir data/RoboBench-hf
```

Create a local config:

```bash
cp config/benchmark.example.yaml config/benchmark.yaml
```

Edit `config/benchmark.yaml` and add the served HY-Embodied model name under
`models`. For the official `Hy-Embodied-VLM-1.0` vLLM server, the default
served model name is `hy_a3b`:

```yaml
models:
  - name: "hy_a3b"
    provider: "openai"
    vision: true
```

If you serve the model under a different name, use that name instead.

To match HY-Embodied's thinking-mode evaluation for `Hy-Embodied-VLM-1.0`, add
the vLLM chat-template extra body and an explicit generation length:

```yaml
api:
  base_url: "${ROBOBENCH_API_BASE_URL}"
  api_key: "${DUBRIFY_API_KEY}"
  max_tokens: 4096
  extra_body:
    chat_template_kwargs:
      enable_thinking: true
```

Set `max_tokens` higher if your endpoint truncates long planning answers. If
you intentionally want direct-answer mode for latency tests, change
`enable_thinking` to `false`; HY-Embodied reports its variants in thinking mode.

Point RoboBench to your OpenAI-compatible endpoint and local paths:

```bash
export ROBOBENCH_API_BASE_URL="http://127.0.0.1:8080/v1"
export DUBRIFY_API_KEY="EMPTY"
export ROBOBENCH_DATA_ROOT="$PWD/data/RoboBench-hf"
export ROBOBENCH_RESULTS_ROOT="$PWD/results"
export ROBOBENCH_CACHE_DIR="$PWD/cache"
export ROBOBENCH_JUDGE_API_BASE_URL="https://your-judge-endpoint/v1"
export ROBOBENCH_JUDGE_API_KEY="your-judge-api-key"
```

`DUBRIFY_API_KEY=EMPTY` is sufficient for a local endpoint that does not require
authentication. The judge variables are required by the strict config even for
MCQ-only commands. For Planning, they must point to an endpoint serving the
model named by `evaluation.planning.eval_model`.

RoboBench reads the downloaded `questions.json` files and
`system_prompt.json` directly. There is no additional prompt-data directory to
download or generate.

## One-Command Runs

Validate the 32 released subtask manifests first:

```bash
ROBOBENCH_MODEL=hy_a3b \
  bash scripts/run_hy_embodied_eval.sh dry-run
```

`dry-run` validates 6,092 questions and 37,126 image references without making
API calls or checking whether all image files have been downloaded. It can be
used before adding the model entry to `config/benchmark.yaml`. For a strict
image-file check, run:

```bash
robobench --config config/benchmark.yaml inspect-data
```

Real `smoke`, `mcq`, `planning`, and `all` runs require the served model name to
be present in `config/benchmark.yaml`.

Then run one MCQ sample and one Planning Q1 sample:

```bash
ROBOBENCH_MODEL=hy_a3b \
  bash scripts/run_hy_embodied_eval.sh smoke
```

Run the MCQ setting:

```bash
ROBOBENCH_MODEL=hy_a3b \
  bash scripts/run_hy_embodied_eval.sh mcq
```

Run the Planning setting:

```bash
ROBOBENCH_MODEL=hy_a3b \
  bash scripts/run_hy_embodied_eval.sh planning
```

Run both settings:

```bash
ROBOBENCH_MODEL=hy_a3b \
  bash scripts/run_hy_embodied_eval.sh all
```

For quick debugging:

```bash
ROBOBENCH_MODEL=hy_a3b \
ROBOBENCH_MAX_SAMPLES=5 \
  bash scripts/run_hy_embodied_eval.sh all
```

## Outputs

Raw responses and evaluated outputs are written under `paths.results_root` from
`config/benchmark.yaml`, typically:

```text
results/<run-id>/<model>/<dimension>_<subtask>/raw.json
results/<run-id>/<model>/<dimension>_<subtask>/evaluated.json
```

Official RoboBench reference score tables and model-output JSON files are
available at:

```text
https://huggingface.co/datasets/lyl010221-pku/RoboBench-Results
```
