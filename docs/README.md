# RoboBench documentation

## Guides

- [Run RoboBench](guides/QUICKSTART.md) covers installation, dataset validation, inference, and scoring.
- [Evaluate HY-Embodied](guides/HY_EMBODIED_EVAL.md) covers the RoboBench-MCQ and RoboBench-Planning helper.
- [External evaluations](EXTERNAL_EVALUATIONS.md) links reported results and their evaluation protocols.

## Reference

- [Benchmark and metrics](reference/BENCHMARK.md) describes the dimensions, dataset coverage, official results, and scoring.
- [Configuration](reference/CONFIGURATION.md) lists model, API, judge, and path settings.
- [Prompts and pipeline](reference/PROMPTS_AND_PIPELINE.md) maps prompts and evaluators to the source and includes a prompt-building example.

## Source layout

| Directory | Contents |
| --- | --- |
| [data](../src/robobench/data/) | Released question and image loading |
| [prompts](../src/robobench/prompts/) | Prompt builder and templates |
| [inference](../src/robobench/inference/) | API and local model backends |
| [evaluation](../src/robobench/evaluation/) | Task evaluators and planning judge |
| [pipeline](../src/robobench/pipeline/) and [generation](../src/robobench/generation/) | Execution graph and generation stage |
| [scoring](../src/robobench/scoring/) and [analysis](../src/robobench/analysis/) | Aggregation and benchmark analysis |
| [utils](../src/robobench/utils/) | Path utilities |

## Development

Run from the repository root.

```bash
pip install -e ".[dev]"
python -m pytest
python -m compileall -q src
black src/
ruff check src/
```
