# Prompts and Annotation Pipeline

This page maps the prompt and pipeline material described in the RoboBench paper appendix to the public repository.

## What is included in this repository

The public code release includes the prompt utilities needed to run inference and evaluation on the released benchmark data.

| Purpose | Public file |
| --- | --- |
| API-ready prompt construction from released question JSONL files | `src/robobench/prompts/builder.py` |
| Single-arm video-to-task/step description template | `src/robobench/prompts/robot_types/single_arm.py` |
| Dual-arm video-to-task/step prompt set | `src/robobench/prompts/robot_types/dual_arm.py` |
| Mobile-manipulator video-to-task/step template | `src/robobench/prompts/robot_types/mobile_manipulation.py` |
| Natural-language step to predefined function conversion | `src/robobench/prompts/robot_types/language_to_function.py` |
| Explicit-to-implicit instruction conversion | `src/robobench/prompts/robot_types/explicit_to_implicit.py` |
| Shared planning templates and action vocabulary | `src/robobench/prompts/templates/planning.py` |
| Production Q1 world-simulator judge prompt and Q2/Q3 evaluator prompts | `src/robobench/evaluation/planning.py` |
| Multiple-choice / open-ended answer normalization prompts | `src/robobench/evaluation/multi_choice.py` |

For most users, `PromptBuilder` is the main entry point: it reads released question records, resolves image paths, injects optional system prompts, and writes API-ready message payloads.

## Paper appendix coverage

The supplementary material contains the full prompt appendix for benchmark construction, model inference, and evaluation. The table below shows how each appendix item relates to the public release.

| Appendix item | Public release status |
| --- | --- |
| `prompt:video2nl_single` single-arm video to task/step descriptions | Released as `single_arm.py` |
| `prompt:video2nl_dual` dual-arm video to task/step descriptions | Released as `dual_arm.py` |
| `prompt:video2nl_mobile` mobile manipulator video to task/step descriptions | Released as `mobile_manipulation.py` |
| `prompt:func_list_manipulation` manipulation function list | Released in `planning.py` and `language_to_function.py` |
| `prompt:func_list_navigation` navigation function list | Released in `language_to_function.py` |
| `prompt:nl2func_conversion` natural-language steps to predefined functions | Released as `promptForFunctionTemplate` in `language_to_function.py` |
| `prompt:func_instantiation` function instantiation with object references | Released as `promptForFunctionCall` in `language_to_function.py` |
| `prompt:explicit2implicit` explicit to implicit instruction conversion | Released as `explicit_to_implicit.py` |
| `prompt:step_attribute_objects` step-level object extraction | Described in the paper appendix; not exposed as a standalone runtime module |
| `prompt:step_attribute_actions` step-level action extraction | Described in the paper appendix; not exposed as a standalone runtime module |
| `prompt:scene_labels` scene-label extraction | Described in the paper appendix; not exposed as a standalone runtime module |
| `prompt:functional_qa_generation` functionality QA generation | Described in the paper appendix; not exposed as a standalone runtime module |
| `prompt:q1_action_extraction` Q1 structured action extraction | Released in `prompts/templates/planning.py`; production scoring path uses `evaluation/planning.py` |
| `prompt:q1_scoring` Q1 DAG-grounded world-simulator scoring | Released as production `PROMPT_V3` in `evaluation/planning.py` |
| `prompt:q2_action_extraction` Q2 action extraction | Released as `PROMPT_TEMPLATE_EXTRACT_STEP` in `evaluation/planning.py` |
| `prompt:q2_scoring` Q2 prompt-based scoring | Released as `PROMPT_TEMPLATE_COMPARE_STEPS` in `evaluation/planning.py` |
| `prompt:q3_yesno` Q3 yes/no conversion | Released as `PROMPT_TEMPLATE_EXTRACT_YES_NO` in `evaluation/planning.py` |

The data-construction prompts that are not standalone runtime modules are still documented in the paper appendix for transparency. They were used for benchmark construction and quality control rather than for ordinary users running inference/evaluation on the released dataset.

## Dataset construction pipeline

RoboBench uses a dimension-specific construction pipeline:

1. Collect and normalize robot videos, images, task descriptions, and metadata.
2. Use VLMs, detection/segmentation tools, or human experts to draft annotations.
3. Convert raw annotations into task-level and step-level structured records.
4. Map planning steps into predefined function vocabularies when needed.
5. Generate released QA records under the unified RoboBench schema.
6. Run task-specific quality-control checks before evaluation.

The released dataset already contains the final prompt-ready question records. Therefore, reproducing the benchmark evaluation does not require rerunning the full annotation pipeline.

## Evaluation pipeline

The public evaluation path is implemented in `src/robobench/evaluation/`:

- Multiple-choice, yes/no, and open-ended QA are handled by `multi_choice.py`.
- Planning Q1/Q2/Q3 are handled by `planning.py`.
- Point, bounding-box, and trajectory tasks are handled by `point.py`, `iou.py`, and `trajectory.py`.

For planning, Q1 uses `PlanningEvaluator` with the production `PROMPT_V3` MLLM-as-world-simulator judge prompt that scores node correctness and embodied task completion. Q2 extracts and compares the next action step. Q3 normalizes model outputs into yes/no decisions.

`src/robobench/prompts/templates/planning.py` contains shared planning templates and earlier extraction/evaluation helpers. For reproducing official planning scores, use the evaluator path in `src/robobench/evaluation/planning.py`.
