"""Planning evaluator for RoboBench Q1/Q2/Q3.

Production evaluator — aligned with the final v3 prompt and paper.

Q1 (Multi-step Planning): MLLM-as-world-simulator with image + DAG + final v3 prompt
Q2 (Single-step Planning): Extract step → compare with GT on 3 dimensions
Q3 (State Estimation): Extract yes/no → binary match

This evaluator loads ground-truth data (gt_answer, image, DAG) from the
configured dataset root on the fly using the record id as the lookup key.
"""

import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseEvaluator, register_evaluator

# ---------------------------------------------------------------------------
# v3 Prompt (production)
# ---------------------------------------------------------------------------

PROMPT_V3 = """You are a world-simulator judge for robot task plans. Given a scene image, a ground-truth (GT) action sequence, a GT DAG of dependencies, and a model's predicted plan, output two integer scores in [0, 10].

## Inputs
- **Scene image** `I_0`: first frame of the task — use it to identify the objects, their initial spatial relations, and physical constraints.
- **GT action list** `A*`: e.g. `1-grasp(cup), 2-pick_up(cup), 3-place(cup, table)` (single-arm) or `1-left:move_to(none, towel), right:move_to(none, towel), 2-left:grasp(edge), right:grasp(edge), ...` (dual-arm).
- **GT DAG** `G`: dependency graph over GT actions; defines which actions enable which milestones.
- **Model plan**: predicted action list in the same syntax.

Each action is a node `(skill, object, parameter?)`.

## Score 1: Node Correctness

**Step 1 — count GT nodes** `N_GT` exactly:
- A line `i-action(args)` (single arm / human / mobile) contributes **1 node**.
- A line `i-left:actionL, right:actionR` (dual-arm) contributes **2 nodes**, one per arm. `no_ops` is a node and only matches another `no_ops`.
- Skip nothing. Do not collapse, merge, or drop nodes.

**Step 2 — greedy one-to-one match**: walk the model plan left-to-right. For each predicted node, find at most one yet-unmatched GT node where ALL three hold:
- **Skill**: identical token. `grasp ≠ pick_up`, `push ≠ pull`, `place ≠ insert`.
- **Object**: same physical referent. Allow alias only when the two names point to the same entity in the scene (`faucet ≈ tap`, `cup ≈ mug`, `dish_rack ≈ rack`). **Different colors / sides / IDs never match** (`red_button ≠ green_button`, `left_door ≠ right_door`, `object_3 ≠ object_8`, `purple_cylinder ≠ pink_cylinder_block`). For dual-arm, the predicted node's arm side must match the GT side.
- **Parameter**: directionally or functionally equivalent (`clockwise ≈ cw`, `open ≈ on` for binary state, `under_table ≈ under_of_table`). Specific named locations must match (`bottom_left_corner ≠ point_5`).

A node is fully correct or 0 — **no partial credit per node**.

**Step 3** — `NodeCorrectness = floor(matched / N_GT × 10)`.

## Score 2: Task Completion (world-simulation rollout)

**IMPORTANT — handle trivial / empty `S*` before scoring:**
- If after applying the exclusion list `S*` is **empty** (the goal contains only excluded robot motions — `move_to` / `grasp` / `pick_up` / `release` / `no_ops` / `observation` etc. — and no genuine object-level state change), output `Completion=10` and stop. Both an empty model plan and a non-empty model plan are scored 10 in this case.
- For **dual-arm tasks**: if one arm's full sequence consists of `no_ops` (or excluded motions only), that arm contributes nothing to `S*`. Evaluate `S*` from the OTHER arm only. Do not penalize the model for matching the `no_ops` arm with `no_ops`.
- For **single-step or one-action tasks** (e.g. human embodiment `1-grasp(X), 2-wipe(...)` where `S*={surface cleaned}`), make the `S*` membership decision based on whether the final scene state changes — not on whether intermediate `grasp` / `move_to` nodes are present.

Follow the 4-step procedure:

1. **Initial world** `W_0`: from `I_0`, list the objects mentioned in `A*`, their starting locations / states / containment.
2. **Critical states** `S*`: from `A*` and `G`, extract the set of **object-level state changes** required by the goal:
   - Object moved to a new container/surface (`apple: table → bowl`).
   - Container open/closed (`drawer: closed → open`).
   - Object activated/deactivated (`light: off → on`).
   - Object orientation/assembly meaningfully changed.
   - **Excluded** (these are robot motions, not critical states): `move_to, grasp, hold, approach, align, release, retract, observation, look, scan, plan, think`.
3. **Order** via `G`: enforce precedence (a state can only be marked achieved after its prerequisites).
4. **Rollout**: simulate the model plan step-by-step against `W_0`. For each predicted action: check preconditions are met, then update `W_t → W_{t+1}` with its effect, and mark any `s ∈ S*` it accomplishes. Aggregate achieved set `Ŝ ⊆ S*`.

Counting rules during rollout:
- **Implicit succession**: a later action can satisfy an earlier prerequisite implicitly. `pick_up` implies `grasp`+`move_to`; `place` implies `move_to`. If the goal state is reached, missing intermediate motion nodes do not block it.
- **Wrong-direction action**: physically infeasible step achieves nothing (`push(drawer, inward)` does not open).
- **Wrong object/target**: an action on the wrong referent yields no state change toward `S*` (model places `object_3` at `cloth_corner` when target is `point_5` → that critical state is not in `Ŝ`).
- **Surrogate skill**: a skill that physically achieves the same end state still counts (e.g. `pour` from a container instead of `scoop`+`pour` with a spoon, when the goal is "contents in bowl"). But a skill that cannot reach the goal does not (e.g. `push` along a flat table when the goal requires lifting onto a shelf).

`TaskCompletion = floor(|Ŝ| / |S*| × 10)`.

## Examples

**A. Single-arm, skip pick_up** — GT: `1-move_to(none,apple), 2-grasp(apple), 3-pick_up(apple), 4-place(apple, bowl)` (`N_GT=4`, `S*={apple in bowl}`, `|S*|=1`). Model: `[move_to, grasp, place(apple, bowl)]`. Match: 3/4 → Node=7. Critical state achieved → Completion=10.

**B. Dual-arm towel-fold** — GT: `1-left:move_to(none,L), right:move_to(none,R), 2-left:grasp(L), right:grasp(R), 3-left:pick_up(L), right:pick_up(R), 4-left:unfold(L,t), right:unfold(R,t)` (`N_GT=8`, `S*={towel unfolded}`, `|S*|=1`). Model omits both `pick_up`s, has 6 matched nodes → Node=floor(6/8·10)=7. Towel still ends unfolded → Completion=10.

**C. Drawer wrong direction** — GT: `1-grasp(drawer), 2-pull(drawer, outward)` (`N_GT=2`, `S*={drawer open}`). Model: `[grasp(drawer), push(drawer, inward)]`. Skills don't match on step 2 → Node=floor(1/2·10)=5. Drawer not opened → Completion=0.

**D. Wrong object ID** — GT: `1-grasp(3), 2-pick_up(3), 3-place(3, target)` (`N_GT=3`, `S*={object 3 at target}`). Model uses object `8` throughout. Object IDs differ → 0/3 nodes match, Node=0. Critical state not achieved → Completion=0.

**E. Functional surrogate** — GT: `1-grasp(spoon), 2-scoop(spoon, sauce, jar), 3-pour(spoon, sauce, bowl)` (`S*={sauce in bowl}`). Model: `[grasp(jar), pick_up(jar), pour(jar, sauce, bowl)]`. 0 nodes match (different objects/skills) → Node=0. But the goal state (sauce in bowl) is reached via `pour(jar, ...)` → Completion=10.

**F. Empty `S*` (dual-arm pre-task observation)** — GT: `1-left:move_to(none, book), right:move_to(none, book), 2-left:no_ops, right:no_ops`. After excluding `move_to` and `no_ops`, `S*` is empty. Both judges should output **Completion=10** regardless of what the model produces (no goal state to fail). Node score is computed normally.

**G. Dual-arm one-side no_ops** — GT: `1-left:no_ops, right:move_to(none, lid), 2-left:no_ops, right:grasp(lid), 3-left:no_ops, right:pull(lid, up)`, `S*={lid open}` (from right arm). Evaluate Completion only against the right-arm sub-plan. If model's right arm achieves `lid=open`, Completion=10. The left-arm `no_ops` lines are not penalized.

## Output (JSON only)

```json
{
  "node_correctness": {"result": <int 0-10>, "reason": "matched=X / N_GT=Y → floor(X/Y*10)=Z. <list any aliases used>"},
  "task_completion":  {"result": <int 0-10>, "reason": "achieved=A / |S*|=B → floor(A/B*10)=C. <name S* and which were achieved>"}
}
```

Both scores must be integers in [0, 10]. The `reason` must show the explicit counts. Output JSON only, no surrounding prose."""


# Prompt templates for Q2 / Q3 (kept from original)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_COMPARE_STEPS = """
You are evaluating the similarity between two robot action steps. Your task is to evaluate the extracted step against the ground truth step using specific criteria.

### Input data format
{{
    "extracted_step": "The step extracted from model response",
    "gt_step": "The ground truth step"
}}

### Evaluation criteria

1. **Skill usage accuracy (0, 1 point)**:
   - Consider only the skill/action part of both steps (ignore the objects/parameters)
   - Score based on whether the skills are completely identical.
   - 1: Perfect skill match (exactly the same action word)
   - 0: Different or not completely identical skills

2. **Operation object reasonableness (0, 0.5, or 1 point)**:
   - Consider only the operation object(s) part of both steps
   - Check if objects refer to similar or related things
   - 1: Objects are identical or clearly refer to the same thing (e.g., "door_handle" vs "door handle", "microwave_door" vs "microwave_handle")
   - 0.5: Objects are similar or related (e.g., "table" vs "table_leg", "cup" vs "mug", objects referring to the same category)
   - 0: Objects are completely different or unrelated

3. **Parameter accuracy (0, 0.5, or 1 point)**:
   - Consider the function parameters and their quality
   - If skill is 0 or object is 0, this parameter score is automatically 0
   - If skill and object both have scores, evaluate the parameter quality:
   - 1: Parameters are completely correct and high quality
   - 0.5: Parameters are partially correct or medium quality
   - 0: Parameters are incorrect or low quality

### Evaluation Guidelines
- Skill evaluation must be strict - only award 1 point when skills are exactly identical
- Object evaluation can be more flexible - consider semantic similarity
- Parameter evaluation depends on skill and object scores being non-zero
- Give detailed reasons explaining your scoring decision
- Consider the context and precision required for robot task execution

### Output format
Please output your evaluation results strictly according to the following JSON structure:
{{
    "skill_usage_accuracy": {{"result": x, "reason": "brief explanation of skill evaluation"}},
    "operation_object_reasonableness": {{"result": y, "reason": "brief explanation of object evaluation"}},
    "parameter_accuracy": {{"result": z, "reason": "brief explanation of parameter evaluation"}}
}}

Extracted step: {extracted_step}
Ground truth step: {gt_step}

Please output your results as required.
"""

PROMPT_TEMPLATE_EXTRACT_STEP = """
Extract the next step from this response. The step should be in the format: skill(element1, element2, ...)

For example:
- "grasp(microwave_handle)"
- "push(microwave_handle, close)"
- "move_to(none, drawer)"

Response: {response}

Return ONLY the step in the correct format, nothing else.
"""

PROMPT_TEMPLATE_EXTRACT_YES_NO = """
Extract ONLY the yes/no answer from this response.
Return ONLY "yes" or "no" with no other text or explanation.

Response: {response}
"""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@register_evaluator("planning")
class PlanningEvaluator(BaseEvaluator):
    """Production evaluator for planning tasks (Q1/Q2/Q3).

    Q1 uses the final v3 prompt with image + DAG inputs.
    Q2/Q3 use LLM-based extraction and comparison.
    """

    name = "planning"

    def __init__(self):
        self._dag_cache: Dict[str, Dict[str, Any]] = {}

    def evaluate(
        self, results: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate planning responses.

        Args:
            results: List of result dicts with 'response', 'gt_answer', 'id'
            config: Evaluator model, API, dataset root, and output configuration

        Returns:
            Dictionary with Q1/Q2/Q3 scores and overall statistics
        """
        if config is None:
            raise ValueError("Planning evaluation requires an explicit configuration")
        self.eval_model = config["eval_model"]
        self.data_root = Path(config["data_root"])
        self.api_config = config["api"]
        self._batch_save_prefix = config["batch_save_prefix"]

        # Step 1: Classify into Q1/Q2/Q3
        classified = self._classify_records(results)

        # Step 2: Evaluate each type
        q1_scores = asyncio.run(self._evaluate_q1(classified["q1"]))
        q2_scores = asyncio.run(self._evaluate_q2(classified["q2"]))
        q3_scores = asyncio.run(self._evaluate_q3(classified["q3"]))

        # Step 3: Calculate overall statistics
        overall = self._calculate_overall(q1_scores, q2_scores, q3_scores)

        return {
            "q1": q1_scores,
            "q2": q2_scores,
            "q3": q3_scores,
            "overall": overall,
        }

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_records(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Classify records into Q1, Q2, Q3 based on ID suffix or content."""
        classified = {"q1": [], "q2": [], "q3": []}

        for record in results:
            if not isinstance(record, dict):
                raise TypeError("Planning results must contain dictionaries")

            task_id = record["request_id"]
            gt_answer = record["gt_answer"]

            # Classify by ID suffix
            if "_Q1" in task_id:
                classified["q1"].append(record)
            elif "_Q2" in task_id:
                classified["q2"].append(record)
            elif "_Q3" in task_id:
                classified["q3"].append(record)
            else:
                # Auto-detect by GT answer pattern
                if PlanningEvaluator._is_yes_no(gt_answer):
                    classified["q3"].append(record)
                elif PlanningEvaluator._is_single_step(gt_answer):
                    classified["q2"].append(record)
                else:
                    classified["q1"].append(record)

        return classified

    def _batch_save_path(self, stage: str) -> str:
        """Return a stage-specific temp-result prefix for AsyncModelClient."""
        return f"{self._batch_save_prefix}_{stage}"

    @staticmethod
    def _is_yes_no(answer: str) -> bool:
        """Check if answer is yes/no type."""
        answer_lower = answer.lower().strip()
        return answer_lower in ("yes", "no", "equal", "left", "right")

    @staticmethod
    def _is_single_step(answer: str) -> bool:
        """Check if answer is a single step (not a sequence)."""
        return not re.search(r"^\d+[-.)]", answer.strip()) and "\n" not in answer

    # ------------------------------------------------------------------
    # Ground-truth loading (NAS dataset)
    # ------------------------------------------------------------------

    def _get_benchmark_path(self, record_id: str) -> str:
        """Infer benchmark path from record id.

        Example:
            '3_generalized_planning/cross_embodiment/dual_arm/images/..._Q1'
            -> '3_generalized_planning/cross_embodiment/dual_arm'
        """
        parts = record_id.split("/")
        if "images" in parts:
            img_idx = parts.index("images")
            return "/".join(parts[:img_idx])
        raise ValueError(f"Record id does not contain an images component: {record_id}")

    def _find_dag(self, record: Dict[str, Any]) -> str:
        """Load the released DAG for a Q1 record."""
        raw_question = record["raw_question"]
        if raw_question["task_type"][-1] == "navigation":
            return "Not applicable for navigation tasks."

        record_id = record["request_id"]
        benchmark = self._get_benchmark_path(record_id)
        if benchmark not in self._dag_cache:
            dag_path = self.data_root / benchmark / "dag.json"
            with open(dag_path, "r", encoding="utf-8") as f:
                dag_records = json.load(f)
            self._dag_cache[benchmark] = {item["id"]: item["gt_dag"] for item in dag_records}

        dag_id = record_id.removesuffix("_Q1")
        return json.dumps(self._dag_cache[benchmark][dag_id], indent=2)

    # ------------------------------------------------------------------
    # Q1 Evaluation (final v3)
    # ------------------------------------------------------------------

    async def _evaluate_q1(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate Q1 (multi-step planning) using final v3 prompt + image + DAG."""
        if not records:
            return {"count": 0, "mean_score": 0, "scores": []}

        from ..inference.client import AsyncModelClient

        # Build evaluation tasks with enriched data
        tasks = []
        for record in records:
            record_id = record["request_id"]
            raw_question = record["raw_question"]
            image_path = Path(raw_question["image_urls"][0])
            if not image_path.is_file():
                raise FileNotFoundError(f"Planning image not found: {image_path}")

            tasks.append(
                {
                    "id": record_id,
                    "gt_answer": record["gt_answer"],
                    "model_response": record["response"] or "",
                    "question": raw_question["question"],
                    "image_path": str(image_path),
                    "dag_info": self._find_dag(record),
                }
            )

        # Build API messages
        batch_requests = []
        for task in tasks:
            msgs = self._build_q1_messages(task)
            batch_requests.append(
                {
                    "request_id": task["id"],
                    "messages": msgs,
                }
            )

        # Call API
        client = AsyncModelClient(self._api_config())
        responses = await client.run_batch(
            batch_requests,
            model=self.eval_model,
            save_path=self._batch_save_path("q1"),
        )

        # Parse scores
        scores = []
        score_details = []
        response_map = {response["id"]: response for response in responses}

        for task in tasks:
            raw_resp = response_map[task["id"]]["response"] or ""
            parsed = self._parse_q1_score(raw_resp)

            if parsed is not None:
                node_score = parsed["node_correctness"]
                comp_score = parsed["task_completion"]
                combined = (node_score + comp_score) / 20.0 * 100.0
                scores.append(combined)
                score_details.append(
                    {
                        "id": task["id"],
                        "node_correctness": node_score,
                        "task_completion": comp_score,
                        "combined": combined,
                        "raw_response": raw_resp,
                    }
                )
            else:
                score_details.append(
                    {
                        "id": task["id"],
                        "node_correctness": None,
                        "task_completion": None,
                        "combined": None,
                        "raw_response": raw_resp,
                        "parse_error": True,
                    }
                )

        valid_scores = [s for s in scores if s is not None]
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        return {
            "count": len(tasks),
            "mean_score": round(mean_score, 2),
            "scores": scores,
            "details": score_details,
            "parse_failures": len(tasks) - len(valid_scores),
        }

    def _build_q1_messages(self, task: Dict[str, Any]) -> List[Dict]:
        """Build evaluation messages for Q1 (final v3 style)."""
        image_path = task["image_path"]
        media_type = "image/jpeg"
        with open(image_path, "rb") as f:
            img_data = f.read()
        img_b64 = base64.b64encode(img_data).decode()
        if img_data[:3] == b"\xff\xd8\xff":
            media_type = "image/jpeg"
        elif img_data[:8] == b"\x89PNG\r\n\x1a\n":
            media_type = "image/png"

        # Build evaluation text
        eval_text = f"""Task Instruction:
{task["question"]}

Ground Truth Action List:
{task["gt_answer"]}

GT DAG Dependencies:
{task["dag_info"]}

Model Plan Action List:
{task["model_response"]}

Evaluate the model plan. Output JSON only."""

        is_claude = self.eval_model.lower().startswith("claude")

        img_content = {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{img_b64}"},
        }
        text_content = {"type": "text", "text": eval_text}

        if is_claude:
            # Claude: put prompt in user message, no system role with images
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_V3 + "\n\n" + eval_text},
                        img_content,
                    ],
                }
            ]
        else:
            # OpenAI / Gemini: system prompt + user content [image, text]
            return [
                {"role": "system", "content": PROMPT_V3},
                {"role": "user", "content": [img_content, text_content]},
            ]

    @staticmethod
    def _parse_q1_score(response: str) -> Optional[Dict[str, int]]:
        """Parse Q1 evaluation score from LLM response."""
        if not response:
            return None

        cleaned = response.strip()
        # Strip markdown code blocks
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)

        # Find JSON object
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if not m:
            return None

        try:
            data = json.loads(m.group())
            nc = data["node_correctness"]
            cc = data["task_completion"]
            n_score = nc["result"]
            c_score = cc["result"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

        def to_0_10_int(value):
            if value is None:
                return None
            if isinstance(value, (int, float)):
                score = int(value)
                return score if 0 <= score <= 10 else None
            text = str(value).strip()
            match = re.search(r"-?\d+(?:\.\d+)?", text)
            if not match:
                return None
            score = int(float(match.group()))
            return score if 0 <= score <= 10 else None

        n_score = to_0_10_int(n_score)
        c_score = to_0_10_int(c_score)

        if n_score is None or c_score is None:
            return None

        return {
            "node_correctness": n_score,
            "task_completion": c_score,
        }

    # ------------------------------------------------------------------
    # Q2 Evaluation (unchanged logic, improved parsing)
    # ------------------------------------------------------------------

    async def _evaluate_q2(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate Q2 (single-step planning) records."""
        if not records:
            return {"count": 0, "mean_score": 0, "scores": []}

        from ..inference.client import AsyncModelClient

        # Step 1: Extract steps
        extract_prompts = []
        for record in records:
            prompt_text = PROMPT_TEMPLATE_EXTRACT_STEP.format(response=record["response"] or "")
            extract_prompts.append(
                {
                    "id": record["request_id"],
                    "prompt": prompt_text,
                    "record": record,
                }
            )

        client = AsyncModelClient(self._api_config())
        messages = []
        request_ids = []
        for p in extract_prompts:
            messages.append([{"role": "user", "content": p["prompt"]}])
            request_ids.append(p["id"])

        extract_results = await client.run_batch(
            [{"request_id": rid, "messages": msg} for rid, msg in zip(request_ids, messages)],
            model=self.eval_model,
            save_path=self._batch_save_path("q2_extract"),
        )

        # Step 2: Compare steps
        compare_prompts = []
        for i, record in enumerate(records):
            extract_result = extract_results[i]
            if not extract_result["response"]:
                continue

            extracted_step = extract_result["response"].strip()
            gt_step = record["gt_answer"].split("-")[-1]

            prompt_text = PROMPT_TEMPLATE_COMPARE_STEPS.format(
                extracted_step=extracted_step,
                gt_step=gt_step,
            )
            compare_prompts.append(
                {
                    "id": record["request_id"],
                    "prompt": prompt_text,
                    "extracted_step": extracted_step,
                    "gt_step": gt_step,
                }
            )

        if compare_prompts:
            messages = []
            request_ids = []
            for p in compare_prompts:
                messages.append([{"role": "user", "content": p["prompt"]}])
                request_ids.append(p["id"])

            compare_results = await client.run_batch(
                [{"request_id": rid, "messages": msg} for rid, msg in zip(request_ids, messages)],
                model=self.eval_model,
                save_path=self._batch_save_path("q2_compare"),
            )
        else:
            compare_results = []

        # Step 3: Calculate scores
        scores = []
        score_details = []
        for i, p in enumerate(compare_prompts):
            result = compare_results[i]
            parsed = self._parse_q2_score(result["response"] or "")

            if parsed is not None:
                skill = parsed["skill_usage_accuracy"]
                obj = parsed["operation_object_reasonableness"]
                param = parsed["parameter_accuracy"]
                # Dependency rule
                if skill == 0 or obj == 0:
                    param = 0
                score = (skill + obj + param) / 3.0 * 100.0
                scores.append(score)
                score_details.append(
                    {
                        "id": p["id"],
                        "skill": skill,
                        "object": obj,
                        "parameter": param,
                        "score": score,
                    }
                )
            else:
                score_details.append({"id": p["id"], "score": None, "parse_error": True})

        valid_scores = [s for s in scores if s is not None]
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        return {
            "count": len(records),
            "mean_score": round(mean_score, 2),
            "scores": scores,
            "details": score_details,
            "parse_failures": len(records) - len(valid_scores),
        }

    @staticmethod
    def _parse_q2_score(response: str) -> Optional[Dict[str, float]]:
        """Parse Q2 comparison score from LLM response (improved)."""
        if not response:
            return None

        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)

        m = re.search(r"\{[\s\S]*\}", cleaned)
        if not m:
            return None

        try:
            data = json.loads(m.group())
            skill = data["skill_usage_accuracy"]["result"]
            obj = data["operation_object_reasonableness"]["result"]
            param = data["parameter_accuracy"]["result"]

            # Validate types and ranges
            def to_float(v):
                if v is None:
                    return None
                try:
                    f = float(v)
                    return f if 0 <= f <= 1 else None
                except (ValueError, TypeError):
                    return None

            skill = to_float(skill)
            obj = to_float(obj)
            param = to_float(param)
            if skill is None or obj is None or param is None:
                return None
            return {
                "skill_usage_accuracy": skill,
                "operation_object_reasonableness": obj,
                "parameter_accuracy": param,
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Q3 Evaluation (unchanged)
    # ------------------------------------------------------------------

    async def _evaluate_q3(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate Q3 (yes/no) records."""
        if not records:
            return {"count": 0, "mean_score": 0, "scores": []}

        from ..inference.client import AsyncModelClient

        prompts = []
        for record in records:
            prompt_text = PROMPT_TEMPLATE_EXTRACT_YES_NO.format(response=record["response"] or "")
            prompts.append(
                {
                    "id": record["request_id"],
                    "prompt": prompt_text,
                    "gt_answer": record["gt_answer"].lower().strip(),
                }
            )

        client = AsyncModelClient(self._api_config())
        messages = []
        request_ids = []
        for p in prompts:
            messages.append([{"role": "user", "content": p["prompt"]}])
            request_ids.append(p["id"])

        results = await client.run_batch(
            [{"request_id": rid, "messages": msg} for rid, msg in zip(request_ids, messages)],
            model=self.eval_model,
            save_path=self._batch_save_path("q3"),
        )

        scores = []
        score_details = []
        for i, p in enumerate(prompts):
            result = results[i]
            if result["response"]:
                extracted = result["response"].strip().lower()
                is_correct = extracted == p["gt_answer"]
                score = 100.0 if is_correct else 0.0
            else:
                is_correct = False
                score = 0.0
            scores.append(score)
            score_details.append(
                {
                    "id": p["id"],
                    "extracted": extracted if result["response"] else None,
                    "gt": p["gt_answer"],
                    "correct": is_correct,
                    "score": score,
                }
            )

        mean_score = sum(scores) / len(scores) if scores else 0

        return {
            "count": len(records),
            "mean_score": round(mean_score, 2),
            "scores": scores,
            "details": score_details,
        }

    # ------------------------------------------------------------------
    # Overall Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_overall(
        q1_scores: Dict[str, Any],
        q2_scores: Dict[str, Any],
        q3_scores: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate overall scores across Q1/Q2/Q3."""
        total_count = q1_scores["count"] + q2_scores["count"] + q3_scores["count"]

        q1_mean = q1_scores["mean_score"]
        q2_mean = q2_scores["mean_score"]
        q3_mean = q3_scores["mean_score"]
        q1_count = q1_scores["count"]
        q2_count = q2_scores["count"]
        q3_count = q3_scores["count"]

        if total_count > 0:
            overall_mean = (
                q1_mean * q1_count + q2_mean * q2_count + q3_mean * q3_count
            ) / total_count
        else:
            overall_mean = 0

        return {
            "total_count": total_count,
            "overall_mean": round(overall_mean, 2),
            "q1_mean": round(q1_mean, 2),
            "q2_mean": round(q2_mean, 2),
            "q3_mean": round(q3_mean, 2),
        }

    def _api_config(self):
        """Return the benchmark API configuration."""
        return self.api_config
