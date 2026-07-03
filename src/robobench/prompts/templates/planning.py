"""
统一Prompt模板文件
基于FlagEvalMM的设计模式，整合extract_plan_information_prompt和dag_generation_prompt
"""

########################
# 预定义动作函数 (与FlagEvalMM中的PREDEFINED_ACTIONS保持一致)
########################
PREDEFINED_ACTIONS = """
#### **Functions for the actions of a gripper**
  -  `move_to(object, target_object)`: This function represents the movement of the gripper, with the first parameter representing the object held in the hand during movement and the second parameter representing the target object. If there is nothing in hand, the first parameter is 'none'. For example, move_to(none, towel) means that the gripper is moving towards a towel with nothing in hand. And move_to(panda_toy, bowl) means that the gripper with a panda toy is moving towards a bowl.
  -  `hold(object)`: This function represents the static state of the gripper with an object. Note that if there is nothing in gripper, this function is not applicable. For example, hold(cup) means that the gripper is holding a cup and keeping static.

#### **Functions for grabbing and releasing**
  - `pick_up(object)`: This function represents that the gripper picks up an object. Note that the object must be graspable. For example, pick_up(apple) means that the gripper picks up an apple.
  - `grasp(object)`: This function represents that the gripper touches and lightly grabs the object. Note that objects can be either pick-upable or non-pick-upable. The difference between this function and pick_up is that grasp means to hold lightly, while pick_up usually means to lift up after contact. For example, grasp(door_handle) means that the gripper grabs the door handle (not lift up).
  - `place(object, target_object)`: This function represents that the gripper place the object at the location of the target_object. Note that target_object can be a specific object or a position relative to a reference object. For example, place(apple, table) means that placing an apple on the table. And place(apple, right_of_banana) means that placing an apple on the right of a banana.

#### **Functions for using a tool to operate objects**
  - `scoop(tool, contents, container)`: This function represents that the gripper holding the tool is scooping something in a container with the tool. If "contents" is uncertain, use "unknown" as the second parameter. For example, scoop(spoon, water, bowl) means that the gripper holds the spoon and uses it to scoop water from the bowl.
  - `pour(container, contents, target_container)`: This function represents that the gripper holding the container is pouring something into the target_container. If "contents" is uncertain, use "unknown" as the second parameter. For example, pour(bowl, water, pot) means that the gripper holds a bowl with water and pours water into a pot.
  - `wipe(tool, object, target_object)`: This function represents that the gripper is using a tool to wipe "object" on the "target_object". For example, wipe(towel, water, table) means that the gripper is wiping water on the table using a towel. If "object" is uncertain, use "unknown" instead.
  - `stir(tool, contents, target_container)`: This function represents that the gripper is using a tool to stir "contents" in the "target_container". For example, stir(spoon, soup, pot) means that using a spoon to stir soup in the pot. If "contents" is uncertain, use "unknown" instead.
  - `draw(tool, character, target_object)`: This function represents that the gripper is drawing a character using a tool on the target_object. For example, draw(marker, 'A', whiteboard) means that drawing an 'A' with a marker on the whiteboard.
  - `cut(tool, object, target_object)`: This function represents that the gripper is cutting object with a tool on the target_object. For example, cut(knife, tomato, chopping_board) means that the gripper is cutting a tomato with a knife on the chopping_board.

#### **Functions for interacting with objects directly**
  -  `fold(object, target_position)`: This function represents that the gripper is folding object towards the target_position. For example, fold(left_side_of_towel, right_side_of_towel) means that the gripper holds the left side of the towel and folds it to the right side.
  -  `unfold(object, target_position)`: This function represents that the gripper is unfolding object towards the target_position. For example, unfold(left_side_of_towel, right_side_of_towel) means that the gripper holds the left side of the towel and unfolds it to the right side.
  -  `turn(object, direction, state_of_target_object)`: This function represents that the gripper rotates the object in a certain direction to the target position or state. The 'direction' can be chosen in {{clockwise, anticlockwise, up, down, forward, backward, left, right}}. If it is an articulated object (it originally has axis rotation, such as a faucet), use the following {{up, down, forward, backward, left, right}}. If it is originally a rigid body (it originally has no axis rotation, such as a bottle adjusting its direction in the air), use clockwise and counterclockwise. For example, turn(faucet, clockwise, middle_of_sink) means that turn faucet right until it faces the middle of sink.
  -  `press(tool, object)`: This function represents that the gripper press object using a tool. If there is no tool, use 'none' instead. For example, press(none, red_button) means that pressing a red button using the gripper directly.
  -  `push(object, target_location)`: This function represents that the gripper pushes the object to the target_location. For example, push(chair, under_of_table) means that pushing a chair under to a table.
  -  `pull(object, target_location)`: This function represents that the gripper pulls the object to the target_location. For example, pull(towel, right_side_of_table) means that pulling a towel to the right side of a table.
  -  `insert(object, target_object)`: This function represents that the gripper inserts the object into the target object. For example, insert(plug, socket) means inserting a plug into the socket.
  -  `pullout(object, target_object)`: This function represents that the gripper pullouts the object from the target object. For example, pullout(plug, socket) means pullout a plug from the socket.

#### **Function for only dual-arm tasks**
  -  `transfer(left/right, right/left, object)`: This function represents the gripper transfers an object from one hand to another hand. For example, transfer(left, right, bottle) means that transfering bottle from left hand to right hand.

#### **Functions for only mobile-manipulation tasks**
  -  `observation(object)`: This function represents that the target object is not in the field of view and needs to be found. For example, observation(chair) means that finding the chair.
  -  `mobile(target_object)`: This function represents that the target object is in the field of view but too far from the robot to operate, so the robot needs to move the chassis to approach the target position. For example, mobile(table) means moving the chassis to approach the table.

#### **No Operation**
  - `no_ops`: Stay still or keep the current state.
"""

########################
# 统一计划信息提取模板 (Q1 Planning) - 基于FlagEvalMM的PROMPT_TEMPLATE_EXTRACT_PLAN_INFO
########################
PROMPT_TEMPLATE_EXTRACT_PLAN_INFO = """
### **Prompt: Extracting Structured Action Plans from Robotic Task Descriptions**

**Task:**
You are given an input dataset containing a robotic manipulation task goal, a previously executed step, and a response describing the remaining steps. Your task is to extract structured action plans in a specific function format.

---

### **Instructions:**
1. **Extract Key Information:**
   - Identify the **task goal** from the `prompt` field and assign it to the `"task_summary"` field.
   - Extract action functions from the `previous_step` and `response` fields to construct a sequence of necessary steps in `"plan_step"`.

2. **Strict Action Function Format:**
   - Use only the predefined action functions listed below. **Do not modify function names or introduce new ones.**
   - Ensure that all extracted function names match exactly with the ones provided.
   - Arguments (`object`, `target_object`, `carry_object`, `direction`) should be **generalized** based on input information but remain **faithful** to the task.

3. **Maintain Execution Order:**
   - The `"plan_step"` list should strictly follow the order in which the robot should execute them.

4. **Determine Action Format Based on Input:**
   - **Single-arm tasks**: If the `response` field contains actions without `left:` or `right:` prefixes, extract actions in single-arm format (e.g., `move_to(object, target)`).
   - **Dual-arm tasks**: If the `response` field contains actions with `left:` or `right:` prefixes, extract actions in dual-arm format (e.g., `left:move_to(object, target), right:no_ops`).
   - **Automatic detection**: Analyze the input content to determine whether it's a single-arm or dual-arm task and format the output accordingly.

5. **Strictly No Assumptions:**
   - Only extract actions explicitly present in the input.
   - Do **not** add missing steps based on assumptions.
   - **Do not reinterpret or correct possibly incorrect actions or arguments** — preserve the input as given.

6. **Reasoning Explanation:**
   - Provide a `"reason"` field explaining how the `"task_summary"` and `"plan_step"` were derived, including how you determined the format (single-arm vs dual-arm).

---

### **Predefined Action Functions**
""" + PREDEFINED_ACTIONS + """

---

### **Output Format (JSON)**
Your response **must be strictly formatted** as a single-line JSON object:

**For Single-Arm Tasks:**
```json
{{
  "task_summary": "<task goal>",
  "plan_step": ["<action_function_1>", "<action_function_2>", ...],
  "reason": "<your reasoning>"
}}
```

**For Dual-Arm Tasks:**
```json
{{
  "task_summary": "<task goal>",
  "plan_step": [
      "<action_function_1>",
      "left:<action_function_2>, right:<action_function_3>",
      ...
  ],
  "reason": "<your reasoning>"
}}
```

### **Examples**

#### **Example 1: Single-Arm Task**
**Input:**
```json
{{
  "prompt": "With <placing a roll of toilet paper onto a holder> as the goal and some steps completed, what are the remaining things to do?",
  "previous_step": "1-move_to(toilet_paper)",
  "response": "From the sequence of images, it appears that a robotic system is in the process of placing the roll of toilet paper onto its holder. Here are the remaining things to do:\\n\\n1. Align the Toilet Paper Roll with the Holder's Rod.  \\n2. Insert the Rod Through the Roll.  \\n3. Attach the Rod and Roll to the Holder Mechanism.  \\n4. Finalize the Setup."
}}
```

**Expected Output:**
```json
{{
  "task_summary": "<placing a roll of toilet paper onto a holder>",
  "plan_step": ["move_to(toilet_paper)", "place(toilet_paper, holder)"],
  "reason": "Detected single-arm task based on response format without left/right prefixes. The first step 'move_to(toilet_paper)' is obtained from the 'previous_step' field, and the second step 'place(toilet_paper, holder)' is derived from the 'prompt' and 'response' fields by summarizing the remaining steps."
}}
```

#### **Example 2: Dual-Arm Task**
**Input:**
```json
{{
  "prompt": "With <cooking shrimp in a pan and serving it in a bowl> as the goal and some steps completed, what are the remaining things to do?",  
  "previous_step": "1-observation(shrimp)",  
  "response": "To plan the remaining steps to achieve the goal of cooking shrimp in a pan and serving it in a bowl, we'll organize the steps required for the robot...\\n\\n1-left:move_to(none, shrimp), right:no_ops\\n2-left:pick_up(shrimp), right:no_ops\\n3-left:no_ops, right:move_to(none, pan)\\n4-left:no_ops, right:turn_on(stove)\\n5-left:no_ops, right:pour(oil, pan)\\n6-left:no_ops, right:move_to(shrimp, pan)\\n7-left:no_ops, right:scoop(shrimp, pan)\\n8-left:no_ops, right:move_to(pan, bowl)\\n9-left:no_ops, right:pour(shrimp, bowl)"
}}
```

**Expected Output:**
```json  
{{
  "task_summary": "<cooking shrimp in a pan and serving it in a bowl>",  
  "plan_step": [  
      "observation(shrimp)",  
      "left:move_to(none, shrimp), right:no_ops",  
      "left:pick_up(shrimp), right:no_ops",  
      "left:no_ops, right:move_to(none, pan)",  
      "left:no_ops, right:turn_on(stove)",  
      "left:no_ops, right:pour(oil, pan)",  
      "left:no_ops, right:move_to(shrimp, pan)",  
      "left:no_ops, right:scoop(shrimp, pan)",  
      "left:no_ops, right:move_to(pan, bowl)",  
      "left:no_ops, right:pour(shrimp, bowl)"  
  ],
  "reason": "Detected dual-arm task based on response format with left/right prefixes. The 'task_summary' field is extracted from the prompt. The 'plan_step' follows the order in 'previous_step' and 'response'. The 'observation' step is included as a standalone action, while all manipulation actions use the 'left:' and 'right:' format. No missing steps were assumed or inferred."
}}
```

The data I provide is as follows:
{data}
Please output your results as required.

### **Final Instructions**
- **Automatically detect task type**: Analyze the input to determine if it's single-arm or dual-arm and format accordingly.
- **Do not modify function names.**
- **Do not add missing steps beyond the input information.**
- **Ensure correct function selection based on context.**
- **Ensure the `"plan_step"` field is strictly ordered.**
- **For dual-arm tasks: All manipulation actions are formatted with `left:` and `right:` unless explicitly global actions.** If an arm does not perform any action in a step, use `no_ops`.
- **Standalone movement (`mobile`) and observation (`observation`) actions must NOT have `left:` or `right:` prefixes in dual-arm tasks.**
- **Output a single-line JSON object with no extra content.**
- **Do not reinterpret or correct possibly incorrect actions or arguments** — preserve the input as given.
"""

########################
# 统一Q1评估模板 - 基于FlagEvalMM的PROMPT_TEMPLATE_Q1_EVALUATION，融合了原有的DAG评估逻辑
########################
PROMPT_TEMPLATE_Q1_EVALUATION = """
You are a judge of embodied multimodal task planning. Evaluate a model's plan under visual-world simulation with concise, strict rules.

### Inputs
You will receive:
* First scene image — initial conditions and visual constraints (layout, obstacles, reachability).
* Ground Truth (GT) Action List (from segmented static video).
* Manually annotated DAG (task order/parallelism; both text and image).
* Model Plan Action List (model only sees the image and question; DAG is for scoring only).

Input data format:
{{
    'GT action list': [...],
    'GT dag': [...],
    'model plan action list': [...]
}}

### Automatic Mode Detection
- Standard Mode: textual object names (e.g., pick_up(apple)).
- CSS Mode: numeric object IDs (e.g., pick_up(3)); must refer by IDs exactly.
- Detect by scanning both GT and model action lists.

### Predefined Action Functions
""" + PREDEFINED_ACTIONS + """

### Evaluation (two scores, 0-10 each)

1) Node correctness
- Node = (skill, object(s), parameters) after normalization.
- Count exact matches against GT (one-to-one best match; no partial credit per node).
- A node is correct iff: skill identical; object semantics match (or IDs equal in CSS); parameters appropriate after normalization.
- Show which nodes match and the proportion Y/X.
- Score = floor((Y/X)*10). Also compute (Y*10)//X; if unequal, use the integer-division result.
- Note: treat function-object equivalences that reflect the same manipulation (e.g., grasp(drawer) ~ grasp(drawer_handle); pull(drawer_handle,outward) ~ pull(drawer,open)).

2) Task completion degree
- Focus on manipulated object state changes, not robot motions.
- Identify critical object states from image + GT + DAG; enforce DAG dependencies and realism.
- Only count states achievable under visual/physical constraints; failed prerequisites block dependents.
- Score = floor((Achieved/Total)*10); also compute (Achieved*10)//Total and ensure same integer.

Critical state identification rules (concise):
- Exclude robot-only steps: move_to, grasp, hold, approach, align, pregrasp, release, retract, look, scan, plan, think.
- Count only physical property changes: pose/support/containment, orientation/angle, openness, activation, assembly.
- pick_up(obj) counts only if the object is lifted off its prior support (support→gripper).

### State Rollout Engine (mandatory)
Evaluate strictly via ordered simulation; compute scores only from achieved states in the rollout ledger.
- A. Object state machine: maintain pose/support/contact; orientation/angle; openness∈{{closed, partially_open(f∈[0,1]), fully_open}}; containment/attachment; activation; integrity/assembly.
- B. Skill→Effects rules (arrow steps):
  1) push(drawer, dir, dist/force?) → Preconditions: reachable; dir aligns with slide (outward/forward); no blocking → Physics: force > static friction; travel ≤ remaining; obstacle truncates → Effects: openness↑; if open_fraction ≥ threshold → opened(partially/fully).
  2) pull(drawer, dir, dist) → Preconditions: as push (outward) → Physics: same → Effects: openness↑ (same counting).
  3) push(door, dir) → Preconditions: act on non-hinge side; openable dir; not latched/blocked → Physics: torque > hinge resistance; stop at obstacle → Effects: door_angle↑; if ≥ threshold → opened.
  4) turn(knob/tap, cw/ccw, angle/target) → Preconditions: reachable; free rotation space → Physics: limited by endstop → Effects: angle updated; cross on/open target → activation/open changes.
  5) place(obj, receptacle/area) → Preconditions: object controlled; valid support → Physics: stable placement; size fits → Effects: pose/support updated; into container → containment=in.
  6) pick_up(obj) → Preconditions: grasp feasible → Physics: must lift off prior support → Effects: if lifted then support=gripper; clamp-only without lift → no state change.
- Default thresholds: drawer opened if open_fraction ≥ 0.3; door opened if angle ≥ 20°.
- C. Visual/physical constraints: direction correctness (drawer outward to open); reachability/occlusion gate preconditions; obstacles truncate travel; insufficient force → fail; cw/ccw synonyms normalized.
- D. Rollout ledger: for each step, normalize skill; check preconditions; if pass, apply effects and record state deltas (including numeric open_fraction/angle); mark newly satisfied critical states; enforce DAG dependencies; later reversals overwrite earlier.
- E. Drawer/Door specifics: compute d_eff = min(requested, remaining, clearance_limit); wrong direction → no progress; blocked → d_eff≈0; opened then closed → not counted.
- F. Redundant integer check: define X, Y; show floor(Z*10) and (Y*10)//X; use the latter if mismatch.

### Output format (JSON only)
```json
{{
    "node_correctness" : {{"reason": "Detailed analysis: List which specific nodes match, calculate exact proportion, explain scoring logic. Format: 'GT nodes: [list]. Model nodes: [list]. Matching nodes: [list]. Total GT nodes: X. Correct model nodes: Y. Proportion: Y/X = Z. Score: floor(Z*10) = W; (Y*10)//X = W_alt; Final score = W (must equal W_alt). Also include an Object Matching Table and a one-to-one node matching table.'", "result": x}},
    "task_completion_degree" : {{"reason": "Detailed world simulation analysis: 1) Visual constraint analysis from image. 2) Identify critical object state changes (exclude robot arm actions). 3) Physics-based realistic simulation. 4) Use DAG for dependencies. 5) Step-by-step simulation with constraints. 6) Rollout trajectory table with state deltas. Format: 'Visual constraints: [analysis]. Critical object states: [list with dependencies and thresholds]. Physics simulation + rollout: [step-by-step analysis with preconditions, effects, and state delta]. Reachable states: [list]. Total critical states: X. Achieved states: Y. Proportion: Y/X = Z. Score: floor(Z*10) = W; (Y*10)//X = W_alt; Final score = W (must equal W_alt).’", "result": y}},
    "planning_issue_analysis": {{"issue_types": ["list of issue categories"], "detailed_analysis": "Comprehensive categorization and analysis of planning problems found in the model's plan. Categories may include: wrong_object, wrong_order, missing_steps, impossible_actions, constraint_violations, physics_violations, spatial_reasoning_errors, parameter_mismatch, aliasing_errors, CSS_mode_reference_errors, etc. Provide specific examples and explanations for each identified issue type."}},
    "comprehensive_evaluation": "Give an overall evaluation as a world simulator, point out the highlights and shortcomings of the model planning performance, and give actionable suggestions for improvement."
}}
```

The data I provide is as follows:
{data}
Please output your results as required.

### Final Guidelines
- Auto-detect Standard vs CSS Mode.
- Object reference: Standard uses text (with alias rules); CSS uses exact IDs (textual names → mismatch). Common alias examples: tap~faucet, fridge~refrigerator, sofa~couch, trash_can~bin, cup~mug, pan~frying_pan, pot~saucepan, phone~cellphone; affordance-critical adjectives must match (red_button≠green_button; left/right; upper/lower).
- Canonicalization: skills lowercase and strip separators (turn_on/turn-on/turn on→turnon); do not infer new skills.
- Parameter normalization: cw~clockwise; ccw~counterclockwise; binary (open/closed, on/off, locked/unlocked); spatial (in/on/under/front_of/behind); numeric exact or within GT-range; nominal equivalents allowed (e.g., fully_open).
- One-to-one best matching; show arithmetic with both floor(Z*10) and (Y*10)//X.
- Rollout-first rule: only count states achieved in the ledger; failed prerequisites block dependents; no partial credit for unreachable states.

Few-shot anchors:
- A Tap rotation (Standard): GT turn(tap, ccw, open_angle) vs Model turn(faucet, anticlockwise, open_angle) → match → Y=1/X=1 → 10.
- B Fruits to bowl (3 parallel): achieve 2/3 → 6.
- C grasp is NOT a state change: move_to/grasp/hold only → 0.
- D CSS strictness: GT pick_up(3), place(3,5); Model pick_up(cup) → mismatch.
- E Drawer push opens: push(drawer, forward, 0.4*d_max) passes; open_fraction 0→0.4≥0.3 → count 1/1 → 10.
"""

########################
# Q2 Planning相关模板 (基于FlagEvalMM) Extract the next step from this response.
########################
PROMPT_TEMPLATE_EXTRACT_STEP = """
Extract the next step from this response. The step should be in the format: skill(element1, element2, ...)

For example:
- "grasp(microwave_handle)"
- "push(microwave_handle, close)"
- "move_to(none, drawer)"

Response: {response}

Return ONLY the step in the correct format, nothing else.
"""

########################
# Q2 Planning相关模板 (基于FlagEvalMM) Compare the extracted step with the ground truth step.
########################
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

The data I provide is as follows:

Extracted step: {extracted_step}
Ground truth step: {gt_step}

Please output your results as required.
"""

########################
# Q3 Planning相关模板 (基于FlagEvalMM)
########################
PROMPT_TEMPLATE_EXTRACT_YES_NO = """
Extract ONLY the yes/no answer from this response.
Return ONLY "yes" or "no" with no other text or explanation.

Response: {response}
"""

########################
# 其他必要的模板 (保持与FlagEvalMM兼容)
########################
PROMPT_TEMPLATE_NORMALIZE_CHOICE = """
Extract ONLY the final multiple choice answer(s) from this response.
Return ONLY the letter(s) A,B,C,D with no other text or explanation.
If multiple answers, separate with commas.
Response: {answer}
"""

PROMPT_TEMPLATE_POINT = """
You are given a coordinate string that represents a point (x, y) in various possible formats.
Your task is to extract and normalize it to the standard format (x, y) where x and y are numbers.

Examples:
- Input: "(0.5, 0.7)" -> Output: "(0.5, 0.7)"
- Input: "[0.3, 0.8]" -> Output: "(0.3, 0.8)"
- Input: "The point is at 0.2, 0.9" -> Output: "(0.2, 0.9)"
- Input: "x=0.1, y=0.6" -> Output: "(0.1, 0.6)"
- Input: "coordinates: (0.4, 0.2)" -> Output: "(0.4, 0.2)"

Return ONLY the normalized coordinate in format (x, y) with no additional text.
If you cannot find valid coordinates, return "(0.0, 0.0)".

Input coordinate string: {coord_string}
"""

PROMPT_TEMPLATE_TRAJECTORY = """
You are given a trajectory string that represents a sequence of points (x, y) in various possible formats.
Your task is to extract and normalize it to the standard format [[x1,y1], [x2,y2], ...] where x and y are numbers.

Examples:
- Input: "[[0.5, 0.7], [0.6, 0.8]]" -> Output: [[0.5, 0.7], [0.6, 0.8]]
- Input: "[(0.3, 0.8), (0.4, 0.9)]" -> Output: [[0.3, 0.8], [0.4, 0.9]]
- Input: "The trajectory is: 0.2,0.9 ; 0.3,1.0 ; 0.4,1.1" -> Output: [[0.2, 0.9], [0.3, 1.0], [0.4, 1.1]]
- Input: "points: x=0.1,y=0.6 ; x=0.2,y=0.7" -> Output: [[0.1, 0.6], [0.2, 0.7]]
- Input: "Trajectory: [(0.4, 0.2), (0.5, 0.3), (0.6, 0.4)]" -> Output: [[0.4, 0.2], [0.5, 0.3], [0.6, 0.4]]

Return ONLY the normalized trajectory in format [[x1,y1], [x2,y2], ...] with no additional text.
If you cannot find valid trajectory, return [[0.0, 0.0]].

Input trajectory string: {trajectory_string}
"""

PROMPT_TEMPLATE_EVAL_OPEN = """
You are evaluating the accuracy of a response to an open-ended question.

Ground truth answer: {gt_answer}

Response to evaluate: {response}

On a scale from 0 to 1, how accurate is the response compared to the ground truth?
- Score 1.0: The response is completely correct and covers all key points in the ground truth.
- Score 0.5: The response is partially correct but misses important details or has significant inaccuracies.
- Score 0.0: The response is completely incorrect or unrelated to the ground truth.

Provide your evaluation in the following JSON format:
{{
    "score": <score as a float between 0 and 1>,
    "explanation": "<brief explanation of your scoring>"
}}
"""
