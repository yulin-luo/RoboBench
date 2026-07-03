def get_prompt(desc: str, cnt: int) -> str:
    return f"""You will analyze a **first-person perspective (ego-centric) video** (represented by image frames) of a robot performing a specific task, where the task is described as: ``{desc}``. Note that the referenced task summary might not be accurate or complete. 

Your task is to identify the primary task during the video with the help of the referenced description, summarize the task and rewrite the description, extract the necessary steps to complete it, and specify the frame range for each step. Each step will include a **state** and a corresponding **action description**. Please ensure that the division of atomic tasks is as detailed as possible, the task description is as clear as possible, and there are no vague descriptions. Follow these instructions:

1. **Task Identification**: First, identify the main task the robot is performing. This task could be a clear goal or a series of related activities (e.g., assembling furniture, repairing equipment, preparing food, etc.). Briefly describe the primary task in one sentence.

2. **Step Extraction**: Once the task is identified, extract the key steps required to complete it, ensuring that each step is clearly described and logically ordered. Each step consists of two parts:
    - **State**: The state of the robot during the step—whether it is moving (`[mobile]`), manipulating (`[manipulation]`), or observing (`[observation]`):
        - `[mobile]`: The robot changes position, but the camera view does not rotate. Describe the movement of the robot (e.g., "Move towards the table").
        - `[manipulation]`: The robot's position and camera view remain static, but the hands are performing actions. Use `[left]`, `[right]`, or `[both]` to describe the actions of the hands (e.g., "[left] holds the frame, [right] tightens the screws").
        - **[observation]**: This state **must** be triggered when the ego-centric camera's view or orientation changes, indicating the robot is observing or searching for an object. This step is required as a transition whenever the camera view changes, regardless of the robot's other actions. For each **[observation]** state, use the following format:
          - **[goal]**: Clearly specify the object or scene the robot is searching for, using **[find]** to indicate what the robot is looking for or detecting (e.g., "[find] the legs of the desk").
          - **[current object or scene]**: Describe what the robot currently sees (e.g., "A table and a chair are visible").
          - **[search result]**: Indicate whether the target object has been found (`yes`, `no`, `part`).
    - **Action Description**: A concise description of what happens during the step, based on the state.

3. **Output Format**: Provide the task description and steps in two parts, formatted as JSON:
    - **Task Summary**: A string summarizing the primary task in the video without mentioning the subjects - the robot's hands or the mobile base.
    - **Steps**: An array where each element represents a step, containing:
        - `state`: The current state of the robot (`[mobile]`, `[manipulation]`, or `[observation]`).
        - `action_description`: A detailed description of the action in the current state (e.g., "[mobile] Move towards the table," "[manipulation] [left] holds the frame, [right] tightens the screws," "[observation] [goal]: [find] the legs of the desk. [current object or scene]: A table is visible. [search result]: no"). Always describe `[left]` first and `[right]` second if the two hands have different functions, when applicable.
        - `start_frame`: The start frame of the step (from `0` to `{cnt-1}`).
        - `end_frame`: The end frame of the step (from `0` to **{cnt-1}**). 
           
4. **Frame Range Constraints**:
    - **No Overlapping Frames**: Ensure that the frame ranges of each step do not overlap with each other. Each frame should be assigned to exactly one step.
    - **Full Frame Coverage**: Ensure that all {cnt} frames (from 0 to {cnt-1}) are included in the steps. No frames should be missed or duplicated.

5. **Important Notes**:
    - Please annotate as finely as possible and try not to have one step with more than ten frames doing the same thing (unless it is surely difficult to distinguish).
    - A step has only one verb (unless two or more actions are strictly performed simultaneously)
    - Please ensure the accuracy of labeling and image matching.  
    - The video consists of exactly **{cnt} frames**. Therefore, ensure that the **end frame** for the last step is always **{cnt-1}**, and no frame should exceed this value. Ensure that all {cnt} frames (from 0 to {cnt-1}) are included in the steps. No frames should be missed or duplicated.
    - Ensure that the frame ranges of each step do not overlap with each other. Each frame should be assigned to exactly only one step.
    - The **[observation]** state should appear **at the beginning** of the task, representing the robot's initial search for the target object. After that, the robot should proceed with **[mobile]** and **[manipulation]** actions as appropriate. Additional **[observation]** states may appear in the middle of the task if the robot needs to confirm or locate new objects or scenes.
    - For the **[observation]** state, use the special tokens **[goal]**, **[current object or scene]**, and **[search result]** to describe the search process. If the target object is found in the first frame, no additional frames are necessary for this state.
    - **The [observation] state must conclude with `search result: yes`**, indicating that the target object has been successfully located before moving on to the next state. The robot cannot transition to the next steps until this condition is met.
    - **The analysis and judgment of the target object must be reasonable**, meaning that the robot should accurately identify the target object based on the given task description. Ensure that the target object is relevant to the primary task and that the robot correctly identifies it during the observation process.
    - Pay attention to the logical dependencies between the different states:
        - Generally, the robot must first identify a target object using **[observation]** (e.g., "[goal]: [find] the target object").
        - After identifying the object, the robot should move towards it using **[mobile]** (e.g., "Move towards the object").
        - Once the robot is in position, it can perform **[manipulation]** actions (e.g., "[left] holds the object, [right] tightens the screws").
        - Ensure that the steps follow this natural progression: **[observation]** → **[mobile]** → **[manipulation]**.
    - **Left and Right Hand Determination**: In some cases, it might be challenging to distinguish between the left and right hands based solely on the ego-centric camera view. Therefore, when determining whether an action is performed by the `[left]` or `[right]` hand:  
        - Combine **global context** (previous steps, object location, and known hand positions) with ego-centric camera input.         
        - Consider the **spatial relationship** of the hands in the frame. For example, if a hand extends from the left side of the camera's view, it is more likely to be the `[left]` hand.         
        - If only one hand is visible in the frame, use clues such as the direction of movement or the relative position of objects to make an informed judgment about which hand is performing the task.

**Task Description**: {desc}
---


**Example Output Format (This is an example with a total frame length of 60, and the specific situation depends on the actual frame length.)**:
```json
{{
  "task_summary": "Assembling an office desk.",
  "steps": [
    {{
      "state": "[observation]",
      "action_description": "[goal]: [find] the legs of the desk. [current object or scene]: A table is visible. [search result]: no.",
      "start_frame": 0,
      "end_frame": 1
    }},
    {{
      "state": "[observation]",
      "action_description": "[goal]: [find] the legs of the desk. [current object or scene]: A chair and some tools are visible. [search result]: no.",
      "start_frame": 2,
      "end_frame": 3
    }},
    {{
      "state": "[observation]",
      "action_description": "[goal]: [find] the legs of the desk. [current object or scene]: The legs of the desk are now in view. [search result]: yes.",
      "start_frame": 4,
      "end_frame": 4
    }},
    {{
      "state": "[mobile]",
      "action_description": "Move towards the desk.",
      "start_frame": 5,
      "end_frame": 8
    }},
    {{
      "state": "[manipulation]",
      "action_description": "[left] holds the leg in place while [right] uses a screwdriver to attach the leg to the tabletop.",
      "start_frame": 9,
      "end_frame": 15
    }},
    {{
      "state": "[observation]",
      "action_description": "[goal]: [find] the support beam for the desk. [current object or scene]: The desk's surface and legs are visible. [search result]: no.",
      "start_frame": 16,
      "end_frame": 17
    }},
    {{
      "state": "[observation]",
      "action_description": "[goal]: [find] the support beam for the desk. [current object or scene]: The support beam is now in view. [search result]: yes.",
      "start_frame": 18,
      "end_frame": 18
    }},
    {{
      "state": "[mobile]",
      "action_description": "Move to the other side of the desk.",
      "start_frame": 19,
      "end_frame": 21
    }},
    {{
      "state": "[manipulation]",
      "action_description": "[left] holds one end of the support beam while [right] aligns the other end with the leg.",
      "start_frame": 22,
      "end_frame": 24
    }},
    {{
      "state": "[manipulation]",
      "action_description": "[right] uses a screwdriver to secure the right end of the beam to the leg, [left] holds the beam steady.",
      "start_frame": 25,
      "end_frame": 26
    }},
    {{
      "state": "[manipulation]",
      "action_description": "[left] adjusts the alignment of the left end of the beam, [right] prepares to tighten the screws.",
      "start_frame": 27,
      "end_frame": 28
    }},
    {{
      "state": "[manipulation]",
      "action_description": "[both] tighten the screws on both ends of the beam to secure it in place.",
      "start_frame": 29,
      "end_frame": 29
    }},
    {{
      "state": "[observation]",
      "action_description": "[goal]: [find] the drawer for the desk. [current object or scene]: The desk's surface and legs are visible. [search result]: no.",
      "start_frame": 30,
      "end_frame": 31
    }},
    {{
      "state": "[observation]",
      "action_description": "[goal]: [find] the drawer for the desk. [current object or scene]: The drawer is now in view. [search result]: yes.",
      "start_frame": 32,
      "end_frame": 32
    }},
    {{
      "state": "[mobile]",
      "action_description": "Move towards the drawer.",
      "start_frame": 33,
      "end_frame": 35
    }},
    {{
      "state": "[manipulation]",
      "action_description": "[left] holds the drawer while [right] aligns it with the desk.",
      "start_frame": 36,
      "end_frame": 38
    }},
    {{
      "state": "[manipulation]",
      "action_description": "[right] uses a screwdriver to secure the drawer to the desk, [left] holds the drawer steady.",
      "start_frame": 39,
      "end_frame": 41
    }},
    {{
      "state": "[manipulation]",
      "action_description": "[left] adjusts the alignment of the drawer, [right] prepares to tighten the screws.",
      "start_frame": 42,
      "end_frame": 43
    }},
    {{
      "state": "[manipulation]",
      "action_description": "[both] tighten the screws on the drawer to secure it in place.",
      "start_frame": 44,
      "end_frame": 44
    }},
    {{
      "state": "[observation]",
      "action_description": "[goal]: [find] the keyboard tray for the desk. [current object or scene]: The desk's surface and drawer are visible. [search result]: no.",
      "start_frame": 45,
      "end_frame": 46
    }},
    {{
      "state": "[observation]",
      "action_description": "[goal]: [find] the keyboard tray for the desk. [current object or scene]: The keyboard tray is now in view. [search result]: yes.",
      "start_frame": 47,
      "end_frame": 47
    }},
    {{
      "state": "[mobile]",
      "action_description": "Move towards the keyboard tray.",
      "start_frame": 48,
      "end_frame": 50
    }},
    {{
      "state": "[manipulation]",
      "action_description": "[left] holds the keyboard tray while [right] aligns it with the desk.",
      "start_frame": 51,
      "end_frame": 53
    }},
    {{
      "state": "[manipulation]",
      "action_description": "[right] uses a screwdriver to secure the keyboard tray to the desk, [left] holds the tray steady.",
      "start_frame": 54,
      "end_frame": 56
    }},
    {{
      "state": "[manipulation]",
      "action_description": "[left] adjusts the alignment of the keyboard tray, [right] prepares to tighten the screws.",
      "start_frame": 57,
      "end_frame": 58
    }},
    {{
      "state": "[manipulation]",
      "action_description": "[both] tighten the screws on the keyboard tray to secure it in place.",
      "start_frame": 59,
      "end_frame": 59
    }}
  ]
}}
'''"""
    return prompts