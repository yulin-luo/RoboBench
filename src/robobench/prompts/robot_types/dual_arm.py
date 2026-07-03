
prompts = """
You will analyze a video (represented by image frames) of a dual-arm robotic system performing a specific task, where the task is described as: ``{}``. Note that the referenced task summary might not be accurate or complete. Your task is to identify the primary task during the video with the help of the referenced description, summarize the task and rewrite the description, extract the necessary steps to complete it, and specify the frame range for each step. Follow these instructions:

1. **Left-Right Hand Identification**: The video is recorded from a first-person view of the dual-arm robotic system. Your first task is to accurately identify which hand is the left arm (`[left]`) and which hand is the right arm (`[right]`). This is crucial as you proceed with the task analysis. Use visual cues such as the relative position of the hands, their orientation, and any distinguishing features to determine which hand is on the left and which is on the right.

2. **Task Identification**: Once the left and right arms are correctly identified, determine the main task the robotic system is performing. This task could be a clear goal or a series of related activities (e.g., assembling furniture, repairing equipment, preparing food, etc.). Briefly describe the primary task in one sentence.

3. **Step Extraction**: After identifying the task and distinguishing the left and right arms, extract the key steps required to complete the task, ensuring that each step is clearly described and logically ordered. **Make sure the following criteria are met**:
    - The **first step** must always start with `start_frame` equal to `0`, and the **last step** must end with `end_frame` equal to `29`.
    - Every step must explicitly describe the actions of both the left arm (`[left]`) and right arm (`[right]`). If one arm is inactive during a step, mention its inactive state (e.g., "[left] holds the object steady while [right] tightens the screw"). If both arms are involved in the same action, use `[both]` to describe their joint activity, but still ensure to detail the left and right arm roles (e.g., "[both] lift the object, with [left] supporting the base and [right] holding the top").
    - **Frame window**: Specify the start and end frame for each step (from `0` to `29`, since the video has 30 frames).
    - **Handling Camera Obstruction**: If at any point the view is obstructed by one or both of the robotic arms, add a special marker `[block]` at the beginning of the step description. Then, based on the context before and after the obstruction, infer the likely actions taking place and provide the most accurate analysis possible. The `[block]` token should be used only when the camera is blocked, and after the token, describe the inferred task as usual (e.g., `[block] [left] holds the object steady, [right] tightens the bolt`).

    
4. **Frame Range Constraints**:
    - **No Overlapping Frames**: Ensure that the frame ranges of each step do not overlap with each other. Each frame should be assigned to exactly one step.
    - **Full Frame Coverage**: Ensure that all 30 frames (from 0 to 29) are included in the steps. No frames should be missed or duplicated.
    
5. **Notes**:  
     - Please annotate as finely as possible and try not to have more than ten frames of the same thing being done (unless it is difficult to distinguish).
     - A step has only one verb (unless two or more actions are strictly performed simultaneously)
     - Please ensure the accuracy of labeling and image matching.
     - For example, if the gripper changes from an open state to a closed state in a sequence of frames, the action is called 'pick'; On the contrary, if it changes from a closed state to an open state, the action is called 'release'.

6. **Output Format**: Provide the task description and steps in two parts, formatted as JSON:
    - **Task Summary**: A string summarizing the primary task in the video without mentioning the subjects - the robotic arms.
    - **Steps**: An array where each element represents a step, containing:
        - `step_description`: A concise description of the step, specifying the actions of `[left]`, `[right]`, or `[both]` arms (e.g., "[left] holds the frame, [right] tightens screws," "[both] lift the object"). Always describe `[left]` first and `[right]` second, when applicable.
        - `start_frame`: The start frame of the step (from `0` to `29`).
        - `end_frame`: The end frame of the step (from `0` to `29`).



**Task Description**: {}
---

**Example Output Format 1: (This is an example with a total frame length of 30, and the specific situation depends on the actual frame length.)**
```json
{{
  "task_summary": "Assembling an office desk.",
  "steps": [
    {{
      "step_description": "[left] removes all components from the package while [right] holds the package steady.",
      "start_frame": 0,
      "end_frame": 4
    }},
    {{
      "step_description": "[left] holds the leg in place while [right] uses a screwdriver to attach the legs to the tabletop.",
      "start_frame": 5,
      "end_frame": 14
    }},
    {{
      "step_description": "[left] installs the leg pads at the bottom while [right] holds the table steady.",
      "start_frame": 15,
      "end_frame": 19
    }},
    {{
      "step_description": "[both] fix the support beam between the legs with screws.",
      "start_frame": 20,
      "end_frame": 28
    }},
    {{
      "step_description": "[both] ensure all screws are tight and the desk is stable.",
      "start_frame": 29,
      "end_frame": 29
    }}
  ]
}}
```
"""