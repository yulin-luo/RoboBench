PLANNING_TEMPLATE ="""You will analyze a video (represented by image frames) of a robotic arm performing a specific task, where the task is described as: ``{desc}``. Note that the referenced task summary might not accurate or complete. Your task is to identify the primary task during the video with the help of the referenced descrition, summarize the task and rewrite the description, extract the necessary steps to complete it, and specify the frame range for each step. Follow these instructions:

1. **Task Identification**: First, identify the main task the robotic arm is performing. This task could be a clear goal or a series of related activities (e.g., assembling furniture, repairing equipment, preparing food, etc.). Briefly describe the primary task in one sentence.

2. **Step Extraction**: Once the task is identified, extract the key steps required to complete it, ensuring that each step is clearly described and logically ordered. Each step may include:
    - Specific actions (e.g., tightening screws, stirring mixtures, pressing buttons, etc.)
    - Frame window: Specify the start and end frame for each step (from `0` to `{maxframeid}`, since the video has {cnt} frames).

3. **Frame Range Constraints**:
    - **No Overlapping Frames**: Ensure that the frame ranges of each step do not overlap with each other. Each frame should be assigned to exactly one step.
    - **Full Frame Coverage**: Ensure that all {cnt} frames (from 0 to {maxframeid}) are included in the steps. No frames should be missed or duplicated.
    - **Must start from frame 0**
    
4. **Notes**:  
     - Please annotate as finely as possible and try not to have more than ten frames of the same thing being done (unless it is difficult to distinguish).
     - A step has only one verb (unless two or more actions are strictly performed simultaneously)
     - Please ensure the accuracy of labeling and image matching.
     - For example, if the gripper changes from an open state to a closed state in a sequence of frames, the action is called 'pick'; On the contrary, if it changes from a closed state to an open state, the action is called 'release'.
     
5. **Failure Identification**: If the robotic arm attempts an action but does not succeed, clearly indicate this in the step description. For example, if the robotic arm tries to pick up a block but fails, the step description should be something like 'Attempt to pick up a block but fails'.

6. **Output Format**: Provide the task description and steps in two parts, formatted as JSON:
    - **Task Summary**: A string summarizing the primary task in the video without mentioning the subjects - the robotic arm.
    - **Steps**: An array where each element represents a step, containing:
        - `step_description`: A concise description of the step which the action being performed in the format of verb phrases without mentioning the subjects - the robotic arm (e.g., "Add syrup in the glass").
        - `start_frame`: The start frame of the step (from `0` to `{cnt-1}`).
        - `end_frame`: The end frame of the step (from `0` to `{cnt-1}`).

**Task Description**: {desc}
---

**Example Output Format 1: (This is an example with a total frame length of 30, and the specific situation depends on the actual frame length.)**
```json
{{
  "task_summary": "Assembling an office desk.",
  "steps": [
    {{
      "step_description": "Remove all components and screws from the package.",
      "start_frame": 0,
      "end_frame": 4
    }},
    {{
      "step_description": "Use a screwdriver to attach the legs to the tabletop.",
      "start_frame": 5,
      "end_frame": 14
    }},
    {{
      "step_description": "Install the leg pads at the bottom.",
      "start_frame": 15,
      "end_frame": 19
    }},
    {{
      "step_description": "Fix the support beam between the legs with screws.",
      "start_frame": 20,
      "end_frame": 25
    }},
    {{
      "step_description": "Ensure all screws are tight and the desk is stable.",
      "start_frame": 25,
      "end_frame": 29
    }}
  ]
}}
'''

**Example Output Format 2: (This is an example with a total frame length of 50, and the specific situation depends on the actual frame length.)**
```json
{{
  "task_summary": "Painting a room.",
  "steps": [
    {{
      "step_description": "Remove all furniture.",
      "start_frame": 0,
      "end_frame": 2
    }},
    {{
      "step_description": "Cover the floor with drop cloths.",
      "start_frame": 3,
      "end_frame": 4
    }},
    {{
      "step_description": "Clean the walls.",
      "start_frame": 5,
      "end_frame": 9
    }},
    {{
      "step_description": "Apply primer to the walls.",
      "start_frame": 10,
      "end_frame": 19
    }},
    {{
      "step_description": "Paint the first coat of the chosen color.",
      "start_frame": 20,
      "end_frame": 29
    }},
    {{
      "step_description": "Allow the first coat to dry.",
      "start_frame": 30,
      "end_frame": 34
    }},
    {{
      "step_description": "Apply the second coat of paint.",
      "start_frame": 35,
      "end_frame": 44
    }},
    {{
      "step_description": "Allow the second coat to dry.",
      "start_frame": 45,
      "end_frame": 49
    }}
  ]
}}
"""
