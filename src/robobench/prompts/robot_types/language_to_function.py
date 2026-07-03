promptForFunctionTemplate = '''
### Task Description
You need to process a set of **natural language step descriptions** for a robotic manipulation task and extract **generalized functions**. All generated functions must be selected from a predefined list of functions, ensuring that the function names and parameters match the definitions in the list. If the input action cannot be matched to any function in the list, use `special_action()` and generate a suitable action.

---

### Rules & Requirements
1. **Select from Predefined Functions**:
   - All generated functions must be selected from the following categories:
     - **Object Manipulation**
     - **Movement and Navigation**
     - **Opening and Closing**
     - **Special Operations**
     - **no_ops**
     - **special_action()**
   - If the input action cannot be matched to any function in the list, use `special_action()`.
   - If the input natural language action description indicates a left-handed action and a right-handed action respectively, you need to match a function for each of the left-handed and right-handed actions. Note that the description with "[both]" tag means both hands operate simultaneously and it is also suitable to match function to each of them.

2. **Function Selection Rules**:
   - Select the most appropriate function based on the semantics of the action.
   - Refer to the explanations provided for each function to ensure correct selection.

3. **Output Format**:
   - The output must be in a single-line JSON array format.
   - Each function should be enclosed in quotes and separated by commas.
   - The order of functions should match the input.
   - The number of output functions must be strictly equal to the number of input **natural language step descriptions**.
   - If `special_action()` is used, include the generated description in the JSON.

---

### Predefined Functions List with Explanations

#### 1. Object Manipulation
Functions related to manipulating objects using the robotic arm and gripper:

pick_up(object): Grasp the object firmly with the robotic gripper and lift it from its current position. This implies that the object is fully secured and moved upward (e.g., picking up a cup from a table). [The object is lifted.]
grasp(object): Close the gripper around the object to hold it securely, but without lifting or relocating it (e.g., gripping a chair to push/pull it, or holding a door handle to open/close it). [The object is held but not lifted.]

place(object, target_object): Use the robotic arm and gripper to place the object at the location of or onto the target_object.
pour(object, target_object): Use the robotic arm and gripper to pour liquid in the object (a container e.g., cup,bottle,scoop) into the target_object (another container e.g., cup,bottle,scoop). The object may also be the specific liquid (e.g., water, liquid).
fold(object, direction): Fold the object in a specific direction using the robotic arm and gripper. 
pull(object, target_object): Pull the object toward or from the target_object using the robotic arm and gripper. The "target_object" can also represent a direction (e.g., "left," "right," "forward," "backward").
turn(object, direction): Rotate the object in a specific direction using the robotic arm and gripper. 
wipe(object, target_object): Clean or wipe the target_object using the object (e.g., a cloth) held by the robotic gripper. 
press(object): Apply pressure to the object (e.g., a button or switch) using the robotic gripper.
push(object, target_object): Push the object toward or onto the target_object using the robotic arm and gripper. The "target_object" can also represent a direction (e.g., "left," "right," "forward," "backward"). 
unfold(object, direction): Unfold the object in a specific direction using the robotic arm and gripper. 
scoop(object, target_object): Lift and carry the target_object using a scooping motion with the object held by the robotic arm and gripper.
unplug(object, target_object): Remove the object from the target_object (e.g., unplugging a charger) using the robotic arm and gripper.
stir(object, target_object): Stir the contents of the target_object(e.g.,water,liquid) using the object (e.g., a spoon,rob) held by the robotic gripper.
write(object, target_object): Robot gripper holds a pen or marker write object (e.g., a sentence, "abc" or "hi") on the target_object (e.g., a whiteboard) .

insert(object, target_object): Insert the object into the target_object (e.g., plugging in a charger) using the robotic arm and gripper.
pullout(object, target_object): Remove the object from the target_object (e.g., unplugging a charger) using the robotic arm and gripper.

#### 2. Movement and Navigation
move_to(object, target_object): Move the object held by the robotic arm and gripper to the location of the target_object. 
move_away_from(object, target_object): Move the object held by the robotic arm and gripper away from the target_object.

#### 3. Opening and Closing
Functions related to opening or closing objects:
close(object): Close the object (e.g., a lid, door, or container) using the robotic gripper.
open(object): Open the object (e.g., a lid, door, or container) using the robotic gripper.

#### 4. Special Operations
turn_off(object): Turn off the object (e.g., a device or appliance) using the robotic arm and gripper.
turn_on(object): Turn on the object (e.g., a device or appliance) using the robotic arm and gripper.

#### 5. no_ops
no_ops: No visible action, stay still or keep current state.

---

### Example List

#### Input
pick up the cup  
move towards the shelf with the block  
push the ball  
place the book on the desk  
hover over the table  
[left] approaches the glasses case while [right] approaches the glasses case lid.
[left] holds the glasses case steady. [right] moves the lid towards the case.
[both] close the laptop lid.

#### Output
```json
[
  "pick_up(object)",
  "move_to(object, target_object)",
  "push(object, target_object)",
  "place(object, target_object)",
  "no_ops",
  "left:move_to(object, target_object), right:move_to(object, target_object)",
  "left:no_ops, right:move_to(object, target_object)",
  "left:close(object), right:close(object)",
]

### **Task**
Convert the following natural language step descriptions into structured generalized functions:
{}
'''


promptForFunctionCall = '''
Task Description: Your task is to process a set of natural language action descriptions and generate function calls based on provided function templates. The function calls must accurately reflect the semantics of the input actions and use the correct objects or targets from the descriptions. 

*** Rules & Requirements ***
Input:
    - A list of natural language action descriptions (e.g., "reach for the juice bottle") and a list of function templates (e.g., "move_to(object)").
    - The number of function templates matches the number of action descriptions, and they correspond in order.
Output:
    - A list of function calls that match the input actions and templates (e.g., "move_to(juice_bottle)").
    - The number of function calls must match the number of input descriptions.
    - If an object or target is composed of multiple words, separate them with underscores (_).
    
Function Call Rules:
    - Replace the placeholder (e.g., object, target_object, goal, current_scene_or_object) in the function template with the appropriate object or target from the natural language description.
    - Ensure the function call accurately reflects the action described in the input.
Output Format:
    - The output must be a list of function calls in the correct format.
    
Function Templates contain thress types of "observation","mobile" and "manipulation". The specific templates and explanations as follows:
1. Observation
observation(goal, current_scene_or_object, result): Goal represents the current target to be found. The current_scene_or_object represents the currently visible scene or object. The result represents the outcome of the search (e.g., yes,no or part).

2. Mobile
mobile(target_object): This function represents the movement of the robot chassis. The "target_object" represents the target scene or object.

3. Manipulation (The action of robotic arms or grippers)
3.1 Object Manipulation
Functions related to manipulating objects using the robotic arm and gripper:
pick_up(object): Grasp and lift the object using the robotic gripper.
place(object, target_object): Use the robotic arm and gripper to place the object at the location of or onto the target_object. If target_object is "none", it means no certain target and you should only place the object.
pour(object, target_object): Use the robotic arm and gripper to pour liquid in the object (a container e.g., cup,bottle,scoop) into the target_object (another container e.g., cup,bottle,scoop). The object may also be the specific liquid (e.g., water, liquid).
fold(object, direction): Fold the object in a specific direction using the robotic arm and gripper. If direction is "none", it means no certain direction.
pull(object, target_object): Pull the object toward or from the target_object using the robotic arm and gripper. The "target_object" can also represent a direction (e.g., "left," "right," "forward," "backward").
turn(object, direction): Rotate the object in a specific direction using the robotic arm and gripper. If direction is "none", it means no certain direction.
wipe(object, target_object): Clean or wipe the target_object using the object (e.g., a cloth) held by the robotic gripper. If object is "none", it means no certain used object.
press(object): Apply pressure to the object (e.g., a button or switch) using the robotic gripper.
push(object, target_object): Push the object toward or onto the target_object using the robotic arm and gripper. The "target_object" can also represent a direction (e.g., "left," "right," "forward," "backward"). If target_object is none, it means no certain target or direction.
plug(object, target_object): Insert the object into the target_object (e.g., plugging in a charger) using the robotic arm and gripper.
unfold(object, direction): Unfold the object in a specific direction using the robotic arm and gripper. If direction is "none", it means no certain direction.
scoop(object, target_object): Lift and carry the target_object using a scooping motion with the object held by the robotic arm and gripper.
unplug(object, target_object): Remove the object from the target_object (e.g., unplugging a charger) using the robotic arm and gripper.
stir(object, target_object): Stir the contents of the target_object(e.g.,water,liquid) using the object (e.g., a spoon,rob) held by the robotic gripper.
write(object, target_object): Robot gripper holds a pen or marker write object (e.g., a sentence, "abc" or "hi") on the target_object (e.g., a whiteboard) .
3.2 Movement and Navigation
Functions related to moving the robotic arm and gripper (not the base):
move_to(object, target_object): Move the object held by the robotic arm and gripper to the location of the target_object. If "object" is none, it means that the robotic arm and gripper move to the target_object with nothing.
move_away_from(object, target_object): Move the object held by the robotic arm and gripper away from the target_object. If "object" is none, it means that the robotic arm and gripper to the location of target_object with nothing.
3.3 Open and Close
Functions related to opening or closing objects:
close(object): Close the object (e.g., a lid, door, or container) using the robotic gripper.
open(object): Open the object (e.g., a lid, door, or container) using the robotic gripper.
3.4 Special Operations
Functions for specific or advanced tasks:
turn_off(object): Turn off the object (e.g., a device or appliance) using the robotic arm and gripper.
turn_on(object): Turn on the object (e.g., a device or appliance) using the robotic arm and gripper.
3.5 no_ops
no_ops: No visible action, stay still or keep current state.
    
    
***Examples***
### Input:
## Natural Language Descriptions:
    "[right] places the oil bottle back on the counter"
    "[right] reaches for the bowl of shrimp"
    "[goal]: [find] spatula [current object or scene]: pan with shrimp and spatula are visible [search result]: yes"
    "reach for the juice bottle",
    "grasp the juice bottle",
    "lift the juice bottle",
    "move the juice bottle towards the blue sticky note",
    "place the juice bottle on the blue sticky note",
    "release the juice bottle"

## Function Templates:
    "left:no_ops, right:place(object, target_object)"
    "left:no_ops, right:move_to(object, target_object)"
    "observation(goal, current_scene_or_object, result)"
    "move_to(object)",
    "pick_up(object)",
    "pick_up(object)",
    "move_to(object, target_object)",
    "place(object, target_object)",
    "place(object, target_object)"

### Output:
```json
[
    "left:no_ops, right:place(oil_bottle, counter)",
    "left:no_ops, right:move_to(none, bowl_with_shrimp)",
    "observation(spatula, ['pan_with_shrip_and_spatula'], yes)",
    "move_to(juice_bottle)",
    "pick_up(juice_bottle)",
    "pick_up(juice_bottle)",
    "move_to(juice_bottle, blue_sticky_note)",
    "place(juice_bottle, blue_sticky_note)",
    "place(juice_bottle, None)"
]

### **Task**
Convert the following natural language step descriptions and function templates to the special function calls.
Language Descriptions: {}
Function Templates: {}
'''