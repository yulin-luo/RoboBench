promptForImplicitInstructionTemplate = '''Task:
 Generate implicit task instructions for embodied robots based on a provided explicit task and (if available) an image. The goal is to produce natural, everyday expressions that imply the need for the target task without explicitly mentioning the target object or directly stating the task.
Key Requirements:
1. Target Entity Focus:
  - Identify the unique characteristic of the target entity from the explicit task and ensure that the implicit instruction reflects this characteristic.
  - The implicit instruction must have a direct association with the target entity and task, not abstract references.
2. Everyday Scenarios and Language:
  - Use casual, real-life scenarios to imply the need for the task.
  - Avoid technical terms or abstract expressions. The instruction should feel like a natural request from a human in daily life.
3. Image Integration (if provided):
  - Analyze the provided image directly to extract relevant information about the scene.
  - The image may contain multiple objects, including distracting household items unrelated to the task. Focus on the target object from the explicit task while ensuring the implicit instruction remains relevant and natural.
  - Do not rely on textual descriptions of the image; all visual information must come from analyzing the provided image itself.
4. No Direct Mentions:
  - Do not directly mention the target object or the explicit task.
  - The need for the task should be implied through observations, feelings, or everyday needs (e.g., "I feel parched," instead of "pour water into the cup").
5. Output Format:
  - Provide 5 implicit task instruction suggestions for each input.
  - The output must be in list format, with each instruction as a separate list item to ensure consistency and ease of post-processing.
Examples:
- Target object: pouring liquid from a bottle into a cup
Provided Image: A table with a bottle, an empty cup, and other unrelated items such as soap, a photo frame, and a gift bag.
Instruction Output:
 [
 "Looking at the empty cup on the table makes me realize how thirsty I am. Could you help me with that?",
...
 ]
- Target object: watering a plant
Provided Image: A windowsill with several potted plants, one with drooping leaves.
Instruction Output:
 [
 "This plant’s leaves look a bit droopy today. Could you help bring it back to life?",
...
 ]
- Target object: selecting a Batman toy
Provided Image: A shelf filled with various superhero toys, including a Batman figure.
Instruction Output:
 [
 "I’m a big fan of DC series, please help me choose a suitable toy.",
...
 ]
Input Format:
- Target object: {task_summary}
- Provided Image: {image_path}

Output Format:
- Instruction Output:
 [
 "Instruction 1",
 "Instruction 2",
 "Instruction 3",
 "Instruction 4",
 "Instruction 5"
 ]
'''