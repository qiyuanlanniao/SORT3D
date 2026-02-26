from .examples import example_object_list

def get_caption_prompt(environment_name, grid_map_shape, robot_coords, objects):
    prompt = f'''You are an intelligent agent, tasked with generating Python code to move a wheeled robot around a 2D environment from natural language instructions.

The environment is {environment_name}. The environment is split into a discrete {grid_map_shape[0]}x{grid_map_shape[1]} grid. The robot starts at {tuple(robot_coords[0].tolist())}.

You are given a list of objects in the environment. Each object is characterized by an identifying number id, a name, a caption describing the object, (x, y, z) coordinates of its center, and the size of it's largest face in the following format: [id, 'name', 'caption', x, y, z, size]

The following Python functions are implemented. Objects are passed into each function by its identifying number id:

def go_near(object):
	"""
	Go near object.
	"""
    
After the list of objects in the environment, you are given a natural language command asking the robot to navigate around the environment given from the visual perspective of an observer commanding the robot, and you are tasked with generating a python function from this prompt that only uses the above functions. Use context clues from the environment to understand some of the commands. Use chain-of-thought reasoning, and explain your reasoning as you parse the language into code. Write out your reasoning, then write out the output code. Here are two examples of the input and output formats:

Example 1:

Object List:
{example_object_list}

User Input:
Go to the coffee machine, first passing near all of the three chairs closest to it, in order from closest to your initial position to furthest.

Your Output:

Reasoning:
[Explain your reasoning here]

Code:
def go():
    go_near(6)
    go_near(11)
    go_near(7)
    go_near(0)

Example 2:

{example_object_list}

User Input:
There are six office workers on the table near the coffee machine, and they need their morning coffee. You can only hold one coffee cup at a time.

Your Output:

Reasoning:
[Explain your reasoning here]

Code:
def go():
	go_near(0)
	go_near(6)
	go_near(0)
	go_near(11)
	go_near(0)
	go_near(7)
	go_near(0)
	go_near(5)
	go_near(0)
	go_near(1)
	go_near(0)
	go_near(12)

Respond ONLY in the following format. Make sure you think step by step, and write out your reasoning, then the code as shown below.

Your Output:

Reasoning: 
[Thoroughly explain the reasoning behind your answer.]

Code:
def go():
    # Insert your output code here.

Given the following list of objects and the input language command, write a Python script that could be executed by the wheeled robot:

Object List:
{objects}
'''
    return prompt


def get_tool_caption_prompt(environment_name, grid_map_shape, robot_coords, objects, tool_descriptions):
    prompt = f'''You are an intelligent agent, tasked to move a wheeled robot around 3D environment from natural language instructions.

The environment is {environment_name}. The environment is split into a discrete {grid_map_shape[0]}x{grid_map_shape[1]} grid. The robot starts at {tuple(robot_coords[0].tolist())}.

You are given a list of objects in the environment. Each object is characterized by an identifying number id, a name, a caption describing the object, (x, y, z) coordinates of its center, and the size of it's largest face in the following format: [id, 'name', 'caption', x, y, z, size].

You have access to the following tools, you should use them *as much as possible* during intermediate steps to help you understand and tackle the query, do not attempt to figure out yourself if you can use a tool to help you. For the tools that take in object names, ONLY provide as arguments the names of objects that are found in the given list of objects. DO NOT provide the name of an object not found in the list.

Pay attention: Your ultimate goal is to call the `command_robot` tool to move the robot through tool call. You should always attempt to call `command_robot` at the end given best of your knowledge. You cannot ask the user for further information. You should only output content when you are directly instructed to say something. You should use the `notepad` to write down your reasoning and thoughts when you are handling the task, but do not output content directly.  Meanwhile, do not use `notepad` alone to solve the task, there are other tools too that you can use to help you solve the task.

{tool_descriptions}
    
You will be given a natural language command asking the robot to navigate around the environment given from the visual perspective of an observer commanding the robot. Think step by step, and write out your reasoning in `notepad`. At the very end, retrace your reasoning steps, and make sure that you are confident of the final answer. Make sure that the logic is sound. Check whether the final answer is correct, or whether you need to change it.
'''
    return prompt


def get_tool_caption_benchmark_prompt(environment_name, grid_map_shape, robot_coords, objects, tool_descriptions):
    prompt = f'''You are an intelligent agent, tasked to pick an object referred to using a natural language utterance in a 3D environment.

The environment is a/an {environment_name}. The environment is split into a discrete {grid_map_shape[0]}x{grid_map_shape[1]} grid.

You are given a list of objects in the environment. Each object is characterized by an identifying number id, a name, a caption describing the object, (x, y, z) coordinates of its center, and the size of it's largest face in the following format: [id, 'name', 'caption', x, y, z, size].

You have access to the following tools, you should use them *as much as possible* during intermediate steps to help you understand and tackle the query, do not attempt to figure out yourself if you can use a tool to help you. For the tools that take in object names, ONLY provide as arguments the names of objects that are found in the given list of objects. DO NOT provide the name of an object not found in the list.

Pay attention: Your ultimate goal is to call the `pick_object` tool to move the robot through tool call. You should always attempt to call `pick_object` at the end with the best of your knowledge. You cannot ask the user for further information. You should only output content when you are directly instructed to say something. You should use the `notepad` to write down your reasoning and thoughts when you are handling the task, but do not output content directly. Meanwhile, do not use `notepad` alone to solve the task, there are other tools too that you can use to help you solve the task. There is exactly ONE answer, so if you receive multiple answers, consider other constraints; if you get no answers, loosen constraints.

{tool_descriptions}
    
You will be given a natural language command asking the robot to navigate around the environment given from the visual perspective of an observer commanding the robot. Think step by step, and write out your reasoning in `notepad`. At the very end, retrace your reasoning steps, and make sure that you are confident of the final answer. Make sure that the logic is sound. Check whether the final answer is correct, or whether you need to change it.
'''
    return prompt


def get_sgnav_hcot_prompt(environment_name, grid_map_shape, robot_coords, scene_hierarchy, tool_descriptions):
    prompt = f'''You are an advanced robot spatial reasoner (SG-Nav + SpatialNav agent). 
The environment is {environment_name}. Start: {tuple(robot_coords[0].tolist())}.

You perceive the world through two primary systems:
1. **Scene Graph (Memory)**: Topological knowledge of rooms and groups.
2. **3D Octant Compass (Sight)**: Immediate 3D relative positions of objects.

### 3D Octant Compass Definition (Like a 2x2 Rubik's Cube):
The space around you is divided into 8 sectors based on your local coordinate system:
- **FRONT_LEFT_TOP**: Objects in front, to your left, and above your camera height.
- **FRONT_RIGHT_BOTTOM**: Objects in front, to your right, and below your camera height (likely on floor).
- ... and so on for ALL 8 combinations of [FRONT/BACK], [LEFT/RIGHT], [TOP/BOTTOM].

### Hierarchical Reasoning (H-CoT):
1. **Global Search**: Find the target Room/Group in the Scene Graph memory.
2. **Local Orientation**: Look at the 3D Octant Compass to see if the target is in your immediate sight.
3. **Height/Depth Inference**: Use TOP/BOTTOM tags to infer placement. (e.g., if a "mug" is TOP and "table" is TOP in the same sector, the mug is likely on the table).
4. **Action**: Generate code.

Scene Hierarchy:
{scene_hierarchy}

Available Tools:
{tool_descriptions}

Final output: call `command_robot` with `go_near(id)` or `go_between(id1, id2)`. Use `notepad` for 3D reasoning.
'''
    return prompt


def get_vlm_relationship_verification_prompt(obj_a_name, obj_b_name):
    """
    用于 VLM 短程边剪枝的提示词。
    要求 VLM 观察图像并判断两个检测到的物体是否真的存在空间关联。
    """
    prompt = (
        f"You are a spatial relationship validator. Look at the image provided. "
        f"There are two objects detected: a '{obj_a_name}' and a '{obj_b_name}'. "
        f"Based on the visual evidence, do they appear to be part of the same functional group "
        f"or are they physically next to each other in a way that makes sense? "
        f"(e.g., a chair next to a table, a monitor on a desk, or a cabinet next to another cabinet). "
        f"Answer ONLY 'Yes' if they are related, or 'No' if they are unrelated or separated by a wall/large gap."
    )
    return prompt