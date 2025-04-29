import hou
import torch
import torch.nn as nn
import time
import numpy as np
import random


class PPOActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)  
        self.fc2 = nn.Linear(8, 4)  
        self.actor = nn.Linear(4, output_dim)  
        self.critic = nn.Linear(4, 1)  

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value


def get_houdini_attr(node_path, attr_name):
    node = hou.node(node_path)
    geometry = node.geometry()
    return geometry.attribValue(attr_name)

def set_houdini_param(node_path, param_name, value):
    node = hou.node(node_path)
    if node:
        node.parm(param_name).set(value)

def cook_houdini_node(node_path):
    node = hou.node(node_path)
    if node:
        node.cook(force=True)

        
input_dim = 1
output_dim = 2
num_steps = 256
epsilon = 0.01  
geo_node = "/obj/Study_A_Rolling_Inference_Environment"
model_path = r"D:\Users\YUTO\Documents\pg20\year 5\Model\Machine Learning\Reinforcement Learning Models\250403_Study_A_Variation_09.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = PPOActorCritic(input_dim, output_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


hou.setFrame(1)
start_time = time.time()

state = get_houdini_attr(f"{geo_node}/State", "state")
actions_taken = []
rewards = []

for step in range(num_steps):
    state_np = np.array([state], dtype=np.float32)
    state_tensor = torch.tensor(state_np).unsqueeze(0).to(device)  # Shape: (1, 1)

    with torch.no_grad():
        action_probs, _ = model(state_tensor)

    if random.random() < epsilon:
        action = random.choice([0, 1])
    else:
        action = torch.argmax(action_probs, dim=-1).item()

    actions_taken.append(action)

    set_houdini_param(f"{geo_node}/Action", "value1v1", action)
    cook_houdini_node(f"{geo_node}/Score")

    reward = get_houdini_attr(f"{geo_node}/Score", "reward")
    rewards.append(reward)
    print(f"Step {step + 1}: Action = {action}, Reward = {reward}")

    state = get_houdini_attr(f"{geo_node}/State", "state")
    hou.setFrame(hou.frame() + 1)


final_reward = rewards[-1]
set_houdini_param(f"{geo_node}/Reward_Recorder", "value1v1", final_reward)
hou.node(f"{geo_node}/Result_Recorder").parm("execute").pressButton()

end_time = time.time()
print("Actions taken:", actions_taken)
print(f"Inference complete. Time taken: {end_time - start_time:.2f} seconds.")
