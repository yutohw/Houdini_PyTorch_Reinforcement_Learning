import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
import hou

# Geometry node reference
geometry_node = "/obj/Massing_Solar_Environment_12"

# Q-Network definition (must match training architecture)
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_size)
        )
    
    def forward(self, x):
        return self.fc(x)

# Parameters
input_size = 4
output_size = 2
epsilon = 0.1
model_path = "D:/Users/YUTO/Documents/pg20/year 5/Model/Machine Learning/Reinforcement Learning Models/250311_Ray_Checker_DQN_01.pth"

# Houdini helper functions
def get_houdini_attr(node_path, attr_name):
    geo = hou.node(node_path).geometry()
    return geo.attribValue(attr_name)

def get_houdini_state():
    state = get_houdini_attr(f"{geometry_node}/State_01", "state")
    return np.array(state, dtype=np.float32).flatten()

def get_reward():
    hou.node(f"{geometry_node}/Score_01").cook()
    return get_houdini_attr(f"{geometry_node}/Score_01", "reward")

def check_termination():
    return get_houdini_attr(f"{geometry_node}/Stop_01", "stop") == 1

def apply_action(action):
    action_node = hou.node(f"{geometry_node}/Action_01")
    action_node.parm("value1v1").set(action)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_network = QNetwork(input_size, output_size).to(device)
q_network.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
q_network.eval()

# Inference
start_time = time.time()
hou.setFrame(1)
state = torch.tensor(get_houdini_state(), dtype=torch.float32, device=device)
action_list = []
episode_reward = None

while True:
    # Epsilon-greedy policy
    if random.random() < epsilon:
        action = random.randint(0, 1)
    else:
        with torch.no_grad():
            action = torch.argmax(q_network(state)).item()
    
    apply_action(action)
    action_list.append(action)
    
    # Update state and check for termination
    reward = get_reward()
    episode_reward = reward
    next_state = torch.tensor(get_houdini_state(), dtype=torch.float32, device=device)
    done = check_termination()
    
    if done:
        break
    
    state = next_state
    hou.setFrame(hou.frame() + 1)

end_time = time.time()

print(f"Inference complete.")
print(f"Final reward: {episode_reward}")
print(f"Actions taken: {action_list}")
print(f"Time taken: {end_time - start_time:.2f} seconds")
