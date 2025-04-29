import os
import torch
import torch.nn as nn
import numpy as np
import random
import hou
import time

# ===========================
# Configurations
# ===========================
GEO_NODE = "/obj/Study_B_Environment_02"
BEST_MODEL_PATH = r"D:\Users\YUTO\Documents\pg20\year 5\Model\Machine Learning\Reinforcement Learning Models\250401_Ray_Checker_PPO_01.pth"

# ===========================
# PPO Neural Network
# ===========================
class PPOActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, output_dim)  # Policy output
        self.critic = nn.Linear(64, 1)  # Value function output

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)  # Policy distribution
        value = self.critic(x)  # Value estimation
        return action_probs, value

# ===========================
# Houdini Utility Functions
# ===========================
def get_houdini_attr(node_path, attr_name):
    node = hou.node(node_path)
    if node:
        return node.geometry().attribValue(attr_name)
    return None

def set_houdini_param(node_path, param_name, value):
    node = hou.node(node_path)
    if node:
        node.parm(param_name).set(int(value))  # Ensure value is a standard Python int

def cook_houdini_node(node_path):
    node = hou.node(node_path)
    if node:
        node.cook(force=True)

# ===========================
# Inference Function
# ===========================
def run_inference():
    # Load trained model
    model = PPOActorCritic(input_dim=3, output_dim=2)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    model.eval()

    hou.setFrame(1)  # Reset Houdini frame at episode start
    done = False
    state = get_houdini_attr(f"{GEO_NODE}/State_01", "state")
    actions_taken = []
    
    start_time = time.time()

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs, _ = model(state_tensor)
        action_probs = action_probs.squeeze(0).detach().numpy()
        
        # Sample action from probability distribution
        action = np.random.choice([0, 1], p=action_probs)
        actions_taken.append(action)

        # Apply action in Houdini
        set_houdini_param(f"{GEO_NODE}/Action_01", "value1v1", action)

        # Cook Score node to get reward
        cook_houdini_node(f"{GEO_NODE}/Score_01")
        reward = get_houdini_attr(f"{GEO_NODE}/Score_01", "reward")

        # Update reward in Houdini
        set_houdini_param(f"{GEO_NODE}/Reward_Recorder_01", "value1v1", reward)

        # Check for termination
        done = get_houdini_attr(f"{GEO_NODE}/Stop_01", "stop") == 1
        if not done:
            hou.setFrame(hou.frame() + 1)  # Increment frame
            state = get_houdini_attr(f"{GEO_NODE}/State_01", "state")  # Get new state

    print(f"Inference Completed: Last Step Reward = {reward}")
    print(f"Actions Taken: {actions_taken}")
    
    end_time = time.time()
    print(f"time: {end_time - start_time:.2f} seconds")

# Run inference
run_inference()
