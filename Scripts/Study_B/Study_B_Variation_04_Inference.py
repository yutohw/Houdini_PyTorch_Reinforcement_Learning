import os
import torch
import numpy as np
import hou
import time

# ===========================
# Configurations for Easy Updates
# ===========================
GEO_NODE = "/obj/Study_B_Environment_01"
BEST_MODEL_PATH = r"D:\Users\YUTO\Documents\pg20\year 5\Model\Machine Learning\Reinforcement Learning Models\250313_Ray_Checker_PPO_20.pth"

# ===========================
# PPO Neural Network (Same Architecture as Training)
# ===========================
class PPOActorCritic(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOActorCritic, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.actor = torch.nn.Linear(64, output_dim)  # Policy output
        self.critic = torch.nn.Linear(64, 1)  # Value function output

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)  # Policy distribution
        value = self.critic(x)  # Value estimation
        return action_probs, value

# ===========================
# Inference Utility Functions
# ===========================
def get_houdini_attr(node_path, attr_name):
    node = hou.node(node_path)
    if node:
        return node.geometry().attribValue(attr_name)
    return None

def set_houdini_param(node_path, param_name, value):
    node = hou.node(node_path)
    if node:
        node.parm(param_name).set(value)

def cook_houdini_node(node_path):
    node = hou.node(node_path)
    if node:
        node.cook(force=True)

# ===========================
# Load the Best Trained Model
# ===========================
model = PPOActorCritic(input_dim=3, output_dim=2)
model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
model.eval()

# ===========================
# Main Inference Loop
# ===========================

epsilon = 0.05  # 5% chance to explore randomly
start_time = time.time()

while True:
    # Get current state from Houdini
    state = get_houdini_attr(f"{GEO_NODE}/State_01", "state")
    state = np.array(state, dtype=np.float32).reshape(1, -1)
    state_tensor = torch.tensor(state, dtype=torch.float32)

    # Get action probabilities from the trained model
    with torch.no_grad():
        action_probs, _ = model(state_tensor)
        if np.random.rand() < epsilon:
            action = int(np.random.choice([0, 1]))  # Convert to native int
        else:
            action = torch.argmax(action_probs, dim=1).item()  # Already a native int

    # Apply the selected action in Houdini
    set_houdini_param(f"{GEO_NODE}/Action_01", "value1v1", action)

    # Cook the Score node to get the updated reward
    cook_houdini_node(f"{GEO_NODE}/Score_01")
    reward = get_houdini_attr(f"{GEO_NODE}/Score_01", "reward")

    # Print the reward (can be used for logging or debugging)
    print(f"Action: {action}, Reward: {reward}")

    # Check for termination (when stop attribute is 1)
    stop_signal = get_houdini_attr(f"{GEO_NODE}/Stop_01", "stop")
    if stop_signal == 1:
        print("Inference complete. Stopping execution.")
        break

    # Increment Houdini frame
    current_frame = hou.frame()
    hou.setFrame(current_frame + 1)

end_time = time.time()
print(f"time: {end_time - start_time:.2f} seconds")
