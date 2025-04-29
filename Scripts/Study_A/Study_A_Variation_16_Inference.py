import hou
import torch
import torch.nn as nn
import time
import random

# Inference environment node
geo_node = "/obj/Study_A_Targeting_Inference_Environment"

# Model path (same as best_model_path in training)
model_path = r"D:\Users\YUTO\Documents\pg20\year 5\Model\Machine Learning\Reinforcement Learning Models\250403_Study_A_Variation_16.pth"

# Parameters
input_dim = 512
output_dim = 256
num_steps = 5
epsilon = 0.01  # Small randomness for inference

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get Houdini attributes
def get_houdini_attr(node_path, attr_name):
    node = hou.node(node_path)
    geometry = node.geometry()
    return geometry.attribValue(attr_name)

# Set Houdini parameters
def set_houdini_attr(node_path, parm_name, value):
    node = hou.node(node_path)
    node.parm(parm_name).set(value)

# Define Q-Network (same as training)
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),  # Increased from 128 to 256
            nn.ReLU(),
            nn.Linear(512, 256),  # Added an extra hidden layer
            nn.ReLU(),
            nn.Linear(256, output_dim)  # Output size remains 100 (0-99 actions)
        )

    def forward(self, x):
        return self.fc(x)

# Load model
model = QNetwork().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Inference loop
hou.setFrame(1)
start_time = time.time()

# Initial state
state = get_houdini_attr(f"{geo_node}/State", "map")
state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)  # Shape: [1, 200]

actions = []
total_reward = 0

for step in range(num_steps):
    # Epsilon-greedy decision
    if random.random() < epsilon:
        action = random.randint(0, 99)
    else:
        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values, dim=1).item()

    actions.append(action)

    # Apply action in Houdini
    set_houdini_attr(f"{geo_node}/Action", "value1v1", action)

    # Cook Score node and get reward
    hou.node(f"{geo_node}/Score").cook()
    reward = get_houdini_attr(f"{geo_node}/Score", "score")
    total_reward = reward

    # Get next state
    next_state = get_houdini_attr(f"{geo_node}/State", "map")
    next_state = torch.tensor(next_state, dtype=torch.float32).to(device).unsqueeze(0)
    state = next_state

    # Increment frame
    if step < num_steps - 1:
        hou.setFrame(hou.frame() + 1)

# Update Reward_Recorder with total reward
set_houdini_attr(f"{geo_node}/Reward_Recorder", "value1v1", total_reward)
hou.node(f"{geo_node}/Result_Recorder").parm("execute").pressButton()

end_time = time.time()

# Output
print(f"Inference complete. Total reward: {total_reward}")
print(f"Actions taken: {actions}")
print(f"Total inference time: {end_time - start_time:.2f} seconds.")
