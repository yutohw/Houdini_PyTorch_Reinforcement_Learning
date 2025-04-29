import hou
import torch
import torch.nn as nn
import time
import numpy as np


class PPOActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2)  # One hidden layer with 2 neurons
        self.actor = nn.Linear(2, output_dim)  # Action probabilities
        self.critic = nn.Linear(2, 1)  # State value

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value


def get_houdini_attr(node_path, attr_name):
    """Get a Houdini detail attribute."""
    node = hou.node(node_path)
    geometry = node.geometry()
    return geometry.attribValue(attr_name)

def set_houdini_param(node_path, param_name, value):
    """Set a Houdini parameter value."""
    node = hou.node(node_path)
    if node:
        node.parm(param_name).set(value)

def cook_houdini_node(node_path):
    """Cook a Houdini node."""
    node = hou.node(node_path)
    if node:
        node.cook(force=True)


input_dim = 1
output_dim = 2
num_steps = 100
geo_node = "/obj/Study_A_Rolling_Inference_Environment"
model_path = r"D:\Users\YUTO\Documents\pg20\year 5\Model\Machine Learning\Reinforcement Learning Models\250403_Study_A_Variation_06.pth"


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
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
