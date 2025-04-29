import hou
import torch
import torch.nn as nn
import time

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_steps = 100
geo_node = "/obj/Study_A_Rolling_Inference_Environment"
model_path = r"D:\Users\YUTO\Documents\pg20\year 5\Model\Machine Learning\Reinforcement Learning Models\250403_Study_A_Variation_01.pth"

def get_houdini_attr(node_path, attr_name):
    """Get a Houdini detail attribute."""
    node = hou.node(node_path)
    geometry = node.geometry()
    return geometry.attribValue(attr_name)

def set_houdini_attr(node_path, parm_name, value):
    """Set a Houdini parameter value."""
    node = hou.node(node_path)
    node.parm(parm_name).set(value)

def cook_node(node_path):
    """Cook a Houdini node."""
    node = hou.node(node_path)
    if node:
        node.cook(force=True)

# Define the model (must match training model architecture)
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 2)
        )

    def forward(self, x):
        return self.fc(x)

# Load model
q_network = QNetwork().to(device)
q_network.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
q_network.eval()

# Start inference
start_time = time.time()
hou.setFrame(1)

# Initial state
state = get_houdini_attr(f"{geo_node}/State", "state")
state = torch.tensor([[state]], dtype=torch.float32).to(device)

actions_taken = []

for step in range(1, num_steps + 1):
    with torch.no_grad():
        q_values = q_network(state)
        action = torch.argmax(q_values).item()

    actions_taken.append(action)

    # Set action and cook
    set_houdini_attr(f"{geo_node}/Action", "value1v1", action)
    cook_node(f"{geo_node}/Score")

    # Get reward
    reward = get_houdini_attr(f"{geo_node}/Score", "reward")
    print(f"Step {step}: Action = {action}, Reward = {reward}")

    # Get next state
    next_state = get_houdini_attr(f"{geo_node}/State", "state")
    next_state = torch.tensor([[next_state]], dtype=torch.float32).to(device)

    state = next_state

    # Advance Houdini frame
    if step < num_steps:
        hou.setFrame(hou.frame() + 1)

print("Actions taken:", actions_taken)

# Record inference result (match training script behavior)
hou.node(f"{geo_node}/Result_Recorder").parm("execute").pressButton()

end_time = time.time()
print(f"Inference complete. Time taken: {end_time - start_time:.2f} seconds.")
