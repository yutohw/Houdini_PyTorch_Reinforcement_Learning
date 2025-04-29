import hou
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_episodes = 1000
num_steps = 256
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon_start = 1.0  # Initial exploration rate
epsilon_end = 0.01  # Final exploration rate
epsilon_decay = 0.99  # Decay rate per episode

geo_node = "/obj/Study_A_Rolling_Environment"
best_model_path = r"D:\Users\YUTO\Documents\pg20\year 5\Model\Machine Learning\Reinforcement Learning Models\250428_Study_A_Variation_03.pth"

def get_houdini_attr(node_path, attr_name):
    """Get a Houdini detail attribute."""
    node = hou.node(node_path)
    geometry = node.geometry()
    return geometry.attribValue(attr_name)

def set_houdini_attr(node_path, parm_name, value):
    """Set a Houdini parameter value."""
    node = hou.node(node_path)
    node.parm(parm_name).set(value)

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

# Initialize model, optimizer, and loss function
model = QNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

epsilon = epsilon_start  # Initialize exploration rate
best_reward = float('-inf')  # Track the best reward
best_model_state = None  # Store the best model parameters

start_time = time.time()

# Training loop
for episode in range(1, num_episodes + 1):
    # Reset Houdini frame to 1
    hou.setFrame(1)

    # Initialize episode data
    state = get_houdini_attr(f"{geo_node}/State", "state")
    state = torch.tensor([[state]], dtype=torch.float32).to(device)
    actions = []
    total_reward = 0

    for step in range(num_steps):
        # Epsilon-greedy policy
        if random.random() < epsilon:
            action = random.randint(0, 1)  # Exploration: random action
        else:
            q_values = model(state)
            action = torch.argmax(q_values).item()  # Exploitation: best action

        actions.append(action)

        # Set action in Houdini
        set_houdini_attr(f"{geo_node}/Action", "value1v1", action)

        # Cook Score node to get reward
        hou.node(f"{geo_node}/Score").cook()
        reward = get_houdini_attr(f"{geo_node}/Score", "reward")
        total_reward += reward

        # Get next state
        next_state = get_houdini_attr(f"{geo_node}/State", "state")
        next_state = torch.tensor([[next_state]], dtype=torch.float32).to(device)

        # Calculate target and loss
        q_values = model(state)
        target = reward + gamma * torch.max(model(next_state)).item()
        target_f = q_values.clone()
        target_f[0][action] = target

        loss = criterion(q_values, target_f.detach())

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update state
        state = next_state

        if step < num_steps - 1:
            hou.node(f"{geo_node}/Geometry_Recorder").cook()
        
        # Increment frame except on last step
        if step < num_steps - 1:
            hou.setFrame(hou.frame() + 1)

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # Print episode reward
    print(f"Episode {episode}: Total Reward = {total_reward}")

    # Print actions every 10 episodes
    if episode % 10 == 0:
        print(f"Episode {episode}: Actions = {actions}")

    # Update Reward_Recorder_01 with total reward
    set_houdini_attr(f"{geo_node}/Reward_Recorder", "value1v1", total_reward)
    
    # Update Result_Recorder_01 parameters
    set_houdini_attr(f"{geo_node}/Result_Recorder", "loadfromdisk", 0)
    set_houdini_attr(f"{geo_node}/Result_Recorder", "version", episode)
    hou.node(f"{geo_node}/Result_Recorder").parm("execute").pressButton()

    # Track the best model (without saving yet)
    if total_reward > best_reward:
        best_reward = total_reward
        best_model_state = model.state_dict()  # Store the best model state

end_time = time.time()

# Save the best model **only once** at the end
if best_model_state:
    torch.save(best_model_state, best_model_path)
    print(f"Best model saved at: {best_model_path} with reward {best_reward}")

total_time = end_time - start_time
print(f"Training complete. Total training time: {total_time:.2f} seconds.")
