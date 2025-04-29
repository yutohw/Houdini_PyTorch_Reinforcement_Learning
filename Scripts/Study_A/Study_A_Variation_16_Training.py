import hou
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim = 512
output_dim = 256  # Ensuring output space is 100 (0-99 actions)
num_episodes = 5000
num_steps = 10
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon_start = 1.0  # Initial exploration rate
epsilon_end = 0.01  # Final exploration rate
epsilon_decay = 0.999  # Decay rate per episode

geo_node = "/obj/Study_A_Targeting_Environment"
best_model_path = r"D:\Users\YUTO\Documents\pg20\year 5\Model\Machine Learning\Reinforcement Learning Models\250403_Study_A_Variation_16.pth"

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
            nn.Linear(input_dim, 512),  # Increased from 128 to 256
            nn.ReLU(),
            nn.Linear(512, 256),  # Added an extra hidden layer
            nn.ReLU(),
            nn.Linear(256, output_dim)  # Output size remains 100 (0-99 actions)
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
    state = get_houdini_attr(f"{geo_node}/State", "map")
    state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)  # Ensure shape is [1, 100]
    
    actions = []
    total_reward = 0

    for step in range(num_steps):
        # Epsilon-greedy policy
        if random.random() < epsilon:
            action = random.randint(0, 99)  # Random action from 0 to 99
        else:
            q_values = model(state)  # Shape should be [1, 100]
            action = torch.argmax(q_values, dim=1).item()  # Get best action

        actions.append(action)

        # Set action in Houdini
        set_houdini_attr(f"{geo_node}/Action", "value1v1", action)

        # Cook Score node to get reward
        hou.node(f"{geo_node}/Score").cook()
        reward = get_houdini_attr(f"{geo_node}/Score", "score")
        total_reward = reward

        # Get next state
        next_state = get_houdini_attr(f"{geo_node}/State", "map")
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device).unsqueeze(0)  # Ensure shape is [1, 100]

        # Calculate target and loss
        q_values = model(state)  # Q-values for current state
        next_q_values = model(next_state).detach()  # Q-values for next state
        target = reward + gamma * torch.max(next_q_values).item()

        target_f = q_values.clone().detach()
        target_f[0, action] = target  # Ensure action is within valid index range

        loss = criterion(q_values, target_f)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update state
        state = next_state

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
