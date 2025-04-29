import hou
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import time

# ===========================
# PPO Neural Network
# ===========================
class PPOActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2)  # Only one hidden layer with 2 neurons
        self.actor = nn.Linear(2, output_dim)  # Policy output (2 possible actions)
        self.critic = nn.Linear(2, 1)  # Value function output (1 value)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # Only one hidden layer with tanh activation
        action_probs = torch.softmax(self.actor(x), dim=-1)  # Policy distribution
        value = self.critic(x)  # Value estimation
        return action_probs, value

# ===========================
# PPO Functions
# ===========================
def select_action(model, state, policy_old, epsilon):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Ensure correct shape (1x1)
    probs, _ = model(state_tensor)

    if random.random() < epsilon:  # Exploration
        action = random.choice([0, 1])
    else:  # Exploitation
        action = torch.argmax(probs, dim=1).item()

    policy_old.append(probs.squeeze(0).detach().numpy())  # Store old policy distribution
    return action

def compute_gae(rewards, values, gamma=0.99, lambda_=0.95):
    advantages = []
    gae = 0
    values = values + [0]  # Terminal state value is 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)

    return advantages

def update_policy(model, optimizer, states, actions, rewards, old_probs, gamma=0.99, epsilon=0.2):
    states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
    actions_tensor = torch.tensor(np.array(actions), dtype=torch.long)
    old_probs_tensor = torch.tensor(np.array(old_probs), dtype=torch.float32)

    _, values = model(states_tensor)
    values = values.squeeze(-1).detach().numpy().tolist()

    advantages = compute_gae(rewards, values)
    advantages_tensor = torch.tensor(np.array(advantages), dtype=torch.float32)

    for _ in range(4):  # Multiple gradient steps
        new_probs, new_values = model(states_tensor)
        new_values = new_values.squeeze(-1)

        # Correctly select action probabilities
        action_probs = new_probs[range(len(actions)), actions]

        ratios = action_probs / old_probs_tensor[range(len(actions)), actions]

        # Clipped PPO loss
        clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
        policy_loss = -torch.min(ratios * advantages_tensor, clipped_ratios * advantages_tensor).mean()

        value_loss = nn.MSELoss()(new_values, torch.tensor(rewards, dtype=torch.float32))

        loss = policy_loss + 0.5 * value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ===========================
# Houdini Utility Functions
# ===========================
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

# ===========================
# Hyperparameters and Initialization
# ===========================
input_dim = 1  # Input size: 1 because state will be scalar (1-dimensional)
output_dim = 2  # Output size
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0  # Start epsilon value
epsilon_decay = 0.99  # Epsilon decay rate
epsilon_end = 0.01  # Minimum epsilon value
lambda_ = 0.95
num_episodes = 1000
num_steps_per_episode = 100  # Define the number of steps per episode

geo_node = "/obj/Environment"

# Create the PPO model and optimizer
model = PPOActorCritic(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Use Adam optimizer from optim

best_reward = float('-inf')
best_model = None

# Start the training process and track time
start_time = time.time()  # Start time tracking

for episode in range(1, num_episodes + 1):
    hou.setFrame(1)  # Reset Houdini frame at episode start
    state = get_houdini_attr(f"{geo_node}/State", "state")
    
    rewards = []
    actions_taken = []
    states = []
    policy_old = []

    for step in range(num_steps_per_episode):  # Loop for steps within each episode
        # Ensure state is a scalar (1-dimensional)
        state = np.array([state])  # Make sure state is an array with 1 element
        action = select_action(model, state, policy_old, epsilon)
        actions_taken.append(action)

        # Apply action in Houdini
        set_houdini_param(f"{geo_node}/Action", "value1v1", action)

        # Cook Score node to get reward
        cook_houdini_node(f"{geo_node}/Score")
        reward = get_houdini_attr(f"{geo_node}/Score", "reward")
        rewards.append(reward)

        states.append(state)
        state = get_houdini_attr(f"{geo_node}/State", "state")  # Get new state

        hou.setFrame(hou.frame() + 1)  # Increment frame

    # The reward for the episode is the reward of the last step
    final_reward = rewards[-1]
    
    update_policy(model, optimizer, states, actions_taken, rewards, policy_old)

    # Logging
    print(f"Episode {episode}: Reward = {final_reward}")
    if episode % 100 == 0:
        print(f"Actions in Episode {episode}: {actions_taken}")

    # Save best model
    if episode > num_episodes * 0.95 and final_reward > best_reward:
        best_reward = final_reward
        best_model = model.state_dict()

    # Update Houdini parameters at the end of episode
    set_houdini_param(f"{geo_node}/Reward_Recorder", "value1v1", final_reward)
    set_houdini_param(f"{geo_node}/Result_Recorder", "loadfromdisk", 0)
    set_houdini_param(f"{geo_node}/Result_Recorder", "version", episode)
    hou.node(f"{geo_node}/Result_Recorder").parm("execute").pressButton()

    # Epsilon decay (reduce exploration)
    epsilon = max(epsilon * epsilon_decay, epsilon_end)

end_time = time.time()
training_time = end_time - start_time

# Print out the total training time
print(f"Total training time: {training_time:.2f} seconds")

# Save best model
torch.save(best_model, "D:/Users/YUTO/Documents/pg20/year 5/Model/Machine Learning/Reinforcement Learning Models/250403_Study_A_Variation_06.pth")
