import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import hou
import time

# ===========================
# Configurations for Easy Updates
# ===========================
GEO_NODE = "/obj/Study_B_Environment_02"
BEST_MODEL_PATH = r"D:\Users\YUTO\Documents\pg20\year 5\Model\Machine Learning\Reinforcement Learning Models\250406_Study_B_Variation_05.pth"

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
# PPO Functions
# ===========================
def select_action(model, state, policy_old, epsilon):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
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
    # ✅ Convert list to NumPy array first to avoid slow tensor creation
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

        # 🔹 FIX: Correctly select action probabilities
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

def update_episode_number(node_path, episode):
    set_houdini_param(node_path, "episodenumber", episode)

# ===========================
# Training Loop
# ===========================
model = PPOActorCritic(input_dim=3, output_dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_episodes = 1000
epsilon = 1.0  
epsilon_decay = 0.99
epsilon_min = 0.05

best_reward = float('-inf')
best_model = None

start_time = time.time()

for episode in range(1, num_episodes + 1):
    hou.setFrame(1)  # Reset Houdini frame at episode start
    update_episode_number(f"{GEO_NODE}/Episode_Update_01", episode)
    done = False
    state = get_houdini_attr(f"{GEO_NODE}/State_01", "state")
    
    rewards = []
    actions_taken = []
    states = []
    policy_old = []

    while not done:
        action = select_action(model, state, policy_old, epsilon)
        actions_taken.append(action)

        # Apply action in Houdini
        set_houdini_param(f"{GEO_NODE}/Action_01", "value1v1", action)

        # Cook Score node to get reward
        cook_houdini_node(f"{GEO_NODE}/Score_01")
        reward = get_houdini_attr(f"{GEO_NODE}/Score_01", "reward")
        rewards.append(reward)

        # Check for termination
        done = get_houdini_attr(f"{GEO_NODE}/Stop_01", "stop") == 1
        if not done:
            hou.setFrame(hou.frame() + 1)  # Increment frame

        states.append(state)
        state = get_houdini_attr(f"{GEO_NODE}/State_01", "state")  # Get new state

    update_policy(model, optimizer, states, actions_taken, rewards, policy_old)
    
    end_time = time.time()

    # Logging
    print(f"Episode {episode}: Reward = {rewards[-1]}")
    
    if episode % 100 == 0:
        print(f"Actions in Episode {episode}: {actions_taken}")
    
    if episode % 10 == 0:
        print(f"time: {end_time - start_time:.2f} seconds")

    # Save best model
    if episode > num_episodes * 0.95 and rewards[-1] > best_reward:
        best_reward = rewards[-1]
        best_model = model.state_dict()

    set_houdini_param(f"{GEO_NODE}/Reward_Recorder_01", "value1v1", rewards[-1])
    set_houdini_param(f"{GEO_NODE}/Result_Recorder_01", "loadfromdisk", 0)
    set_houdini_param(f"{GEO_NODE}/Result_Recorder_01", "version", episode)
    hou.node(f"{GEO_NODE}/Result_Recorder_01").parm("execute").pressButton()

    epsilon = max(epsilon * epsilon_decay, epsilon_min)  # Reduce exploration

torch.save(best_model, BEST_MODEL_PATH)
