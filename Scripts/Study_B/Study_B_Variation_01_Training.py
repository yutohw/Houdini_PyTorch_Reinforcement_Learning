import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import hou
import time
import numpy as np
from collections import deque

# Define the geometry node reference
GEOMETRY_NODE = "/obj/Massing_Solar_Environment_12"

# Define the Q-Network
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

# Hyperparameters
input_size = 4  # Length of "state" array
output_size = 2  # Two discrete actions: 0 or 1
lr = 0.001
batch_size = 32
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995
replay_memory_size = 10000
update_target_every = 10  # Update target network every X episodes

# Initialize networks, optimizer, and replay buffer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_network = QNetwork(input_size, output_size).to(device)
target_network = QNetwork(input_size, output_size).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()
optimizer = optim.Adam(q_network.parameters(), lr=lr)
memory = deque(maxlen=replay_memory_size)

# Houdini helper functions
def get_houdini_attr(node_path, attr_name):
    geo = hou.node(node_path).geometry()
    return geo.attribValue(attr_name)

def get_houdini_state():
    state = get_houdini_attr(f"{GEOMETRY_NODE}/State_01", "state")
    return np.array(state, dtype=np.float32).flatten()

def get_reward():
    hou.node(f"{GEOMETRY_NODE}/Score_01").cook()
    return get_houdini_attr(f"{GEOMETRY_NODE}/Score_01", "reward")

def check_termination():
    return get_houdini_attr(f"{GEOMETRY_NODE}/Stop_01", "stop") == 1

def apply_action(action):
    action_node = hou.node(f"{GEOMETRY_NODE}/Action_01")
    action_node.parm("value1v1").set(action)

def update_result_recorder(episode, reward):
    recorder = hou.node(f"{GEOMETRY_NODE}/Result_Recorder_01")
    recorder.parm("loadfromdisk").set(0)
    recorder.parm("version").set(episode)
    recorder.parm("execute").pressButton()

def save_model(model, path):
    torch.save(model.state_dict(), path)

# Training parameters
num_episodes = 1000
best_reward = -float("inf")
best_model_path = "D:/Users/YUTO/Documents/pg20/year 5/Model/Machine Learning/Reinforcement Learning Models/250406_Study_B_Variation_01.pth"

def train_from_memory():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0]
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))
    
    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

start_time = time.time()
    
for episode in range(1, num_episodes + 1):
    hou.setFrame(1)
    state = torch.tensor(get_houdini_state(), dtype=torch.float32, device=device)
    episode_reward = None
    action_list = []
    
    while True:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                action = torch.argmax(q_network(state)).item()
        
        apply_action(action)
        action_list.append(action)
        
        # Get new reward and next state
        reward = get_reward()
        episode_reward = reward  # Not cumulative, always latest reward
        next_state = torch.tensor(get_houdini_state(), dtype=torch.float32, device=device)
        done = check_termination()
        
        # Store in replay buffer
        memory.append((state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done))
        state = next_state
        
        # Train DQN
        train_from_memory()
        
        # Check termination condition
        if done:
            break
        
        # Move Houdini frame forward
        hou.setFrame(hou.frame() + 1)
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Log results
    print(f"Episode {episode}: Reward = {episode_reward}")
    hou.node(f"{GEOMETRY_NODE}/Reward_Recorder_01").parm("value1v1").set(episode_reward)
    update_result_recorder(episode, episode_reward)
    
    if episode % 100 == 0:
        print(f"Episode {episode} actions: {action_list}")
    
    # Update target network every X episodes
    if episode % update_target_every == 0:
        target_network.load_state_dict(q_network.state_dict())
    
    # Save best-performing model
    if episode >= num_episodes * 0.95 and episode_reward > best_reward:
        best_reward = episode_reward
        save_model(q_network, best_model_path)

print("Training complete.")
print(f"Total training time: {time.time() - start_time:.2f} seconds")