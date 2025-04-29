import torch
import hou
import random

# ===========================
# PPOActorCritic Model Definition
# ===========================
class PPOActorCritic(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOActorCritic, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 1024)  # Increased width
        self.fc2 = torch.nn.Linear(1024, 512)  # Increased width
        self.fc3 = torch.nn.Linear(512, 256)  # Added an extra layer
        self.fc4 = torch.nn.Linear(256, 64)  # Added an extra layer
        self.actor = torch.nn.Linear(64, output_dim)  # Policy output
        self.critic = torch.nn.Linear(64, 1)  # Value function output

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)  # Policy distribution
        value = self.critic(x)  # Value estimation
        return action_probs, value


# ===========================
# Load the Saved Model with weights_only=True
# ===========================
def load_model(model_path):
    checkpoint = torch.load(model_path, weights_only=True)
    model = PPOActorCritic(input_dim=checkpoint['input_dim'], output_dim=checkpoint['output_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['epsilon']

# ===========================
# Inference Logic for One Episode (With Exploration)
# ===========================
def infer_one_episode(state, model, geo_node, epsilon=0.01):
    done = False
    actions_taken = []
    rewards = []
    states = []

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():  # No need to compute gradients during inference
            action_probs, _ = model(state_tensor)
        
        # Exploration (random action) or exploitation (best action)
        if random.random() < epsilon:  # Exploration
            action = random.choice([0, 1])  # Random action selection
        else:  # Exploitation
            action = torch.argmax(action_probs, dim=1).item()  # Best action (greedy)

        actions_taken.append(action)

        # Set action to Houdini
        hou.node(f"{geo_node}/Action_01").parm("value1v1").set(action)
        hou.node(f"{geo_node}/Score_01").cook(force=True)  # Cook the node to get the reward
        reward = hou.node(f"{geo_node}/Score_01").geometry().attribValue("reward")
        rewards.append(reward)

        done = hou.node(f"{geo_node}/Stop_01").geometry().attribValue("stop") == 1

        # Move to the next state
        states.append(state)
        state = hou.node(f"{geo_node}/State_01").geometry().attribValue("state")

        if not done:
            hou.setFrame(hou.frame() + 1)

    return actions_taken, rewards, states


BEST_MODEL_PATH = r"D:\Users\YUTO\Documents\pg20\year 5\Model\Machine Learning\Reinforcement Learning Models\250328_Neighbour_Ray_Checker_PPO_08.pth"

# Load the saved model for inference
model, epsilon = load_model(BEST_MODEL_PATH)
model.eval()  # Set the model to evaluation mode

# Get the state from Houdini
GEO_NODE = "/obj/Study_C_Environment_01"
state = hou.node(f"{GEO_NODE}/State_01").geometry().attribValue("state")

# Perform inference for one entire episode, using the epsilon value from training
actions_taken, rewards, states = infer_one_episode(state, model, GEO_NODE, epsilon)

# Optionally, print or use the actions and rewards
print(f"Actions Taken: {actions_taken}")
print(f"Rewards: {rewards}")
