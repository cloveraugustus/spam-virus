
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Set up environment and device
env = gym.make("MountainCar-v0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural Network (Q-network)
class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model, optimizer, loss
brain = Brain().to(device)
optimizer = optim.Adam(brain.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.98
memory = deque(maxlen=10000)
batch_size = 64

# Choose action using ε-greedy policy
def choose_action(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return brain(state_tensor).argmax().item()

# Training function
def train():
    if len(memory) < 1000:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.BoolTensor(dones).to(device)

    q_values = brain(states).gather(1, actions).squeeze()
    next_q_values = brain(next_states).max(1)[0]
    targets = rewards + gamma * next_q_values * (~dones)

    loss = loss_fn(q_values, targets.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Optional reward shaping (can improve learning)
        reward += abs(next_state[0] + 0.5)

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

env.close()
