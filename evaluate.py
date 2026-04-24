import gymnasium as gym
import torch
import numpy as np
import time

from model import DQN

# Run evaluation headlessly by default. If you want on-screen rendering,
# install pygame and change this to render_mode="human".
env = gym.make("CartPole-v1")

model = DQN(4, 2)
model.load_state_dict(torch.load("best_dqn_cartpole.pth"))

model.eval()

torch.manual_seed(0)
np.random.seed(0)

state, _ = env.reset()

done = False
total_reward = 0

while not done:

    state_tensor = torch.FloatTensor(state)
    action = torch.argmax(model(state_tensor)).item()

    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    total_reward += reward

    time.sleep(0.02)

print("Total Reward:", total_reward)

env.close()