import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np

from model import DQN

env = gym.make("CartPole-v1", render_mode="rgb_array")

env = RecordVideo(env, video_folder="videos", episode_trigger=lambda x: x == 0)

model = DQN(4, 2)
model.load_state_dict(torch.load("best_dqn_cartpole.pth", weights_only=True))
model.eval()

seed = 29
torch.manual_seed(seed)
np.random.seed(seed)

state, _ = env.reset(seed=seed)
done = False
total_reward = 0

while not done:
    state_tensor = torch.FloatTensor(state)
    action = torch.argmax(model(state_tensor)).item()

    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    total_reward += reward

print("Video recorded successfully! Total Reward:", total_reward)

env.close()
