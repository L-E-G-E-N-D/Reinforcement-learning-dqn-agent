import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import DQN
from replay_buffer import ReplayBuffer

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print("State dim:", state_dim)
print("Action dim:", action_dim)

model = DQN(state_dim, action_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

buffer = ReplayBuffer(10000)

episodes = 500
batch_size = 64
gamma = 0.99

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

rewards = []

def moving_average(data, window_size=20):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

for episode in range(episodes):

    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:

        # epsilon-greedy action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)

        if len(buffer) > batch_size:

            states, actions, rewards_b, next_states, dones = buffer.sample(batch_size)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards_b = torch.FloatTensor(rewards_b)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            q_values = model(states)
            next_q_values = model(next_states)

            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1)[0]

            expected_q = rewards_b + gamma * next_q_value * (1 - dones)

            loss = torch.nn.functional.mse_loss(q_value, expected_q.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state
        total_reward += reward
    
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    rewards.append(total_reward)

    print(f"Episode {episode}, Reward: {total_reward}")

plt.plot(rewards, label="Raw Rewards")

ma_rewards = moving_average(rewards)
plt.plot(range(len(ma_rewards)), ma_rewards, label="Moving Average", linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Curve")

plt.legend()

plt.savefig("training_results.png")
torch.save(model.state_dict(), "dqn_cartpole.pth")
plt.show()

