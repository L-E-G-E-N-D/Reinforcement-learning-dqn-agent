# Deep Q Network (DQN) for CartPole

## Overview

This project implements a Deep Q-Network (DQN) agent using PyTorch to solve the CartPole-v1 environment from Gymnasium.

The goal is to train an agent that learns to balance a pole on a cart by maximizing cumulative reward.

---

## Method

The agent uses:

* Deep Q-Learning (DQN)
* Experience Replay Buffer
* Epsilon-Greedy Exploration

### State Space

* Cart Position
* Cart Velocity
* Pole Angle
* Pole Angular Velocity

### Action Space

* 0 → Move Left
* 1 → Move Right

---

## Results

The agent successfully learns to balance the pole over time.

* Initial reward: ~10–20
* Final reward: 200–500

Training curve:

![Training Curve](training_results.png)

---

## Tech Stack

* Python
* PyTorch
* Gymnasium
* NumPy
* Matplotlib

---

## How to Run

```bash
pip install -r requirements.txt
python train.py
```

---

## Future Improvements

* Implement Target Network (Double DQN)
* Try PPO (Policy Gradient)
* Hyperparameter tuning
* Apply to more complex environments (Atari)

---

## Author

Tanush
