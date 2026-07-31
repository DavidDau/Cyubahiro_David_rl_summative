# Kigali Urban Noise Inspection Reinforcement Learning System

## Project Overview

This project develops a reinforcement learning-based intelligent inspection agent for optimizing urban noise pollution monitoring in Kigali, Rwanda.

The agent represents an environmental inspection vehicle that learns how to efficiently navigate inspection zones, identify potential noise regulation violations, and maximize inspection performance while minimizing operational costs.

The project compares four reinforcement learning algorithms:

- Deep Q-Network (DQN)
- Proximal Policy Optimization (PPO)
- Advantage Actor-Critic (A2C)
- REINFORCE Policy Gradient

---

# Mission Objective

Urban noise pollution affects public health, productivity, and quality of life.

The objective of this system is to train an intelligent inspection agent that can:

- Prioritize high-risk noise pollution zones
- Reduce unnecessary travel
- Detect more violations within limited resources
- Support data-driven environmental inspection decisions

---

# System Architecture

Cyubahiro_David_rl_summative/

│
├── api/
│ ├── app.py
│ └── routes.py
│
├── assets/
│
├── configs/
│ ├── dqn.yaml
│ ├── ppo.yaml
│ ├── a2c.yaml
│ └── reinforce.yaml
│
├── environment/
│ ├── custom_env.py
│ ├── map.py
│ ├── rendering.py
│ └── init.py
│
├── evaluation/
│ ├── metrics.py
│ ├── comparison.py
│ └── plots.py
│
├── training/
│ ├── dqn_training.py
│ ├── ppo_training.py
│ ├── a2c_training.py
│ └── reinforce_training.py
│
├── models/
│
├── logs/
│
├── main.py
├── pyproject.toml
└── uv.lock

---

# Environment Description

## Environment Type

Custom Gymnasium reinforcement learning environment.

The environment represents Kigali as a network of inspection zones connected through roads.

## Zones

Each zone contains:

- Location coordinates
- Zone type
- Noise violation probability
- Inspection status

Supported zone categories:

- Residential
- Commercial
- Industrial
- Entertainment
- Worship

---

# Agent

The agent represents an autonomous environmental inspection vehicle.

The goal of the agent is to determine the optimal inspection sequence.

---

# Action Space

The agent can:

| Action  | Description                               |
| ------- | ----------------------------------------- |
| Move    | Navigate between inspection zones         |
| Inspect | Measure noise level and detect violations |
| Wait    | Consume time without movement             |

---

# Observation Space

The agent observes:

- Current location
- Remaining battery
- Remaining inspection time
- Number of detected violations
- Number of inspected zones
- Current zone violation probability

---

# Reward Function

| Action                     | Reward |
| -------------------------- | ------ |
| Detect violation           | +100   |
| Inspect compliant location | +20    |
| Move to new location       | -2     |
| Revisit inspected location | -10    |
| Hit invalid route          | -50    |
| Complete mission           | +150   |
| Mission failure            | -100   |

The reward function encourages efficient inspection planning and high violation detection.

---

# Reinforcement Learning Algorithms

## 1. Deep Q-Network (DQN)

Value-based reinforcement learning algorithm.

Implementation: training/dqn_training.py

---

## 2. Proximal Policy Optimization (PPO)

Policy optimization algorithm focused on stable learning.

## Implementation: training/ppo_training.py

## 3. Advantage Actor-Critic (A2C)

Actor-critic algorithm combining value estimation and policy learning.

Implementation: training/a2c_training.py

---

## 4. REINFORCE

Monte Carlo policy gradient algorithm.

Implementation: training/reinforce_training.py

---

# Installation

This project uses `uv` for dependency and environment management.

Clone repository:

```bash
git clone <repository-url>

Navigate into project:

cd Cyubahiro_David_rl_summative

Install dependencies:

uv sync

Activate environment:

Windows:

.venv\Scripts\activate
Running the Environment

Run the simulation:

python main.py

The visualization displays:

Kigali inspection zones
Road connections
Agent movement
Inspection progress
Training Models
DQN
python training/dqn_training.py
PPO
python training/ppo_training.py
A2C
python training/a2c_training.py
REINFORCE
python training/reinforce_training.py

Trained models are stored in:

models/
Evaluation

Run model comparison:

python evaluation/comparison.py

Generate performance graphs:

python evaluation/plots.py

Evaluation metrics:

Average cumulative reward
Number of violations detected
Mission completion rate
Episode duration

API Deployment

The trained PPO model can be accessed through FastAPI.

Start API:

python -m uvicorn api.app:app --reload

Swagger documentation:

http://127.0.0.1:8000/docs

Available endpoints:

Endpoint	Purpose
/health	API status
/predict	Generate agent action
/reset	Reset environment
```

### Technologies Used

Programming
Python 3.11+
Reinforcement Learning
Gymnasium
Stable-Baselines3
PyTorch
Visualization
Pygame
Matplotlib
Deployment
FastAPI
Uvicorn
Environment Management
uv

### Future Improvements

Possible extensions:

Real Kigali geographic data integration
Real-time noise sensor streams
3D city simulation
Multi-agent inspection vehicles
Mobile/web monitoring dashboard

Author: David Cyubahiro

Mission-Based Reinforcement Learning Project
