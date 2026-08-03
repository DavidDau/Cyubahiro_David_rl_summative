# Kigali Urban Noise Inspection Reinforcement Learning Environment

## Project Overview

This project develops a custom **mission-based reinforcement learning (RL) environment** for urban noise pollution inspection in Kigali, Rwanda.

The goal is to train and compare different reinforcement learning algorithms to optimize an inspection agent that navigates Kigali zones, discovers hidden noise violations, and completes an inspection mission while managing limited resources such as time and battery.

The project compares:

- **Deep Q-Network (DQN)** — Value-Based Reinforcement Learning
- **Proximal Policy Optimization (PPO)** — Policy Optimization
- **Advantage Actor-Critic (A2C)** — Actor-Critic Method
- **REINFORCE** — Policy Gradient Method

The trained agents are evaluated based on:

- Total reward achieved
- Mission completion rate
- Number of detected violations
- Episode duration
- Learning performance

---

# Mission Description

Urban noise pollution is a growing challenge in rapidly developing cities. Effective inspection requires deciding:

- Which locations should be inspected first?
- How should inspection vehicles move through the city?
- How can violations be detected efficiently under limited resources?

This project models the problem as a reinforcement learning task where an autonomous inspection agent learns optimal navigation and inspection strategies.

---

# Environment Design

The environment represents Kigali as a connected road network containing multiple inspection zones.

Each zone contains:

- Location coordinates
- Zone category
- Noise violation probability
- Hidden violation state
- Inspection status

The agent does not initially know which zones contain violations. It must explore and inspect locations to discover violations.

## Kigali Zones Included

| Zone ID | Location   | Category      |
| ------- | ---------- | ------------- |
| 0       | Kacyiru    | Residential   |
| 1       | Kimihurura | Commercial    |
| 2       | Remera     | Commercial    |
| 3       | Amahoro    | Entertainment |
| 4       | Kimironko  | Commercial    |
| 5       | Kanombe    | Residential   |
| 6       | Gikondo    | Industrial    |
| 7       | Nyamirambo | Entertainment |
| 8       | Nyarugenge | Commercial    |
| 9       | Kimisagara | Worship       |

---

# Reinforcement Learning Formulation

## State Space

The agent observes:

```
[
Current Zone,
Battery Level,
Remaining Time,
Detected Violations,
Inspected Zones,
Zone Risk Probability
]
```

## Action Space

The agent can:

```
0-3  : Move between connected zones
4    : Inspect current zone
5    : Wait
```

## Reward System

The agent receives rewards based on:

Positive rewards:

- Discovering new zones
- Successfully inspecting violations
- Completing missions

Negative rewards:

- Repeated inspections
- Wasting time
- Resource exhaustion
- Failed actions

The objective is to maximize total mission reward.

---

# Project Structure

```
Cyubahiro_David_rl_summative

├── api
│   ├── app.py
│   └── routes.py

├── assets
│   ├── plots
│   │   ├── algorithm_comparison.png
│   │   ├── completion_comparison.png
│   │   ├── reward_comparison.png
│   │   ├── violation_comparison.png
│   │   └── convergence_plot.png
│   │
│   └── report_tables
│       ├── A2C_hyperparameters.csv
│       ├── DQN_hyperparameters.csv
│       ├── PPO_hyperparameters.csv
│       └── REINFORCE_hyperparameters.csv

├── configs
│   ├── a2c.yaml
│   ├── dqn.yaml
│   ├── ppo.yaml
│   └── reinforce.yaml

├── environment
│   ├── custom_env.py
│   ├── map.py
│   ├── rendering.py
│   └── test_render.py

├── evaluation
│   ├── comparison.py
│   ├── metrics.py
│   ├── plots.py
│   ├── advanced_plots.py
│   └── generate_report_tables.py

├── experiments
│   ├── hyperparameter_search.py
│   └── reinforce_experiments.py

├── models
│   ├── dqn
│   ├── ppo
│   ├── a2c
│   └── reinforce

├── training
│   ├── dqn_training.py
│   ├── ppo_training.py
│   ├── a2c_training.py
│   └── reinforce_training.py

├── main.py
├── run.py
├── pyproject.toml
└── uv.lock
```

---

# Training Pipeline

The training workflow:

```
Custom Environment
        |
        ↓
Agent Training
        |
        ↓
Saved Model
        |
        ↓
Evaluation
        |
        ↓
Performance Comparison
        |
        ↓
Visualization
```

---

# Training Algorithms

## DQN

Deep Q-Network learns the optimal action-value function:

```
Q(state, action)
```

Strength:

- Effective for discrete action spaces
- Stable value estimation

---

## PPO

Proximal Policy Optimization improves policy stability by limiting large policy updates.

Strength:

- Stable learning
- Good performance in complex environments

---

## A2C

Advantage Actor-Critic combines:

- Actor: chooses actions
- Critic: evaluates actions

Strength:

- Faster learning through policy/value combination

---

## REINFORCE

Policy gradient method that directly optimizes expected rewards.

Strength:

- Simple policy optimization approach

---

# Model Evaluation

Each trained agent is evaluated using:

## Metrics

- Average reward
- Reward standard deviation
- Average detected violations
- Mission completion rate
- Average episode length

Evaluation scripts:

```
evaluation/metrics.py

evaluation/comparison.py
```

---

# Visualization

Generated plots include:

- Algorithm reward comparison
- Mission completion comparison
- Violation detection comparison
- Learning convergence curves

Stored in:

```
assets/plots/
```

---

# Running the Project

## Install Dependencies

Using UV:

```bash
uv sync
```

Activate environment:

```bash
.venv\Scripts\activate
```

---

# Test Environment Rendering

```bash
python environment/test_render.py
```

Expected:

- Pygame window opens
- Kigali zones displayed
- Agent movement visualization

---

# Train Models

### PPO

```bash
python training/ppo_training.py
```

### DQN

```bash
python training/dqn_training.py
```

### A2C

```bash
python training/a2c_training.py
```

### REINFORCE

```bash
python training/reinforce_training.py
```

---

# Run Agent Simulation

Run:

```bash
python main.py
```

The system loads a trained model and displays:

- Agent movement
- Zone inspection
- Rewards
- Mission completion

---

# Evaluation

Compare trained algorithms:

```bash
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

# Kigali Urban Noise Inspection Reinforcement Learning Environment

## Project Overview

This project develops a custom **mission-based reinforcement learning (RL) environment** for urban noise pollution inspection in Kigali, Rwanda.

The goal is to train and compare different reinforcement learning algorithms to optimize an inspection agent that navigates Kigali zones, discovers hidden noise violations, and completes an inspection mission while managing limited resources such as time and battery.

The project compares:

- **Deep Q-Network (DQN)** — Value-Based Reinforcement Learning
- **Proximal Policy Optimization (PPO)** — Policy Optimization
- **Advantage Actor-Critic (A2C)** — Actor-Critic Method
- **REINFORCE** — Policy Gradient Method

The trained agents are evaluated based on:

- Total reward achieved
- Mission completion rate
- Number of detected violations
- Episode duration
- Learning performance

---

# Mission Description

Urban noise pollution is a growing challenge in rapidly developing cities. Effective inspection requires deciding:

- Which locations should be inspected first?
- How should inspection vehicles move through the city?
- How can violations be detected efficiently under limited resources?

This project models the problem as a reinforcement learning task where an autonomous inspection agent learns optimal navigation and inspection strategies.

---

# Environment Design

The environment represents Kigali as a connected road network containing multiple inspection zones.

Each zone contains:

- Location coordinates
- Zone category
- Noise violation probability
- Hidden violation state
- Inspection status

The agent does not initially know which zones contain violations. It must explore and inspect locations to discover violations.

## Kigali Zones Included

| Zone ID | Location   | Category      |
| ------- | ---------- | ------------- |
| 0       | Kacyiru    | Residential   |
| 1       | Kimihurura | Commercial    |
| 2       | Remera     | Commercial    |
| 3       | Amahoro    | Entertainment |
| 4       | Kimironko  | Commercial    |
| 5       | Kanombe    | Residential   |
| 6       | Gikondo    | Industrial    |
| 7       | Nyamirambo | Entertainment |
| 8       | Nyarugenge | Commercial    |
| 9       | Kimisagara | Worship       |

---

# Reinforcement Learning Formulation

## State Space

The agent observes:

```
[
Current Zone,
Battery Level,
Remaining Time,
Detected Violations,
Inspected Zones,
Zone Risk Probability
]
```

## Action Space

The agent can:

```
0-3  : Move between connected zones
4    : Inspect current zone
5    : Wait
```

## Reward System

The agent receives rewards based on:

Positive rewards:

- Discovering new zones
- Successfully inspecting violations
- Completing missions

Negative rewards:

- Repeated inspections
- Wasting time
- Resource exhaustion
- Failed actions

The objective is to maximize total mission reward.

---

# Project Structure

```
Cyubahiro_David_rl_summative

├── api
│   ├── app.py
│   └── routes.py

├── assets
│   ├── plots
│   │   ├── algorithm_comparison.png
│   │   ├── completion_comparison.png
│   │   ├── reward_comparison.png
│   │   ├── violation_comparison.png
│   │   └── convergence_plot.png
│   │
│   └── report_tables
│       ├── A2C_hyperparameters.csv
│       ├── DQN_hyperparameters.csv
│       ├── PPO_hyperparameters.csv
│       └── REINFORCE_hyperparameters.csv

├── configs
│   ├── a2c.yaml
│   ├── dqn.yaml
│   ├── ppo.yaml
│   └── reinforce.yaml

├── environment
│   ├── custom_env.py
│   ├── map.py
│   ├── rendering.py
│   └── test_render.py

├── evaluation
│   ├── comparison.py
│   ├── metrics.py
│   ├── plots.py
│   ├── advanced_plots.py
│   └── generate_report_tables.py

├── experiments
│   ├── hyperparameter_search.py
│   └── reinforce_experiments.py

├── models
│   ├── dqn
│   ├── ppo
│   ├── a2c
│   └── reinforce

├── training
│   ├── dqn_training.py
│   ├── ppo_training.py
│   ├── a2c_training.py
│   └── reinforce_training.py

├── main.py
├── run.py
├── pyproject.toml
└── uv.lock
```

---

# Training Pipeline

The training workflow:

```
Custom Environment
        |
        ↓
Agent Training
        |
        ↓
Saved Model
        |
        ↓
Evaluation
        |
        ↓
Performance Comparison
        |
        ↓
Visualization
```

---

# Training Algorithms

## DQN

Deep Q-Network learns the optimal action-value function:

```
Q(state, action)
```

Strength:

- Effective for discrete action spaces
- Stable value estimation

---

## PPO

Proximal Policy Optimization improves policy stability by limiting large policy updates.

Strength:

- Stable learning
- Good performance in complex environments

---

## A2C

Advantage Actor-Critic combines:

- Actor: chooses actions
- Critic: evaluates actions

Strength:

- Faster learning through policy/value combination

---

## REINFORCE

Policy gradient method that directly optimizes expected rewards.

Strength:

- Simple policy optimization approach

---

# Model Evaluation

Each trained agent is evaluated using:

## Metrics

- Average reward
- Reward standard deviation
- Average detected violations
- Mission completion rate
- Average episode length

Evaluation scripts:

```
evaluation/metrics.py

evaluation/comparison.py
```

---

# Visualization

Generated plots include:

- Algorithm reward comparison
- Mission completion comparison
- Violation detection comparison
- Learning convergence curves

Stored in:

```
assets/plots/
```

---

# Running the Project

## Install Dependencies

Using UV:

```bash
uv sync
```

Activate environment:

```bash
.venv\Scripts\activate
```

---

# Test Environment Rendering

```bash
python environment/test_render.py
```

Expected:

- Pygame window opens
- Kigali zones displayed
- Agent movement visualization

---

# Train Models

### PPO

```bash
python training/ppo_training.py
```

### DQN

```bash
python training/dqn_training.py
```

### A2C

```bash
python training/a2c_training.py
```

### REINFORCE

```bash
python training/reinforce_training.py
```

---

# Run Agent Simulation

Run:

```bash
python main.py
```

The system loads a trained model and displays:

- Agent movement
- Zone inspection
- Rewards
- Mission completion

---

# Evaluation

Compare trained algorithms:

```bash
python evaluation/comparison.py
```

Generate report tables:

```bash
python evaluation/generate_report_tables.py
```

Generate plots:

```bash
python evaluation/advanced_plots.py
```

---

# Results

The trained agents successfully:

- Navigate Kigali inspection zones
- Discover hidden noise violations
- Complete inspection missions
- Demonstrate different learning behaviours across RL methods

Example successful runs:

| Algorithm |                      Final Reward |
| --------- | --------------------------------: |
| DQN       |    High reward mission completion |
| PPO       |   Stable policy-based exploration |
| A2C       | Actor-critic learning performance |
| REINFORCE |       Policy gradient exploration |

---

# Technologies Used

- Python
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Pygame
- UV Package Manager

---

# Author

**David Cyubahiro**

Machine Learning / Software Engineering Project

Reinforcement Learning Mission-Based Environment

### Future Improvements

Possible extensions:

Real Kigali geographic data integration
Real-time noise sensor streams
3D city simulation
Multi-agent inspection vehicles
Mobile/web monitoring dashboard

Author: David Cyubahiro

Mission-Based Reinforcement Learning Project
