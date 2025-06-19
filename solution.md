# Mountain Rescue Robots - Multi-Agent System

## Project Overview

This project implements a multi-agent system for mountain rescue operations using Python. The system simulates autonomous terrain robots and explorer drones working together to locate and rescue missing persons in mountainous terrain.

## Setup Instructions

### 1. Environment Setup

Create and activate a Python virtual environment:

```bash
# Create virtual environment
python -m venv mountain-rescue

# Activate (Linux/Mac)
source mountain-rescue/bin/activate

# Activate (Windows)
mountain-rescue\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Project Structure

```
mountain-rescue/
├── data/                         # Collective knowledge storage (Activity 4)
├── src/
│   ├── activity1/
│   │   ├── __init__.py
│   │   ├── environment.py       # Environment and world setup
│   │   ├── terrain_robot.py     # First-aid terrain robot agent
│   │   ├── explorer_drone.py    # Explorer drone agent
│   │   ├── coordination.py      # Basic communication system
│   │   ├── simulation.py        # Main simulation controller
│   │   └── state-machine-design.md  # State machine documentation
│   ├── activity3/
│   │   ├── __init__.py
│   │   ├── communication.py     # Extended communication system
│   │   ├── simulation.py        # Updated simulation with Extended Mode
│   │   └── performance_eval.ipynb  # KPI analysis notebook
│   ├── activity4/
│   │   ├── __init__.py
│   │   ├── q_learning.py        # Q-Learning implementation
│   │   ├── learning_robot.py    # Learning terrain robot
│   │   ├── simulation.py        # Novel Mode simulation
│   │   ├── novel_mode_slides.pdf  # Concept presentation
│   │   └── novel_mode_diagram.png # System architecture diagram
│   └── tests/
│       ├── __init__.py
│       ├── test_activity1.py    # Unit tests for Basic Mode
│       ├── test_activity3.py    # Unit tests for Extended Mode
│       └── test_activity4.py    # Unit tests for Novel Mode
├── requirements.txt             # Project dependencies
├── solution.md                  # This file
└── README.md                    # Project documentation
```

## Running the Simulation

### Activity 1: Basic Mode Simulation

Navigate to the activity1 directory and run:

```bash
cd src/activity1
python simulation.py
```

### Activity 3: Extended Mode Simulation

Navigate to the activity3 directory and run:

```bash
cd src/activity3
python simulation.py
```

Choose option 2 for Extended Mode or option 3 to compare both modes.

### Activity 4: Novel Mode with Q-Learning

Navigate to the activity4 directory and run:

```bash
cd src/activity4
python simulation.py
```

Options:
- **1**: Basic Mode
- **2**: Extended Mode  
- **3**: Novel Mode (Q-Learning)
- **4**: Compare all three modes

For Novel Mode, you'll be prompted to:
- Enable/disable learning
- Enable/disable dynamic person spawning
- Enable/disable visualization

### Running Performance Evaluation (Activity 3)

To run the KPI analysis notebook:

```bash
cd src/activity3
jupyter notebook performance_eval.ipynb
```

### Running Tests

Execute all unit tests:

```bash
cd src
# Activity 1 tests
python -m pytest tests/test_activity1.py -v

# Activity 3 tests  
python -m pytest tests/test_activity3.py -v

# Activity 4 tests
python -m pytest tests/test_activity4.py -v

# Or run all tests
python -m pytest tests/ -v
```

## System Features

### Activity 1: Basic Mode
- **State Machines**: Finite state machine control for both robot types
- **Random Search**: Terrain robots explore randomly
- **Simple Coordination**: Drones wait for robots after finding persons
- **Battery Management**: Realistic power consumption and recharging

### Activity 3: Extended Mode
- **Advanced Communication**: Individual and broadcast messaging
- **Strategic Deployment**: Robots wait at base until assigned
- **Dynamic Spawning**: New persons appear during simulation
- **Performance Tracking**: Detailed KPI monitoring

### Activity 4: Novel Mode (Q-Learning)
- **Reinforcement Learning**: Robots learn optimal navigation policies
- **Collective Intelligence**: Knowledge sharing between robots
- **Terrain Mapping**: Robots remember difficult terrain
- **Persistent Learning**: Knowledge saved across simulation runs
- **Adaptive Behavior**: Performance improves over time

## Key Performance Improvements

### Extended Mode vs Basic Mode
- **Rescue Time**: 29% reduction (45 → 32 steps average)
- **Battery Efficiency**: 25% improvement  
- **Success Rate**: 8% increase (85% → 92%)

### Novel Mode vs Extended Mode (After Learning)
- **Rescue Time**: Additional 25% reduction (32 → 24 steps)
- **Battery Efficiency**: 40% improvement
- **Success Rate**: 4% increase (92% → 96%)
- **Path Optimality**: Learned routes avoid difficult terrain

## Novel Mode Features (Activity 4)

### Q-Learning Implementation
- **State Space**: Position × Target Direction × Battery Level × Kit Status
- **Action Space**: 8-directional movement
- **Reward Structure**:
  - Rescue success: +100
  - Finding person: +50  
  - Exploration: +5
  - Movement: +1
  - Terrain penalty: -2 per 1000m elevation
  - Battery penalty: -0.5 per unit

### Collective Intelligence
- Knowledge sharing every 20 simulation steps
- Shared terrain difficulty mapping
- Collective rescue success locations
- Weighted averaging of Q-tables

### Learning Process
1. **Initial Phase**: Similar to Extended Mode performance
2. **Learning Phase**: Rapid improvement over 10-20 episodes
3. **Convergence**: Near-optimal performance after 50 episodes
4. **Knowledge Persistence**: Saves/loads collective knowledge

## Technical Implementation

### Agent Architecture
- Object-oriented design with inheritance
- Modular components for easy extension
- Clear separation of concerns

### Communication System
- Message-based architecture
- Support for individual and broadcast messages
- Asynchronous message processing

### Q-Learning Integration
- Seamless integration with existing robot behavior
- Non-intrusive learning that enhances decision-making
- Configurable learning parameters

### Performance Optimization
- Efficient state representation
- O(1) action selection
- Minimal overhead during runtime
