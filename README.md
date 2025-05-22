# Mountain Rescue Robots - Multi-Agent System

A comprehensive multi-agent simulation system for autonomous mountain rescue operations using terrain robots and explorer drones.

##  **Project Overview**

This project implements a sophisticated multi-agent system for **UFCXR-15-3 Autonomous Agents and Multiagent Systems** at the University of the West of England. The system simulates autonomous rescue operations in mountainous terrain using:

- ** Terrain Robots**: Ground-based rescue vehicles that deliver first-aid kits
- ** Explorer Drones**: Aerial reconnaissance units that locate missing persons
- ** Mountain Environment**: Realistic terrain with elevation zones and hazards
- ** Communication System**: Coordination between different agent types

##  **Key Features**

### **Intelligent Agent Behaviors**
- **Finite State Machines**: Robust state-based decision making
- **Battery Management**: Realistic power consumption and recharging cycles
- **Autonomous Navigation**: Smart pathfinding and obstacle avoidance
- **Coordination Protocols**: Communication between drones and robots

### **Realistic Simulation Environment**
- **Multi-elevation Terrain**: 0-3K meters above sea level
- **Dynamic Rescue Scenarios**: Missing persons randomly distributed
- **Professional Visualization**: Real-time animated simulation
- **Performance Metrics**: Comprehensive success rate tracking

### **Multiple Operation Modes**
- **Basic Mode**: Simple coordination and random search patterns
- **Extended Mode**: Advanced communication and strategic coordination *(Coming in Activity 3)*
- **Novel Mode**: Machine learning and advanced AI features *(Coming in Activity 4)*

##  **Quick Start**

### **Prerequisites**
- Python 3.8+
- Virtual environment (recommended)

### **Installation**
```bash
# Clone the repository
git clone [repository-url]
cd mountain-rescue

# Create virtual environment
python -m venv mountain-rescue
source mountain-rescue/bin/activate  # Linux/Mac
# or: mountain-rescue\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### **Run Simulation**
```bash
cd src/activity1
python simulation.py
```

Choose option **1** for animated visualization or **2** for faster batch mode.

##  **Testing**

Run the comprehensive test suite:
```bash
cd src
python -m pytest tests/test_activity1.py -v
# or
python -m unittest tests.test_activity1 -v
```

##  **Performance Results**

Current **Activity 1 (Basic Mode)** achievements:
- **Success Rate**: 100% (all persons rescued)
- **Battery Efficiency**: Optimized recharge cycles
- **Coordination**: Effective drone-robot communication
- **Coverage**: Complete mountain area exploration

##  **System Architecture**

### **Agent State Machines**
- **Terrain Robots**: `AT_BASE → SEARCHING → DELIVERING → RETURNING`
- **Explorer Drones**: `AT_BASE → EXPLORING → FOUND_PERSON → WAITING → RETURNING`

### **Core Components**
- `environment.py` - Simulation world and terrain management
- `terrain_robot.py` - Ground rescue vehicle agent
- `explorer_drone.py` - Aerial reconnaissance agent
- `simulation.py` - Main simulation controller
- `coordination.py` - Inter-agent communication system

##  **Project Roadmap**

- ** Activity 1**: Basic Mode operation and fundamental behaviors
- ** Activity 2**: Game theory analysis for agent interactions
- ** Activity 3**: Extended Mode with advanced communication
- ** Activity 4**: Novel features and machine learning integration
- ** Activity 5**: Comprehensive video demonstration

