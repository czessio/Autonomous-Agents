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
├── src/
│   ├── activity1/
│   │   ├── environment.py       # Environment and world setup
│   │   ├── terrain_robot.py     # First-aid terrain robot agent
│   │   ├── explorer_drone.py    # Explorer drone agent
│   │   ├── coordination.py      # Communication system
│   │   └── simulation.py        # Main simulation controller
│   └── tests/
│       └── test_activity1.py    # Unit tests
├── requirements.txt             # Project dependencies
└── solution.md                  # This file
```

## Running the Simulation

### Activity 1: Basic Mode Simulation

Navigate to the activity1 directory and run the simulation:

```bash
cd src/activity1
python simulation.py
```

This will launch the simulation with visualisation showing:
- Mountain terrain with elevation zones (0, 1K, 2K, 3K MASL)
- Missing persons scattered across the mountain
- Terrain robots (with tank-like appearance) delivering first-aid
- Explorer drones (with quadcopter design) searching for persons
- Real-time status information and statistics

### Running Tests

Execute the unit tests to verify system functionality:

```bash
cd src
python tests/test_activity1.py
```

## Unit Test Cases for Activity 1

The comprehensive test suite validates all core functionality with 25 individual test cases:

```
Running Mountain Rescue Simulation Tests...
==================================================
 1. Test elevation zones are set up with multiple height levels
 2. Test environment initialisation with correct dimensions and agent counts
 3. Test rescuing at empty location returns False
 4. Test mountain area boundaries are properly defined
 5. Test missing persons are correctly placed within mountain boundaries
 6. Test person rescue mechanism works correctly
 7. Test drone battery decreases during flight operations
 8. Test drone starts with correct initial values
 9. Test drone properly resets mission data when returning to base
10. Test drone detects persons and changes state appropriately
11. Test drone position validation works correctly
12. Test drone generates systematic search patterns
13. Test drone has 6-directional movement capability as required
14. Test complete simulation runs without errors
15. Test agents are created at their designated base positions
16. Test simulation detects mission completion correctly
17. Test simulation creates correct number of agents and environment
18. Test single simulation step executes without errors
19. Test simulation properly tracks performance statistics
20. Test battery decreases when robot operates away from base
21. Test robot tracks its movement history correctly
22. Test robot can successfully rescue persons
23. Test robot starts with correct initial values
24. Test robot state machine transitions work correctly
25. Test position validation for environment boundaries
==================================================
TEST SUMMARY
==================================================
ALL TESTS PASSED: 25/25
Mountain Rescue System is ready for operation!
==================================================
```

### Test Coverage Areas

The test suite comprehensively validates:

- **Environment Setup (Tests 1-6)**: Terrain generation, elevation zones, person placement, and rescue mechanics
- **Drone Functionality (Tests 7-13)**: Battery management, 6-directional movement, search patterns, and person detection
- **Simulation Management (Tests 14-19)**: Agent coordination, statistics tracking, and mission completion detection
- **Robot Functionality (Tests 20-25)**: Ground navigation, state transitions, rescue capability, and battery management

## System Features (Activity 1)

### Environment
- Grid-based mountain terrain with realistic elevation zones (0, 1K, 2K, 3K MASL)
- Configurable number of missing persons, robots, and drones
- Visual representation matching project specifications
- Professional animated visualization with realistic agent icons

### Terrain Robots
- **State Machine**: AT_BASE → SEARCHING → DELIVERING → RETURNING → AT_BASE
- **Battery Management**: Depletes during operation, increased drain at higher elevations
- **Communication Response**: Responds to drone-found person locations
- **Mountain-Focused Search**: Prioritizes mountain areas for random exploration
- **First-Aid Delivery**: Delivers aid when person is found
- **Autonomous Return**: Returns to base when mission complete or battery low

### Explorer Drones
- **State Machine**: AT_BASE → EXPLORING → FOUND_PERSON → WAITING → RETURNING → AT_BASE
- **6-Directional Movement**: Front, back, left, right, up, down movement capability
- **Systematic Search**: TOP/BOTTOM starting positions for efficient coverage
- **Person Detection**: Identifies missing persons and broadcasts locations
- **Waiting Behaviour**: Waits for terrain robots after finding persons
- **Memory System**: Avoids revisiting previously explored areas

### Key Behaviours (Basic Mode Compliance)
- **Exploration and movement towards mountain**: Both agents prioritize mountain areas
- **Drones locate and wait**: Drones find persons and wait for robots as specified
- **Robots move randomly**: Terrain robots use random search in Basic Mode
- **Battery consumption**: All agents have realistic battery management
- **Return to base stations**: Both agent types return to recharge after operations

## Agent Design Documentation

### Terrain Robot States and Transitions
1. **AT_BASE**: Recharging and restocking first-aid kits
   - *Transition*: Battery > 90% → SEARCHING
2. **SEARCHING**: Random movement prioritizing mountain areas
   - *Transition*: Person found → DELIVERING
   - *Transition*: Battery < 20% → BATTERY_LOW
   - *Transition*: Communication assignment → DELIVERING
3. **DELIVERING**: Moving to and rescuing found person
   - *Transition*: Person rescued → RETURNING
4. **RETURNING**: Moving back to base station
   - *Transition*: At base → AT_BASE
5. **BATTERY_LOW**: Emergency return to base
   - *Transition*: At base → AT_BASE

### Explorer Drone States and Transitions
1. **AT_BASE**: Recharging and mission planning
   - *Transition*: Battery > 90% → EXPLORING
2. **EXPLORING**: Systematic search pattern execution
   - *Transition*: Person found → FOUND_PERSON
   - *Transition*: Battery < 20% → BATTERY_LOW
3. **FOUND_PERSON**: Initial person detection and assessment
   - *Transition*: Assessment complete → WAITING
4. **WAITING**: Waiting for terrain robot arrival
   - *Transition*: Person rescued → RETURNING
   - *Transition*: Timeout → EXPLORING
5. **RETURNING**: Moving back to base station
   - *Transition*: At base → AT_BASE
6. **BATTERY_LOW**: Emergency return to base
   - *Transition*: At base → AT_BASE

### Decision Making Logic
- **Terrain Robots**: Prioritize drone-reported locations, fallback to random mountain search
- **Explorer Drones**: Follow systematic search patterns, avoid previously explored areas
- **Battery Management**: Conservative thresholds ensure successful return to base
- **Communication**: Simple broadcast system for person location sharing

## Performance Metrics

The simulation tracks:
- Total persons rescued
- Mission duration (steps and real-time)
- Battery consumption efficiency
- Individual agent performance
- Success rate calculation

## Technical Implementation

### Agent Architecture
- Finite State Machines for behaviour control
- Modular design with clear separation of concerns
- Comprehensive error handling and boundary checking

### Visualisation
- Real-time matplotlib animation
- Status monitoring and progress tracking
- Professional presentation matching specifications

### Testing
- Comprehensive unit test coverage (25 test cases)
- Integration testing for multi-agent interactions
- Automated validation of core functionality

## Acknowledgements

### Use of Generative AI Tools

This solution was developed with assistance from Claude 3.7 Sonnet (Anthropic) for:
- Code structure and implementation guidance
- Algorithm optimisation suggestions
- Documentation and comment generation
- Testing strategy development

All generated code was thoroughly reviewed, tested, and adapted to meet specific project requirements. The core logic, state machine designs, and system architecture decisions were made through human analysis of the project specifications.