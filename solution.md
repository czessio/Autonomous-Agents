# Mountain Rescue Robots - Multi-Agent System



## Setup Instructions

### 1. Environment Setup

Create and activate a Python virtual environment:

```bash
# Create virtual environment
python -m venv mountain-rescueR

# Activate (Linux/Mac)
source mountain-rescue/bin/activate

# Activate (Windows)
.\mountain-rescue\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
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
- Terrain robots (white squares with red crosses) delivering first-aid
- Explorer drones (black squares) searching for persons
- Real-time status information and statistics

### Running Tests

Execute the unit tests to verify system functionality:

```bash
cd src
python -m pytest tests/test_activity1.py -v
```

Or using unittest:

```bash
cd src
python -m unittest tests.test_activity1 -v
```

## System Features (Activity 1)

### Environment
- Grid-based mountain terrain with realistic elevation zones
- Configurable number of missing persons, robots, and drones
- Visual representation matching project specifications

### Terrain Robots
- **State Machine**: AT_BASE → SEARCHING → DELIVERING → RETURNING → AT_BASE
- **Battery Management**: Depletes during operation, increased drain at higher elevations
- **Random Search Pattern**: Moves randomly to locate missing persons
- **First-Aid Delivery**: Delivers aid when person is found
- **Autonomous Return**: Returns to base when mission complete or battery low

### Explorer Drones
- **State Machine**: AT_BASE → EXPLORING → FOUND_PERSON → WAITING → RETURNING → AT_BASE
- **6-Directional Movement**: Front, back, left, right, up, down movement capability
- **Systematic Search**: Generates search patterns for efficient area coverage
- **Person Detection**: Identifies missing persons and calculates urgency
- **Waiting Behaviour**: Waits for terrain robots after finding persons

### Key Behaviours
- **Basic Mode Operation**: Drones explore and wait, robots search randomly
- **Battery Management**: All agents monitor power levels and return when low
- **Collision Avoidance**: Agents avoid boundaries and track movement history
- **Mission Coordination**: Autonomous decision-making without communication

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
- Comprehensive unit test coverage
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

## Future Development

This foundation supports extension to:
- Activity 2: Game theory analysis
- Activity 3: Extended mode with communication
- Activity 4: Novel features and learning agents
- Activity 5: Comprehensive evaluation and demonstration

