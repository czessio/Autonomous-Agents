# Mountain Rescue Agents - State Machine Design Document

## Overview

This document describes the finite state machine designs for the Mountain Rescue multi-agent system. Each agent type follows a well-defined state machine to ensure predictable, reliable behavior during rescue operations.

---

## 🤖 Terrain Robot State Machine

### States

#### **AT_BASE**
- **Purpose**: Recharging and restocking first-aid kits
- **Behavior**: 
  - Recharge battery at 10% per step
  - Restock first-aid kit when available
  - Monitor battery level and mission assignments
- **Entry Conditions**: 
  - Robot returns to base position
  - Mission completed or battery depleted
- **Exit Conditions**: 
  - Battery > 90% AND has first-aid kit → SEARCHING

#### **SEARCHING** 
- **Purpose**: Locate missing persons through exploration
- **Behavior**:
  - Move randomly with mountain area priority
  - Respond to drone communications
  - Check for persons at current location
  - Avoid recently visited positions
- **Entry Conditions**:
  - Left base with sufficient battery
  - Assigned new target location
- **Exit Conditions**:
  - Person found at location → DELIVERING
  - Battery < 20% → BATTERY_LOW
  - Received target assignment → DELIVERING

#### **DELIVERING**
- **Purpose**: Move to and rescue found person
- **Behavior**:
  - Navigate directly to target person
  - Deliver first-aid kit upon arrival
  - Update rescue statistics
- **Entry Conditions**:
  - Person detected at current location
  - Assigned to rescue specific person
- **Exit Conditions**:
  - Person successfully rescued → RETURNING
  - Battery < 20% → BATTERY_LOW

#### **RETURNING**
- **Purpose**: Navigate back to base station
- **Behavior**:
  - Move directly toward base position
  - Conserve battery during return
- **Entry Conditions**:
  - Mission completed (person rescued)
  - No first-aid kit remaining
- **Exit Conditions**:
  - Arrived at base → AT_BASE

#### **BATTERY_LOW**
- **Purpose**: Emergency return when battery critically low
- **Behavior**:
  - Priority movement toward base
  - Abandon current mission
  - Ignore new assignments
- **Entry Conditions**:
  - Battery ≤ 20% while away from base
- **Exit Conditions**:
  - Arrived at base → AT_BASE

### Transition Rules

```
AT_BASE → SEARCHING: battery > 90% AND has_first_aid_kit
SEARCHING → DELIVERING: person_found OR assigned_target
SEARCHING → BATTERY_LOW: battery ≤ 20%
DELIVERING → RETURNING: person_rescued
DELIVERING → BATTERY_LOW: battery ≤ 20%
RETURNING → AT_BASE: at_base_position
BATTERY_LOW → AT_BASE: at_base_position
```

### Decision Logic

1. **Battery Management**: Conservative 20% threshold ensures safe return
2. **Mission Priority**: Person rescue takes precedence over exploration
3. **Communication Response**: Immediate response to drone-found persons
4. **Exploration Strategy**: Mountain-focused random search in Basic Mode

---

## 🚁 Explorer Drone State Machine

### States

#### **AT_BASE**
- **Purpose**: Recharging and mission planning
- **Behavior**:
  - Recharge battery at 15% per step
  - Generate systematic search patterns
  - Reset mission data from previous flights
- **Entry Conditions**:
  - Drone returns to base position
  - Mission completed or battery depleted
- **Exit Conditions**:
  - Battery > 90% → EXPLORING

#### **EXPLORING**
- **Purpose**: Systematic search for missing persons
- **Behavior**:
  - Follow predetermined search pattern
  - Use 6-directional movement capability
  - Mark explored positions to avoid revisiting
  - Prioritize mountain areas
- **Entry Conditions**:
  - Left base with sufficient battery
  - Search pattern generated
- **Exit Conditions**:
  - Person found → FOUND_PERSON
  - Battery < 20% → BATTERY_LOW
  - Search pattern complete → RETURNING

#### **FOUND_PERSON**
- **Purpose**: Initial person detection and assessment
- **Behavior**:
  - Assess person urgency using computer vision
  - Broadcast person location to robots
  - Prepare for waiting phase
- **Entry Conditions**:
  - Person detected at current location
  - First time finding this person
- **Exit Conditions**:
  - Assessment complete → WAITING

#### **WAITING**
- **Purpose**: Guard person location until robot arrives
- **Behavior**:
  - Maintain position at person location
  - Continue broadcasting location
  - Monitor for robot arrival
  - Track waiting time
- **Entry Conditions**:
  - Person assessment completed
  - Location broadcast to robots
- **Exit Conditions**:
  - Person rescued by robot → RETURNING
  - Waiting timeout (15 steps) → EXPLORING
  - Battery < 20% → BATTERY_LOW

#### **RETURNING**
- **Purpose**: Navigate back to base station
- **Behavior**:
  - Move directly toward base using 6-directional movement
  - Conserve battery during return
- **Entry Conditions**:
  - Mission completed (person rescued)
  - Search pattern completed
  - Waiting timeout exceeded
- **Exit Conditions**:
  - Arrived at base → AT_BASE

#### **BATTERY_LOW**
- **Purpose**: Emergency return when battery critically low
- **Behavior**:
  - Priority movement toward base
  - Abandon current mission
  - Stop broadcasting person locations
- **Entry Conditions**:
  - Battery ≤ 20% while away from base
- **Exit Conditions**:
  - Arrived at base → AT_BASE

### Transition Rules

```
AT_BASE → EXPLORING: battery > 90%
EXPLORING → FOUND_PERSON: person_detected AND first_time_finding
EXPLORING → BATTERY_LOW: battery ≤ 20%
FOUND_PERSON → WAITING: assessment_complete
WAITING → RETURNING: person_rescued OR timeout_exceeded
WAITING → BATTERY_LOW: battery ≤ 20%
RETURNING → AT_BASE: at_base_position
BATTERY_LOW → AT_BASE: at_base_position
```

### Decision Logic

1. **Search Efficiency**: TOP/BOTTOM starting positions for different drones
2. **Memory System**: Avoid revisiting explored areas
3. **Communication**: Immediate broadcast when person found
4. **Patience**: Wait reasonable time for robot arrival
5. **Battery Conservation**: 20% threshold for safe return

---

## 🔄 Communication Integration

### Basic Mode Communication Flow

1. **Drone finds person** → Broadcasts location to all robots
2. **Robots receive location** → Nearest available robot assigned
3. **Robot navigates to location** → Drone continues waiting
4. **Robot rescues person** → Drone stops waiting and returns
5. **System updates** → Location removed from active targets

### Coordination Rules

- **One robot per person**: Prevent multiple robots targeting same person
- **Nearest assignment**: Minimize travel time and battery usage
- **Timeout handling**: Drone resumes exploration if robot doesn't arrive
- **Status synchronization**: All agents aware of mission progress

---

## 🎯 Design Principles

### Reliability
- **Conservative battery thresholds** ensure agents can always return to base
- **Clear state transitions** prevent agents from getting stuck
- **Timeout mechanisms** handle communication failures

### Efficiency  
- **Systematic search patterns** maximize area coverage
- **Smart assignment logic** minimizes robot travel time
- **Memory systems** prevent redundant exploration

### Modularity
- **Independent state machines** allow individual agent testing
- **Clean interfaces** between states enable easy modification
- **Separation of concerns** between movement, communication, and mission logic

### Scalability
- **Configurable parameters** (battery thresholds, timeouts) for different scenarios
- **Modular communication** system supports additional agent types
- **Pattern-based search** easily extended for larger environments

---

## 🔧 Implementation Notes

### State Persistence
- Each agent maintains current state as enum value
- State transitions logged for debugging and analysis
- State-specific data (targets, timers) managed separately

### Error Handling
- Invalid state transitions prevented by guard conditions
- Boundary checking for all movement operations
- Graceful degradation when communication fails

### Performance Optimization
- Efficient pathfinding using Manhattan distance
- Minimal computation in tight simulation loops
- Smart data structures for position tracking

This state machine design ensures robust, predictable behavior while maintaining flexibility for future enhancements in Activities 3 and 4.