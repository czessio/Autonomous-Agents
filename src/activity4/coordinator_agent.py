# Coordinator agent for strategic mission planning and resource allocation

from typing import List, Tuple, Dict, Optional
from enum import Enum
import numpy as np


class CoordinatorState(Enum):
    PLANNING = "planning"
    ALLOCATING = "allocating"
    MONITORING = "monitoring"
    OPTIMISING = "optimising"


class CoordinatorAgent:
    """
    Strategic coordinator that optimises multi-agent rescue operations.
    
    Key features:
    1. Dynamic role assignment for robots
    2. Priority-based rescue target allocation
    3. Battery management and recall decisions
    4. Performance monitoring and strategy adaptation
    5. Knowledge aggregation from all agents
    """
    
    def __init__(self, agent_id: int, position: Tuple[int, int]):
        self.agent_id = agent_id
        self.position = position  # Central command position
        self.state = CoordinatorState.PLANNING
        
        # Mission planning
        self.mission_priorities = {}  # location -> priority score
        self.robot_assignments = {}   # robot_id -> (location, role)
        self.rescue_queue = []        # Prioritised list of rescue targets
        
        # Performance tracking
        self.mission_stats = {
            'total_assigned': 0,
            'successful_rescues': 0,
            'failed_rescues': 0,
            'avg_response_time': 0,
            'battery_efficiency': 0
        }
        
        # Strategy parameters
        self.allocation_strategy = "nearest"  # "nearest", "battery_optimal", "learned"
        self.battery_threshold = 40  # Recall threshold
        self.urgency_weights = {
            'elevation': 0.3,
            'distance': 0.2,
            'waiting_time': 0.3,
            'robot_availability': 0.2
        }
        
        print(f"Coordinator Agent {agent_id} initialised at position {position}")
    
    def update(self, environment, robots, drones, communication, current_step=0):
        """Main update cycle for coordinator agent"""
        
        # State machine logic
        if self.state == CoordinatorState.PLANNING:
            self._plan_mission(environment, robots, drones, communication)
        elif self.state == CoordinatorState.ALLOCATING:
            self._allocate_resources(robots, communication)
        elif self.state == CoordinatorState.MONITORING:
            self._monitor_progress(robots, drones, communication)
        elif self.state == CoordinatorState.OPTIMISING:
            self._optimise_strategy(robots, communication)
        
        # Periodic strategy review
        if current_step % 50 == 0 and current_step > 0:
            self._review_performance()
    
    def _plan_mission(self, environment, robots, drones, communication):
        """Analyse situation and create mission plan"""
        # Get all pending rescues
        pending_locations = []
        
        # From communication system
        if hasattr(communication, 'found_persons'):
            for location, info in communication.found_persons.items():
                if not info.get('assigned', False):
                    pending_locations.append(location)
        
        # From environment
        for location in environment.missing_persons:
            if location not in pending_locations:
                pending_locations.append(location)
        
        if pending_locations:
            # Calculate priorities
            self.mission_priorities.clear()
            for location in pending_locations:
                priority = self._calculate_priority(location, robots, environment)
                self.mission_priorities[location] = priority
            
            # Sort by priority
            self.rescue_queue = sorted(
                self.mission_priorities.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            self.state = CoordinatorState.ALLOCATING
        else:
            self.state = CoordinatorState.MONITORING
    
    def _calculate_priority(self, location: Tuple[int, int], robots: List, 
                          environment) -> float:
        """Calculate rescue priority using multiple factors"""
        priority = 0.0
        
        # Factor 1: Elevation (higher = more urgent)
        elevation = environment.get_elevation(*location)
        priority += (elevation / 3000) * self.urgency_weights['elevation'] * 100
        
        # Factor 2: Average distance from available robots
        available_robots = [r for r in robots 
                           if r.state.value == 'at_base' 
                           and r.has_first_aid_kit
                           and r.current_battery > self.battery_threshold]
        
        if available_robots:
            avg_distance = np.mean([
                abs(r.position[0] - location[0]) + abs(r.position[1] - location[1])
                for r in available_robots
            ])
            # Closer targets get higher priority
            priority += (20 - avg_distance) * self.urgency_weights['distance']
        
        # Factor 3: Robot availability
        availability_score = len(available_robots) / max(1, len(robots))
        priority += availability_score * self.urgency_weights['robot_availability'] * 50
        
        return priority
    
    def _allocate_resources(self, robots, communication):
        """Allocate robots to rescue targets based on strategy"""
        allocated_count = 0
        
        # Get available robots
        available_robots = [
            r for r in robots
            if r.state.value == 'at_base'
            and r.has_first_aid_kit
            and r.current_battery > self.battery_threshold
            and r.robot_id not in self.robot_assignments
        ]
        
        # Allocate based on strategy
        for location, priority in self.rescue_queue:
            if not available_robots:
                break
            
            # Skip if already assigned
            if hasattr(communication, 'assigned_rescues'):
                if location in communication.assigned_rescues.values():
                    continue
            
            # Find best robot for this location
            best_robot = self._select_best_robot(location, available_robots)
            
            if best_robot:
                # Send allocation message
                if hasattr(communication, 'send_individual_message'):
                    communication.send_individual_message(
                        sender_id=f"coordinator_{self.agent_id}",
                        recipient_id=f"robot_{best_robot.robot_id}",
                        msg_type=communication.MessageType.RESCUE_REQUEST,
                        location=location,
                        urgency=int(priority),
                        data={'role': 'rescuer', 'priority': priority}
                    )
                
                # Track assignment
                self.robot_assignments[best_robot.robot_id] = (location, 'rescuer')
                available_robots.remove(best_robot)
                allocated_count += 1
                self.mission_stats['total_assigned'] += 1
                
                print(f"Coordinator assigned Robot {best_robot.robot_id} to rescue at {location} (priority: {priority:.1f})")
        
        if allocated_count > 0:
            self.state = CoordinatorState.MONITORING
        else:
            # No allocations possible, monitor situation
            self.state = CoordinatorState.MONITORING
    
    def _select_best_robot(self, location: Tuple[int, int], 
                          available_robots: List) -> Optional[object]:
        """Select optimal robot for a rescue location"""
        if self.allocation_strategy == "nearest":
            # Simple nearest robot
            return min(available_robots, 
                      key=lambda r: abs(r.position[0] - location[0]) + 
                                   abs(r.position[1] - location[1]))
        
        elif self.allocation_strategy == "battery_optimal":
            # Consider battery and distance
            best_robot = None
            best_score = float('-inf')
            
            for robot in available_robots:
                distance = abs(robot.position[0] - location[0]) + abs(robot.position[1] - location[1])
                battery_needed = distance * 2 + 20  # Estimate
                
                if robot.current_battery >= battery_needed:
                    # Score based on battery efficiency
                    score = (robot.current_battery - battery_needed) - distance * 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_robot = robot
            
            return best_robot
        
        elif self.allocation_strategy == "learned":
            # Use Q-learning knowledge if available
            best_robot = None
            best_score = float('-inf')
            
            for robot in available_robots:
                score = 0
                
                # Base score on distance
                distance = abs(robot.position[0] - location[0]) + abs(robot.position[1] - location[1])
                score -= distance
                
                # Bonus if robot knows this location
                if hasattr(robot, 'q_agent') and hasattr(robot.q_agent, 'rescue_success_map'):
                    if location in robot.q_agent.rescue_success_map:
                        score += 20  # Familiar location bonus
                
                # Battery consideration
                score += robot.current_battery * 0.1
                
                if score > best_score:
                    best_score = score
                    best_robot = robot
            
            return best_robot
        
        return None
    
    def _monitor_progress(self, robots, drones, communication):
        """Monitor ongoing rescue operations"""
        # Check assigned robots
        completed_assignments = []
        
        for robot_id, (location, role) in self.robot_assignments.items():
            robot = next((r for r in robots if r.robot_id == robot_id), None)
            
            if robot:
                # Check if rescue completed
                if robot.state.value == 'returning' or robot.state.value == 'at_base':
                    if location not in [r.position for r in robots]:
                        # Assume rescue completed
                        self.mission_stats['successful_rescues'] += 1
                        completed_assignments.append(robot_id)
                
                # Check if robot needs recall
                elif robot.current_battery < 30:
                    print(f"Coordinator: Recalling Robot {robot_id} due to low battery")
                    # Could send recall message here
                    completed_assignments.append(robot_id)
        
        # Remove completed assignments
        for robot_id in completed_assignments:
            del self.robot_assignments[robot_id]
        
        # Check if need to replan
        if len(self.robot_assignments) == 0:
            self.state = CoordinatorState.PLANNING
        
        # Periodic optimisation check
        if self.mission_stats['total_assigned'] > 0 and self.mission_stats['total_assigned'] % 10 == 0:
            self.state = CoordinatorState.OPTIMISING
    
    def _optimise_strategy(self, robots, communication):
        """Optimise allocation strategy based on performance"""
        # Calculate current efficiency
        if self.mission_stats['total_assigned'] > 0:
            success_rate = self.mission_stats['successful_rescues'] / self.mission_stats['total_assigned']
            
            # Adjust strategy based on performance
            if success_rate < 0.5:
                # Poor performance, try different strategy
                if self.allocation_strategy == "nearest":
                    self.allocation_strategy = "battery_optimal"
                    print("Coordinator: Switching to battery-optimal allocation")
                elif self.allocation_strategy == "battery_optimal":
                    self.allocation_strategy = "learned"
                    print("Coordinator: Switching to learned allocation")
            
            # Adjust battery threshold
            if self.mission_stats['failed_rescues'] > 2:
                self.battery_threshold = min(60, self.battery_threshold + 10)
                print(f"Coordinator: Increased battery threshold to {self.battery_threshold}")
        
        self.state = CoordinatorState.PLANNING
    
    def _review_performance(self):
        """Periodic performance review and reporting"""
        if self.mission_stats['total_assigned'] > 0:
            success_rate = self.mission_stats['successful_rescues'] / self.mission_stats['total_assigned']
            print(f"\nCoordinator Performance Review:")
            print(f"  Assignments: {self.mission_stats['total_assigned']}")
            print(f"  Success Rate: {success_rate:.1%}")
            print(f"  Strategy: {self.allocation_strategy}")
            print(f"  Active Assignments: {len(self.robot_assignments)}")
    
    def get_status(self) -> Dict:
        """Get coordinator status for monitoring"""
        return {
            'id': self.agent_id,
            'state': self.state.value,
            'strategy': self.allocation_strategy,
            'active_assignments': len(self.robot_assignments),
            'queue_size': len(self.rescue_queue),
            'success_rate': (self.mission_stats['successful_rescues'] / 
                           max(1, self.mission_stats['total_assigned']) * 100)
        }