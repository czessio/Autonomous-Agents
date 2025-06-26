# Mediator agent for coordinating rescue operations and resource allocation

from typing import List, Tuple, Dict
from enum import Enum
import numpy as np

class MediatorState(Enum):
    MONITORING = "monitoring"
    ALLOCATING = "allocating"
    COORDINATING = "coordinating"
    EMERGENCY = "emergency"

class MediatorAgent:
    """
    Advanced coordination agent that optimises rescue operations.
    
    Key responsibilities:
    1. Prioritise rescue targets based on urgency and distance
    2. Allocate robots efficiently to minimise total rescue time
    3. Coordinate battery management and agent recalls
    4. Learn optimal allocation strategies over time
    """
    
    def __init__(self, agent_id: int, position: Tuple[int, int]):
        self.agent_id = agent_id
        self.position = position  # Central command position
        self.state = MediatorState.MONITORING
        
        # Tracking systems
        self.mission_priorities = {}  # location -> priority score
        self.robot_assignments = {}   # robot_id -> assignment
        self.rescue_history = []
        self.allocation_success_rate = 0.0
        
        # Learning components
        self.allocation_patterns = {}  # Successful allocation strategies
        self.failed_allocations = []
        
    def update(self, environment, robots, drones, communication):
        """Main update cycle for mediator agent"""
        if self.state == MediatorState.MONITORING:
            self._monitor_situation(environment, robots, drones, communication)
        elif self.state == MediatorState.ALLOCATING:
            self._allocate_resources(robots, communication)
        elif self.state == MediatorState.COORDINATING:
            self._coordinate_rescue(robots, drones, communication)
        elif self.state == MediatorState.EMERGENCY:
            self._handle_emergency(robots, drones, communication)
    
    def _monitor_situation(self, environment, robots, drones, communication):
        """Monitor overall mission status and identify priorities"""
        # Check for unassigned rescue targets
        pending_rescues = communication.get_pending_rescues()
        
        if pending_rescues:
            # Calculate priorities for each pending rescue
            for location in pending_rescues:
                priority = self._calculate_priority(location, robots, environment)
                self.mission_priorities[location] = priority
            
            self.state = MediatorState.ALLOCATING
        
        # Check for low battery situations
        low_battery_agents = [r for r in robots if r.current_battery < 30]
        if len(low_battery_agents) > len(robots) * 0.5:
            self.state = MediatorState.EMERGENCY
    
    def _calculate_priority(self, location, robots, environment):
        """Calculate rescue priority based on multiple factors"""
        priority = 100.0
        
        # Factor 1: Elevation (higher elevation = higher priority)
        elevation = environment.get_elevation(*location)
        priority += elevation / 100
        
        # Factor 2: Distance to nearest available robot
        min_distance = float('inf')
        for robot in robots:
            if robot.state.value == 'at_base' and robot.has_first_aid_kit:
                distance = abs(robot.position[0] - location[0]) + abs(robot.position[1] - location[1])
                min_distance = min(min_distance, distance)
        
        if min_distance < float('inf'):
            priority -= min_distance * 2  # Closer targets get higher priority
        
        # Factor 3: Time waiting (if tracked)
        # Could add urgency from drone assessment here
        
        return priority
    
    def _allocate_resources(self, robots, communication):
        """Intelligent resource allocation based on priorities"""
        # Sort locations by priority
        sorted_locations = sorted(self.mission_priorities.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        available_robots = [r for r in robots 
                           if r.state.value == 'at_base' 
                           and r.has_first_aid_kit 
                           and r.current_battery > 50]
        
        # Allocate robots to highest priority targets
        for location, priority in sorted_locations:
            if not available_robots:
                break
            
            # Find best robot for this location
            best_robot = self._find_best_robot(location, available_robots)
            if best_robot:
                # Send individual assignment
                communication.send_individual_message(
                    sender_id=f"mediator_{self.agent_id}",
                    recipient_id=f"robot_{best_robot.robot_id}",
                    msg_type=communication.MessageType.RESCUE_REQUEST,
                    location=location,
                    urgency=int(priority)
                )
                
                self.robot_assignments[best_robot.robot_id] = location
                available_robots.remove(best_robot)
        
        self.state = MediatorState.COORDINATING
    
    def _find_best_robot(self, location, available_robots):
        """Select optimal robot for a rescue location"""
        best_robot = None
        best_score = float('-inf')
        
        for robot in available_robots:
            # Calculate suitability score
            distance = abs(robot.position[0] - location[0]) + abs(robot.position[1] - location[1])
            battery_score = robot.current_battery / 100
            
            # Estimated battery needed
            battery_needed = distance * 2 + 20  # Round trip + rescue
            
            if robot.current_battery > battery_needed:
                score = battery_score * 100 - distance
                
                # Bonus for robots with Q-learning experience
                if hasattr(robot, 'q_agent') and location in robot.q_agent.rescue_success_map:
                    score += 20  # Familiar with location
                
                if score > best_score:
                    best_score = score
                    best_robot = robot
        
        return best_robot
    
    def get_status(self) -> Dict:
        """Get mediator status for monitoring"""
        return {
            'id': self.agent_id,
            'state': self.state.value,
            'active_assignments': len(self.robot_assignments),
            'pending_priorities': len(self.mission_priorities),
            'allocation_success': f"{self.allocation_success_rate:.1%}"
        }