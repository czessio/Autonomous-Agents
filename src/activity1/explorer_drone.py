# Explorer drone agent for mountain rescue simulation - handles aerial exploration and person detection

import random
from enum import Enum
from typing import Tuple, Optional, List


class DroneState(Enum):
    AT_BASE = "at_base"
    EXPLORING = "exploring"
    FOUND_PERSON = "found_person"
    WAITING = "waiting"
    RETURNING = "returning"
    BATTERY_LOW = "battery_low"


class ExplorerDrone:
    def __init__(self, drone_id: int, start_position: Tuple[int, int], 
                 max_battery: int = 120, battery_drain_rate: int = 2):
        self.drone_id = drone_id
        self.position = start_position
        self.base_position = start_position
        self.state = DroneState.AT_BASE
        
        # Battery management (drones use more battery due to flight)
        self.max_battery = max_battery
        self.current_battery = max_battery
        self.battery_drain_rate = battery_drain_rate
        self.low_battery_threshold = 25
        
        # Mission tracking
        self.found_persons = []  # List of persons this drone has found
        self.current_target = None
        self.waiting_time = 0
        self.max_waiting_time = 10  # Time steps to wait after finding person
        
        # Exploration tracking
        self.explored_positions = set()
        self.search_pattern = []
        self.pattern_index = 0
        
        # 6-directional movement capability
        self.movement_directions = [
            (1, 0),   # front (right)
            (-1, 0),  # back (left)
            (0, 1),   # up
            (0, -1),  # down
            (1, 1),   # front-up (diagonal)
            (-1, -1)  # back-down (diagonal)
        ]
    
    def update(self, environment) -> None:
        """Main update method called each simulation step"""
        # Drain battery
        self._drain_battery()
        
        # Check battery level
        if self._is_battery_low() and self.state != DroneState.BATTERY_LOW:
            self.state = DroneState.BATTERY_LOW
            self.current_target = None
            self.waiting_time = 0
        
        # Execute behaviour based on current state
        if self.state == DroneState.AT_BASE:
            self._handle_at_base_state(environment)
        elif self.state == DroneState.EXPLORING:
            self._handle_exploring_state(environment)
        elif self.state == DroneState.FOUND_PERSON:
            self._handle_found_person_state(environment)
        elif self.state == DroneState.WAITING:
            self._handle_waiting_state(environment)
        elif self.state == DroneState.RETURNING:
            self._handle_returning_state(environment)
        elif self.state == DroneState.BATTERY_LOW:
            self._handle_battery_low_state(environment)
    
    def _drain_battery(self) -> None:
        """Drain battery for flight operations"""
        self.current_battery = max(0, self.current_battery - self.battery_drain_rate)
    
    def _is_battery_low(self) -> bool:
        """Check if battery is below low threshold"""
        return self.current_battery <= self.low_battery_threshold
    
    def _handle_at_base_state(self, environment) -> None:
        """Handle behaviour when drone is at base station"""
        if self.current_battery < self.max_battery:
            # Recharge battery
            self.current_battery = min(self.max_battery, self.current_battery + 8)
        
        if self.current_battery >= 40:
            # Ready to start mission (lowered threshold so drones start sooner)
            self._generate_search_pattern(environment)
            self.state = DroneState.EXPLORING
    
    def _handle_exploring_state(self, environment) -> None:
        """Handle exploration behaviour using 6-directional movement"""
        # Check if person found at current location
        if environment.person_at_location(*self.position):
            if self.position not in self.found_persons:
                self.found_persons.append(self.position)
                self.current_target = self.position
                self.state = DroneState.FOUND_PERSON
                return
        
        # Mark current position as explored
        self.explored_positions.add(self.position)
        
        # Move according to search pattern or randomly
        self._execute_movement(environment)
    
    def _handle_found_person_state(self, environment) -> None:
        """Handle immediate response when person is found"""
        # Take pictures and assess urgency (simulated)
        person_urgency = self._assess_person_urgency()
        
        # Move to waiting state
        self.waiting_time = 0
        self.state = DroneState.WAITING
    
    def _handle_waiting_state(self, environment) -> None:
        """Handle waiting behaviour after finding a person"""
        self.waiting_time += 1
        
        # Check if terrain robot has arrived
        terrain_robot_nearby = self._check_for_terrain_robot_nearby(environment)
        
        if terrain_robot_nearby or self.waiting_time >= self.max_waiting_time:
            # Either robot arrived or waited long enough
            self.current_target = None
            
            # Decide next action based on battery and mission progress
            if len(self.found_persons) >= 3 or self._is_battery_low():
                self.state = DroneState.RETURNING
            else:
                self.state = DroneState.EXPLORING
    
    def _handle_returning_state(self, environment) -> None:
        """Handle return to base station"""
        if self.position == self.base_position:
            # Arrived at base
            self._reset_mission_data()
            self.state = DroneState.AT_BASE
        else:
            # Move towards base using 6-directional movement
            self._move_towards_target(self.base_position, environment)
    
    def _handle_battery_low_state(self, environment) -> None:
        """Handle emergency return when battery is low"""
        if self.position == self.base_position:
            # At base, can recharge
            self._reset_mission_data()
            self.state = DroneState.AT_BASE
        else:
            # Move towards base urgently
            self._move_towards_target(self.base_position, environment)
    
    def _generate_search_pattern(self, environment) -> None:
        """Generate efficient search pattern for the mountain area"""
        self.search_pattern = []
        
        # Focus on mountain area
        if hasattr(environment, 'mountain_area'):
            x_start, y_start, x_end, y_end = environment.mountain_area['bounds']
            
            # Create systematic search pattern
            for y in range(y_start, y_end, 2):
                for x in range(x_start, x_end, 2):
                    self.search_pattern.append((x, y))
            
            # Add some random exploration points
            for _ in range(5):
                x = random.randint(x_start, x_end - 1)
                y = random.randint(y_start, y_end - 1)
                self.search_pattern.append((x, y))
        
        self.pattern_index = 0
    
    def _execute_movement(self, environment) -> None:
        """Execute movement using 6-directional capability"""
        # Try to follow search pattern first
        if self.pattern_index < len(self.search_pattern):
            target = self.search_pattern[self.pattern_index]
            if self.position == target or self._move_towards_target(target, environment):
                self.pattern_index += 1
        else:
            # Random exploration using 6 directions
            self._move_in_random_direction(environment)
    
    def _move_towards_target(self, target: Tuple[int, int], environment) -> bool:
        """Move one step towards target using 6-directional movement"""
        if not target or self.position == target:
            return True
        
        current_x, current_y = self.position
        target_x, target_y = target
        
        # Calculate best direction from the 6 available
        best_direction = None
        min_distance = float('inf')
        
        for dx, dy in self.movement_directions:
            new_x, new_y = current_x + dx, current_y + dy
            
            if self._is_valid_position((new_x, new_y), environment):
                distance = abs(new_x - target_x) + abs(new_y - target_y)
                if distance < min_distance:
                    min_distance = distance
                    best_direction = (dx, dy)
        
        # Move in best direction
        if best_direction:
            dx, dy = best_direction
            new_position = (current_x + dx, current_y + dy)
            self.position = new_position
            return self.position == target
        
        return False
    
    def _move_in_random_direction(self, environment) -> None:
        """Move randomly using one of the 6 directions"""
        # Prefer unexplored areas
        unexplored_moves = []
        all_valid_moves = []
        
        for dx, dy in self.movement_directions:
            new_x, new_y = self.position[0] + dx, self.position[1] + dy
            new_position = (new_x, new_y)
            
            if self._is_valid_position(new_position, environment):
                all_valid_moves.append(new_position)
                if new_position not in self.explored_positions:
                    unexplored_moves.append(new_position)
        
        # Choose move
        if unexplored_moves:
            self.position = random.choice(unexplored_moves)
        elif all_valid_moves:
            self.position = random.choice(all_valid_moves)
    
    def _is_valid_position(self, position: Tuple[int, int], environment) -> bool:
        """Check if position is valid (within bounds)"""
        x, y = position
        return 0 <= x < environment.width and 0 <= y < environment.height
    
    def _assess_person_urgency(self) -> int:
        """Simulate computer vision assessment of person urgency"""
        # Simulate age and health assessment
        # Returns urgency score from 1-10
        return random.randint(3, 9)
    
    def _check_for_terrain_robot_nearby(self, environment) -> bool:
        """Check if a terrain robot is nearby (simulated)"""
        # In basic mode, this is simplified
        # Just simulate that robots eventually arrive
        return random.random() < 0.3  # 30% chance per time step
    
    def _reset_mission_data(self) -> None:
        """Reset mission-specific data when returning to base"""
        self.found_persons = []
        self.current_target = None
        self.waiting_time = 0
        self.explored_positions = set()
        self.search_pattern = []
        self.pattern_index = 0
    
    def get_status(self) -> dict:
        """Get current drone status for monitoring"""
        return {
            'id': self.drone_id,
            'position': self.position,
            'state': self.state.value,
            'battery': self.current_battery,
            'found_persons': len(self.found_persons),
            'exploring': len(self.explored_positions),
            'waiting_time': self.waiting_time,
            'target': self.current_target
        }