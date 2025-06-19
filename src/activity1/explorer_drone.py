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
                 max_battery: int = 125, battery_drain_rate: int = 1):
        self.drone_id = drone_id
        self.position = start_position
        self.base_position = start_position
        self.state = DroneState.AT_BASE
        
        # Simple, reliable battery management
        self.max_battery = max_battery
        self.current_battery = max_battery
        self.battery_drain_rate = battery_drain_rate
        self.low_battery_threshold = 20  # Return to base at 35%
        self.min_leave_threshold = 90   # Don't leave base until 85% battery
        self.recharge_rate = 1         # Much faster recharge: 15% per step at base
        
        # Mission tracking
        self.found_persons = []  # List of persons this drone has found
        self.current_target = None
        self.waiting_time = 0
        self.max_waiting_time = 15  # Wait longer for robots
        
        # Exploration tracking - fixed
        self.explored_positions = set()
        self.search_pattern = []
        self.pattern_index = 0
        self.total_areas_explored = 0  # Track total exploration
        self.current_step = 0
        
        # 6-directional movement capability
        self.movement_directions = [
            (1, 0),   # front (right)
            (-1, 0),  # back (left)
            (0, 1),   # up
            (0, -1),  # down
            (1, 1),   # front-up (diagonal)
            (-1, -1)  # back-down (diagonal)
        ]
    
    def update(self, environment, current_step=0, communication=None) -> None:
        """Main update method called each simulation step"""
        self.current_step = current_step
        
        # Battery management: drain only when not at base
        if self.position != self.base_position:
            self._drain_battery()
        elif self.state == DroneState.AT_BASE:
            # Recharge when at base
            old_battery = self.current_battery
            self.current_battery = min(self.max_battery, self.current_battery + self.recharge_rate)
        
        # Check battery level
        if self._is_battery_low() and self.state not in [DroneState.BATTERY_LOW, DroneState.RETURNING, DroneState.AT_BASE]:
            self.state = DroneState.BATTERY_LOW
            self.current_target = None
            self.waiting_time = 0
            print(f"⚡ Drone {self.drone_id} battery low ({self.current_battery}%), returning to base")
        
        # Execute behaviour based on current state
        if self.state == DroneState.AT_BASE:
            self._handle_at_base_state(environment, communication)
        elif self.state == DroneState.EXPLORING:
            self._handle_exploring_state(environment, communication)
        elif self.state == DroneState.FOUND_PERSON:
            self._handle_found_person_state(environment, communication)
        elif self.state == DroneState.WAITING:
            self._handle_waiting_state(environment, communication)
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
    
    def _handle_at_base_state(self, environment, communication=None) -> None:
        """Handle behaviour when drone is at base station"""        
        # Only leave base if battery is above minimum threshold
        if self.current_battery > self.min_leave_threshold:
            self._generate_search_pattern(environment)
            self.state = DroneState.EXPLORING
            print(f" Drone {self.drone_id} leaving base with {self.current_battery}% battery")
    
    def _handle_exploring_state(self, environment, communication=None) -> None:
        """Handle exploration behaviour using 6-directional movement"""
        # Check if person found at current location
        if environment.person_at_location(*self.position):
            # Only count as found if we haven't found this person before
            if self.position not in self.found_persons:
                self.found_persons.append(self.position)
                self.current_target = self.position
                
                # Report to communication system
                if communication:
                    communication.drone_found_person(self.drone_id, self.position, urgency=1)
                
                self.state = DroneState.FOUND_PERSON
                return
        
        # Mark current position as explored and increment counter
        if self.position not in self.explored_positions:
            self.explored_positions.add(self.position)
            self.total_areas_explored += 1
        
        # Move according to search pattern or randomly
        self._execute_movement(environment)
    
    def _handle_found_person_state(self, environment, communication=None) -> None:
        """Handle immediate response when person is found"""
        # Take pictures and assess urgency (simulated)
        person_urgency = self._assess_person_urgency()
        
        # Move to waiting state
        self.waiting_time = 0
        self.state = DroneState.WAITING
    
    def _handle_waiting_state(self, environment, communication=None) -> None:
        """Handle waiting behaviour after finding a person - Basic Mode"""
        self.waiting_time += 1
        
        # Report waiting location to communication system
        if communication:
            communication.drone_waiting_at(self.position, self.drone_id)
        
        # Check if person has been rescued (robot arrived and completed rescue)
        if not environment.person_at_location(*self.position):
            print(f" Person at {self.position} rescued! Drone {self.drone_id} returning to base.")
            if communication:
                communication.drone_stopped_waiting(self.position)
            self.current_target = None
            self.state = DroneState.RETURNING
            return
        
        # If waited too long, continue exploring
        if self.waiting_time >= self.max_waiting_time:
            print(f" Drone {self.drone_id} waited too long, continuing exploration...")
            if communication:
                communication.drone_stopped_waiting(self.position)
            self.current_target = None
            # Continue exploring or return based on battery
            if self._is_battery_low():
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
            # At base, can recharge - transition to at_base state
            self.state = DroneState.AT_BASE
            print(f" Drone {self.drone_id} returned to base for recharging")
        else:
            # Move towards base urgently
            self._move_towards_target(self.base_position, environment)
    
    def _generate_search_pattern(self, environment) -> None:
        """Generate efficient search pattern - TOP and BOTTOM starting positions"""
        self.search_pattern = []
        
        # Focus on mountain area with systematic coverage
        if hasattr(environment, 'mountain_area'):
            x_start, y_start, x_end, y_end = environment.mountain_area['bounds']
            
            if self.drone_id == 0:
                # Drone 0: Start from TOP of mountain, work downwards (top-to-bottom)
                print(f" Drone {self.drone_id} will search from TOP of mountain (top-to-bottom)")
                print(f"   Starting at top: ({x_start}, {y_start})")
                
                for row in range(y_start, y_end):  # Start from TOP row, go down
                    row_index = row - y_start  # Calculate which row we're on (0, 1, 2...)
                    if row_index % 2 == 0:  # Even rows: left to right
                        for x in range(x_start, x_end):
                            self.search_pattern.append((x, row))
                    else:  # Odd rows: right to left
                        for x in range(x_end - 1, x_start - 1, -1):
                            self.search_pattern.append((x, row))
                        
            elif self.drone_id == 1:
                # Drone 1: Start from BOTTOM of mountain, work upwards (bottom-to-top)
                print(f" Drone {self.drone_id} will search from BOTTOM of mountain (bottom-to-top)")
                print(f"   Starting at bottom: ({x_start}, {y_end-1})")
                
                for row in range(y_end - 1, y_start - 1, -1):  # Start from BOTTOM row, go up
                    row_index = (y_end - 1) - row  # Calculate which row we're on (0, 1, 2...)
                    if row_index % 2 == 0:  # Even rows: left to right
                        for x in range(x_start, x_end):
                            self.search_pattern.append((x, row))
                    else:  # Odd rows: right to left
                        for x in range(x_end - 1, x_start - 1, -1):
                            self.search_pattern.append((x, row))
                        
            else:
                # Additional drones (if any): Use alternating top/bottom starts
                if self.drone_id % 2 == 0:
                    # Even drone IDs: start from top
                    print(f" Drone {self.drone_id} will search from TOP")
                    for row in range(y_start, y_end):
                        row_index = row - y_start
                        if row_index % 2 == 0:
                            for x in range(x_start, x_end):
                                self.search_pattern.append((x, row))
                        else:
                            for x in range(x_end - 1, x_start - 1, -1):
                                self.search_pattern.append((x, row))
                else:
                    # Odd drone IDs: start from bottom
                    print(f" Drone {self.drone_id} will search from BOTTOM")
                    for row in range(y_end - 1, y_start - 1, -1):
                        row_index = (y_end - 1) - row
                        if row_index % 2 == 0:
                            for x in range(x_start, x_end):
                                self.search_pattern.append((x, row))
                        else:
                            for x in range(x_end - 1, x_start - 1, -1):
                                self.search_pattern.append((x, row))
            
            # Add a couple boundary exploration points
            if self.drone_id == 0:
                # Top drone gets some points around top edge
                for _ in range(2):
                    x = random.randint(x_start, x_end - 1)
                    self.search_pattern.append((x, max(0, y_start - 1)))
            elif self.drone_id == 1:
                # Bottom drone gets some points around bottom edge
                for _ in range(2):
                    x = random.randint(x_start, x_end - 1)
                    self.search_pattern.append((x, min(environment.height - 1, y_end)))
        
        self.pattern_index = 0
        print(f" Drone {self.drone_id} generated search pattern with {len(self.search_pattern)} waypoints")
    
    def _execute_movement(self, environment) -> None:
        """Execute movement using 6-directional capability with memory"""
        # Try to follow search pattern first
        if self.pattern_index < len(self.search_pattern):
            target = self.search_pattern[self.pattern_index]
            if self.position == target:
                self.pattern_index += 1
            else:
                self._move_towards_target(target, environment)
        else:
            # Pattern complete - explore unvisited areas only
            self._move_to_unexplored_area(environment)
    
    def _move_to_unexplored_area(self, environment) -> None:
        """Move to unexplored areas only, avoiding previously visited positions"""
        # Get all valid moves
        possible_moves = []
        unexplored_moves = []
        mountain_unexplored_moves = []
        
        for dx, dy in self.movement_directions:
            new_x, new_y = self.position[0] + dx, self.position[1] + dy
            new_position = (new_x, new_y)
            
            if self._is_valid_position(new_position, environment):
                possible_moves.append(new_position)
                
                # Check if this position hasn't been explored yet
                if new_position not in self.explored_positions:
                    unexplored_moves.append(new_position)
                    
                    # Prefer unexplored mountain areas
                    if environment.is_mountain_area(*new_position):
                        mountain_unexplored_moves.append(new_position)
        
        # Priority: unexplored mountain > unexplored any > any valid move
        if mountain_unexplored_moves:
            self.position = random.choice(mountain_unexplored_moves)
            print(f" Drone {self.drone_id} exploring new mountain area at {self.position}")
        elif unexplored_moves:
            self.position = random.choice(unexplored_moves)
            print(f" Drone {self.drone_id} exploring new area at {self.position}")
        elif possible_moves:
            # All nearby areas explored, move to a random valid position
            # But try to find areas further away that might be unexplored
            furthest_move = self._find_furthest_unexplored_area(environment)
            if furthest_move:
                # Move towards the furthest unexplored area
                self._move_towards_target(furthest_move, environment)
            else:
                # Truly all areas explored, return to base or random move
                print(f" Drone {self.drone_id} has explored all reachable areas")
                if self.current_battery > 50:
                    self.position = random.choice(possible_moves)
                else:
                    self.state = DroneState.RETURNING
        
        # Always mark new position as explored
        if self.position not in self.explored_positions:
            self.explored_positions.add(self.position)
            self.total_areas_explored += 1
    
    def _find_furthest_unexplored_area(self, environment) -> Optional[Tuple[int, int]]:
        """Find the furthest unexplored area in the mountain"""
        if hasattr(environment, 'mountain_area'):
            x_start, y_start, x_end, y_end = environment.mountain_area['bounds']
            
            # Look for unexplored mountain positions
            unexplored_mountain_areas = []
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    if (x, y) not in self.explored_positions:
                        unexplored_mountain_areas.append((x, y))
            
            if unexplored_mountain_areas:
                # Return the furthest one from current position
                def distance_from_current(pos):
                    return abs(pos[0] - self.position[0]) + abs(pos[1] - self.position[1])
                
                return max(unexplored_mountain_areas, key=distance_from_current)
        
        return None
    
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
    
    def _is_valid_position(self, position: Tuple[int, int], environment) -> bool:
        """Check if position is valid (within bounds)"""
        x, y = position
        return 0 <= x < environment.width and 0 <= y < environment.height
    
    def _assess_person_urgency(self) -> int:
        """Simulate computer vision assessment of person urgency"""
        # Simulate age and health assessment
        # Returns urgency score from 1-10
        return random.randint(3, 9)
    
    def _reset_mission_data(self) -> None:
        """Reset mission-specific data when returning to base"""
        self.found_persons = []
        self.current_target = None
        self.waiting_time = 0
        self.explored_positions = set()
        self.search_pattern = []
        self.pattern_index = 0
        self.total_areas_explored = 0  # Reset counter as well
    
    def get_status(self) -> dict:
        """Get current drone status for monitoring"""
        return {
            'id': self.drone_id,
            'position': self.position,
            'state': self.state.value,
            'battery': self.current_battery,
            'found_persons': len(self.found_persons),
            'exploring': self.total_areas_explored,  # Use the counter instead of set length
            'waiting_time': self.waiting_time,
            'target': self.current_target
        }