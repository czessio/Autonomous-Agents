# Terrain robot agent for mountain rescue simulation - handles first-aid kit delivery with battery management

import random
from enum import Enum
from typing import Tuple, Optional


class RobotState(Enum):
    SEARCHING = "searching"
    DELIVERING = "delivering"
    RETURNING = "returning"
    AT_BASE = "at_base"
    BATTERY_LOW = "battery_low"


class TerrainRobot:
    def __init__(self, robot_id: int, start_position: Tuple[int, int], 
                 max_battery: int = 100, battery_drain_rate: int = 1):
        self.robot_id = robot_id
        self.position = start_position
        self.base_position = start_position
        self.state = RobotState.AT_BASE
        
        # battery management 
        self.max_battery = max_battery
        self.current_battery = max_battery
        self.battery_drain_rate = battery_drain_rate
        self.low_battery_threshold = 20  
        self.min_leave_threshold = 90   
        self.recharge_rate = 10        
        
        # mission tracking 
        self.target_person = None
        self.has_first_aid_kit = True
        self.persons_rescued = 0
        self.assigned_location = None  # For communication system
        
        # movement tracking
        self.last_positions = []  # Track recent positions to avoid getting stuck
        self.max_position_history = 8  # Increased to avoid loops better
        self.stuck_counter = 0  # Track if robot is stuck
        self.current_step = 0
    
    def update(self, environment, current_step=0, communication=None) -> None:
        """Main update method called each simulation step"""
        self.current_step = current_step
        
        # Battery management: drain only when not at base
        if self.position != self.base_position:
            self._drain_battery(environment)
        elif self.state == RobotState.AT_BASE:
            # Recharge when at base
            old_battery = self.current_battery
            self.current_battery = min(self.max_battery, self.current_battery + self.recharge_rate)
        
        # Check battery level
        if self._is_battery_low() and self.state not in [RobotState.BATTERY_LOW, RobotState.RETURNING, RobotState.AT_BASE]:
            self.state = RobotState.BATTERY_LOW
            self.target_person = None
            self.assigned_location = None
            print(f" Robot {self.robot_id} battery low ({self.current_battery}%), returning to base")
        
        # Execute behaviour based on current state
        if self.state == RobotState.AT_BASE:
            self._handle_at_base_state(environment, communication)
        elif self.state == RobotState.SEARCHING:
            self._handle_searching_state(environment, communication)
        elif self.state == RobotState.DELIVERING:
            self._handle_delivering_state(environment, communication)
        elif self.state == RobotState.RETURNING:
            self._handle_returning_state(environment)
        elif self.state == RobotState.BATTERY_LOW:
            self._handle_battery_low_state(environment)
    
    def _drain_battery(self, environment) -> None:
        """Drain battery based on current terrain and actions"""
        drain = self.battery_drain_rate
        
        # (higher elevation = more drain)
        elevation = environment.get_elevation(*self.position)
        if elevation > 0:
            drain += elevation // 1000  
        
        self.current_battery = max(0, self.current_battery - drain)
    
    def _is_battery_low(self) -> bool:
        """Check if battery is below low threshold"""
        return self.current_battery <= self.low_battery_threshold
    
    def _handle_at_base_state(self, environment, communication=None) -> None:
        """Handle behaviour when robot is at base station"""        
        # Only leave base if battery is above minimum threshold and has first aid kit
        if self.current_battery > self.min_leave_threshold and self.has_first_aid_kit:
            # Check for drone-found persons first (Basic Mode communication)
            if communication:
                nearest_location = communication.get_nearest_person_location(self.position)
                if nearest_location:
                    self.assigned_location = nearest_location
                    communication.assign_robot_to_location(self.robot_id, nearest_location)
                    self.state = RobotState.SEARCHING  # Will navigate to assigned location
                    self.stuck_counter = 0
                    print(f" Robot {self.robot_id} leaving base ({self.current_battery}%) for reported person at {nearest_location}")
                    return
            
            # No drone reports, do random search
            if self.current_battery > 50:  # Only do random search with good battery
                self.state = RobotState.SEARCHING
                self.stuck_counter = 0
                print(f" Robot {self.robot_id} leaving base ({self.current_battery}%) for random search")
    
    def _handle_searching_state(self, environment, communication=None) -> None:
        """Handle searching behaviour - respond to communications or move randomly"""
        # Check if person found at current location
        if environment.person_at_location(*self.position):
            # Found a person, deliver aid immediately
            self.target_person = self.position
            self.state = RobotState.DELIVERING
            print(f" Robot {self.robot_id} found person at {self.position}!")
            return
        
        # If robot has assigned location, navigate there
        if self.assigned_location:
            if communication and communication.is_robot_near_location(self.position, self.assigned_location):
                # Arrived at assigned location
                if environment.person_at_location(*self.assigned_location):
                    self.target_person = self.assigned_location
                    self.state = RobotState.DELIVERING
                    print(f" Robot {self.robot_id} arrived at assigned location {self.assigned_location}")
                    return
                else:
                    # Person already rescued or moved
                    print(f" Robot {self.robot_id} arrived but person at {self.assigned_location} already rescued")
                    self.assigned_location = None
            else:
                # Move towards assigned location
                self._move_towards_target(self.assigned_location, environment)
                return
        
        # No assigned location, move randomly towards mountain (Basic Mode)
        self._move_randomly(environment)
    
    def _handle_delivering_state(self, environment, communication=None) -> None:
        """Handle first-aid delivery to found person"""
        if self.target_person and self.position == self.target_person:
            # Deliver first aid
            if environment.rescue_person(*self.position):
                self.has_first_aid_kit = False
                self.persons_rescued += 1
                
                # Notify communication system
                if communication and self.assigned_location:
                    communication.robot_completed_rescue(self.robot_id, self.assigned_location)
                
                self.target_person = None
                self.assigned_location = None
                self.state = RobotState.RETURNING
                print(f" Robot {self.robot_id} rescued person, returning to base")
        else:
            # Move towards target person
            self._move_towards_target(self.target_person, environment)
    
    def _handle_returning_state(self, environment) -> None:
        """Handle return to base station"""
        if self.position == self.base_position:
            # Arrived at base
            self.has_first_aid_kit = True  # Restock first aid kit
            self.state = RobotState.AT_BASE
        else:
            # Move towards base
            self._move_towards_target(self.base_position, environment)
    
    def _handle_battery_low_state(self, environment) -> None:
        """Handle emergency return when battery is low"""
        if self.position == self.base_position:
            # At base, can recharge - transition to at_base state
            self.state = RobotState.AT_BASE
            print(f" Robot {self.robot_id} returned to base for recharging")
        else:
            # Move towards base urgently
            self._move_towards_target(self.base_position, environment)
    
    def _move_randomly(self, environment) -> None:
        """Move in a random direction, prioritizing mountain area (Basic Mode)"""
        possible_moves = self._get_valid_moves(environment)
        
        if possible_moves:
            # Prioritize moves toward mountain area and avoid recent positions
            mountain_moves = []
            new_area_moves = []
            toward_mountain_moves = []
            
            for move in possible_moves:
                if environment.is_mountain_area(*move):
                    mountain_moves.append(move)
                if move not in self.last_positions:
                    new_area_moves.append(move)
                # Add moves that get closer to mountain center
                if self._is_moving_toward_mountain(move, environment):
                    toward_mountain_moves.append(move)
            
            # Prioritize: mountain area > toward mountain > new areas > any valid move
            if mountain_moves and new_area_moves:
                preferred_moves = [m for m in mountain_moves if m in new_area_moves]
                if preferred_moves:
                    new_position = random.choice(preferred_moves)
                else:
                    new_position = random.choice(mountain_moves)
            elif mountain_moves:
                new_position = random.choice(mountain_moves)
            elif toward_mountain_moves:
                new_position = random.choice(toward_mountain_moves)
            elif new_area_moves:
                new_position = random.choice(new_area_moves)
            else:
                new_position = random.choice(possible_moves)
            
            # Check if stuck (same position for too long)
            if new_position == self.position:
                self.stuck_counter += 1
                if self.stuck_counter > 3:
                    # Force movement to break out of stuck state
                    new_position = random.choice(possible_moves)
                    self.stuck_counter = 0
            else:
                self.stuck_counter = 0
            
            self._update_position(new_position)
        else:
            # No valid moves - might be stuck
            self.stuck_counter += 1
    
    def _is_moving_toward_mountain(self, position, environment) -> bool:
        """Check if a position moves closer to mountain area"""
        if hasattr(environment, 'mountain_area'):
            x_start, y_start, x_end, y_end = environment.mountain_area['bounds']
            mountain_center_x = (x_start + x_end) // 2
            mountain_center_y = (y_start + y_end) // 2
            
            current_dist = abs(self.position[0] - mountain_center_x) + abs(self.position[1] - mountain_center_y)
            new_dist = abs(position[0] - mountain_center_x) + abs(position[1] - mountain_center_y)
            
            return new_dist < current_dist
        return False
    
    def _move_towards_target(self, target: Tuple[int, int], environment) -> None:
        """Move one step towards a target position"""
        if not target:
            return
        
        current_x, current_y = self.position
        target_x, target_y = target
        
        # Calculate direction
        dx = 0 if target_x == current_x else (1 if target_x > current_x else -1)
        dy = 0 if target_y == current_y else (1 if target_y > current_y else -1)
        
        # Try to move diagonally first, then orthogonally
        possible_moves = []
        
        if dx != 0 and dy != 0:
            possible_moves.append((current_x + dx, current_y + dy))
        if dx != 0:
            possible_moves.append((current_x + dx, current_y))
        if dy != 0:
            possible_moves.append((current_x, current_y + dy))
        
        # Filter valid moves
        valid_moves = [move for move in possible_moves 
                      if self._is_valid_position(move, environment)]
        
        if valid_moves:
            # Choose the move that gets closest to target
            best_move = min(valid_moves, 
                           key=lambda pos: abs(pos[0] - target_x) + abs(pos[1] - target_y))
            self._update_position(best_move)
    
    def _get_valid_moves(self, environment) -> list:
        """Get all valid adjacent positions"""
        x, y = self.position
        potential_moves = [
            (x + 1, y), (x - 1, y),
            (x, y + 1), (x, y - 1),
            (x + 1, y + 1), (x - 1, y - 1),
            (x + 1, y - 1), (x - 1, y + 1)
        ]
        
        return [move for move in potential_moves 
                if self._is_valid_position(move, environment)]
    
    def _is_valid_position(self, position: Tuple[int, int], environment) -> bool:
        """Check if a position is valid (within bounds)"""
        x, y = position
        return 0 <= x < environment.width and 0 <= y < environment.height
    
    def _update_position(self, new_position: Tuple[int, int]) -> None:
        """Update robot position and maintain position history"""
        self.last_positions.append(self.position)
        if len(self.last_positions) > self.max_position_history:
            self.last_positions.pop(0)
        
        self.position = new_position
    
    def get_status(self) -> dict:
        """Get current robot status for monitoring"""
        return {
            'id': self.robot_id,
            'position': self.position,
            'state': self.state.value,
            'battery': self.current_battery,
            'has_kit': self.has_first_aid_kit,
            'rescued': self.persons_rescued,
            'target': self.target_person
        }