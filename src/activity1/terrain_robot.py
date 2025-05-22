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
        
        # Battery management
        self.max_battery = max_battery
        self.current_battery = max_battery
        self.battery_drain_rate = battery_drain_rate
        self.low_battery_threshold = 20
        
        # Mission tracking
        self.target_person = None
        self.has_first_aid_kit = True
        self.persons_rescued = 0
        
        # Movement tracking
        self.last_positions = []  # Track recent positions to avoid getting stuck
        self.max_position_history = 5
    
    def update(self, environment) -> None:
        """Main update method called each simulation step"""
        # Drain battery
        self._drain_battery(environment)
        
        # Check battery level
        if self._is_battery_low() and self.state != RobotState.BATTERY_LOW:
            self.state = RobotState.BATTERY_LOW
            self.target_person = None
        
        # Execute behaviour based on current state
        if self.state == RobotState.AT_BASE:
            self._handle_at_base_state(environment)
        elif self.state == RobotState.SEARCHING:
            self._handle_searching_state(environment)
        elif self.state == RobotState.DELIVERING:
            self._handle_delivering_state(environment)
        elif self.state == RobotState.RETURNING:
            self._handle_returning_state(environment)
        elif self.state == RobotState.BATTERY_LOW:
            self._handle_battery_low_state(environment)
    
    def _drain_battery(self, environment) -> None:
        """Drain battery based on current terrain and actions"""
        drain = self.battery_drain_rate
        
        # Increase drain when climbing (higher elevation = more drain)
        elevation = environment.get_elevation(*self.position)
        if elevation > 0:
            drain += elevation // 1000  # Extra drain per 1000m elevation
        
        self.current_battery = max(0, self.current_battery - drain)
    
    def _is_battery_low(self) -> bool:
        """Check if battery is below low threshold"""
        return self.current_battery <= self.low_battery_threshold
    
    def _handle_at_base_state(self, environment) -> None:
        """Handle behaviour when robot is at base station"""
        if self.current_battery < self.max_battery:
            # Recharge battery
            self.current_battery = min(self.max_battery, self.current_battery + 5)
        
        if self.current_battery >= 40 and self.has_first_aid_kit:
            # Ready to start mission (lowered threshold so robots start sooner)
            self.state = RobotState.SEARCHING
    
    def _handle_searching_state(self, environment) -> None:
        """Handle searching behaviour - random movement to find missing persons"""
        # Check if person found at current location
        if environment.person_at_location(*self.position):
            self.target_person = self.position
            self.state = RobotState.DELIVERING
            return
        
        # Move randomly to search
        self._move_randomly(environment)
    
    def _handle_delivering_state(self, environment) -> None:
        """Handle first-aid delivery to found person"""
        if self.target_person and self.position == self.target_person:
            # Deliver first aid
            if environment.rescue_person(*self.position):
                self.has_first_aid_kit = False
                self.persons_rescued += 1
                self.target_person = None
                self.state = RobotState.RETURNING
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
            # At base, can recharge
            self.state = RobotState.AT_BASE
        else:
            # Move towards base urgently
            self._move_towards_target(self.base_position, environment)
    
    def _move_randomly(self, environment) -> None:
        """Move in a random direction, avoiding obstacles and boundaries"""
        possible_moves = self._get_valid_moves(environment)
        
        if possible_moves:
            # Prefer moves to new areas (avoid recent positions)
            new_area_moves = [move for move in possible_moves 
                             if move not in self.last_positions]
            
            if new_area_moves:
                new_position = random.choice(new_area_moves)
            else:
                new_position = random.choice(possible_moves)
            
            self._update_position(new_position)
    
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