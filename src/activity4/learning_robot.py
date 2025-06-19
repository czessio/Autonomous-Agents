# Learning terrain robot with Q-learning capabilities for Novel Mode

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Tuple, Optional, List
import numpy as np

from activity1.terrain_robot import TerrainRobot, RobotState
from q_learning import QLearningAgent


class LearningTerrainRobot(TerrainRobot):
    """
    Enhanced terrain robot that uses Q-learning for intelligent navigation.
    
    Key improvements over base robot:
    1. Learns optimal paths through reinforcement learning
    2. Remembers successful rescue locations
    3. Shares knowledge with other robots
    4. Adapts to terrain difficulty over time
    5. Balances exploration vs exploitation
    """
    
    def __init__(self, robot_id: int, start_position: Tuple[int, int], 
                 q_learning_agent: QLearningAgent,
                 max_battery: int = 120, battery_drain_rate: int = 1):
        
        # Initialize parent class
        super().__init__(robot_id, start_position, max_battery, battery_drain_rate)
        
        # Q-learning components
        self.q_agent = q_learning_agent
        self.learning_enabled = True
        
        # Learning state tracking
        self.previous_state_key = None
        self.previous_action = None
        self.previous_position = start_position
        self.episode_start_position = start_position
        self.episode_rewards = []
        
        # Performance metrics
        self.total_distance_traveled = 0
        self.successful_rescues = 0
        self.failed_attempts = 0
        self.learning_episodes_completed = 0
        
        print(f"🤖🧠 Learning Robot {robot_id} initialized with Q-learning capabilities")
    
    def update(self, environment, current_step=0, communication=None) -> None:
        """
        Enhanced update method with Q-learning integration.
        """
        self.current_step = current_step
        
        # Store current state before update
        old_battery = self.current_battery
        old_position = self.position
        old_state = self.state
        
        # Perform base robot update
        super().update(environment, current_step, communication)
        
        # Learning and reward calculation after action
        if self.learning_enabled and self.state != RobotState.AT_BASE:
            self._update_learning(old_position, old_battery, old_state, environment)
    
    def _handle_searching_state(self, environment, communication=None) -> None:
        """
        Override searching behavior to use Q-learning for decision making.
        """
        # Check if person found at current location
        if environment.person_at_location(*self.position):
            self.target_person = self.position
            self.state = RobotState.DELIVERING
            print(f"🤖🧠 Learning Robot {self.robot_id} found person at {self.position}!")
            return
        
        # If robot has assigned location, navigate there intelligently
        if self.assigned_location:
            if communication and communication.is_robot_near_location(self.position, self.assigned_location):
                if environment.person_at_location(*self.assigned_location):
                    self.target_person = self.assigned_location
                    self.state = RobotState.DELIVERING
                    return
                else:
                    print(f"🤖 Robot {self.robot_id} arrived but person already rescued")
                    self.assigned_location = None
            else:
                # Use Q-learning to navigate to assigned location
                self._move_with_q_learning(self.assigned_location, environment)
                return
        
        # No assigned location - use Q-learning for exploration
        self._explore_with_q_learning(environment)
    
    def _move_with_q_learning(self, target: Tuple[int, int], environment) -> None:
        """
        Use Q-learning to choose optimal movement towards target.
        """
        # Get current state representation
        state_key = self.q_agent.get_state_key(
            position=self.position,
            target=target,
            battery_level=self.current_battery,
            has_kit=self.has_first_aid_kit
        )
        
        # Get valid actions (movements)
        valid_actions = self._get_valid_actions(environment)
        
        if not valid_actions:
            return
        
        # Choose action using Q-learning
        action = self.q_agent.choose_action(state_key, valid_actions, self.position)
        
        # Store for learning update
        self.previous_state_key = state_key
        self.previous_action = action
        self.previous_position = self.position
        
        # Execute action
        self._execute_action(action, environment)
    
    def _explore_with_q_learning(self, environment) -> None:
        """
        Use Q-learning for intelligent exploration without a specific target.
        """
        # Get current state for exploration
        state_key = self.q_agent.get_state_key(
            position=self.position,
            target=None,  # No specific target
            battery_level=self.current_battery,
            has_kit=self.has_first_aid_kit
        )
        
        # Get valid actions
        valid_actions = self._get_valid_actions(environment)
        
        if not valid_actions:
            self.stuck_counter += 1
            return
        
        # Get learned preferences for next positions
        possible_next_positions = [self._get_position_from_action(action) 
                                  for action in valid_actions]
        preferences = self.q_agent.get_learned_path_preference(
            self.position, possible_next_positions
        )
        
        # Combine Q-values with learned preferences
        if np.random.random() < self.q_agent.exploration_rate:
            # Exploration: weighted random based on preferences
            if preferences:
                weights = np.array([preferences.get(pos, 0) + 10 for pos in possible_next_positions])
                weights = np.maximum(weights, 0.1)  # Ensure positive weights
                weights = weights / weights.sum()
                action = np.random.choice(valid_actions, p=weights)
            else:
                action = np.random.choice(valid_actions)
        else:
            # Exploitation: use Q-learning
            action = self.q_agent.choose_action(state_key, valid_actions, self.position)
        
        # Store for learning
        self.previous_state_key = state_key
        self.previous_action = action
        self.previous_position = self.position
        
        # Execute action
        self._execute_action(action, environment)
        self.stuck_counter = 0
    
    def _get_valid_actions(self, environment) -> List[int]:
        """
        Get list of valid actions (movement directions) from current position.
        Actions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
        """
        valid_actions = []
        action_deltas = [
            (0, -1),   # North
            (1, -1),   # Northeast
            (1, 0),    # East
            (1, 1),    # Southeast
            (0, 1),    # South
            (-1, 1),   # Southwest
            (-1, 0),   # West
            (-1, -1)   # Northwest
        ]
        
        for action, (dx, dy) in enumerate(action_deltas):
            new_x = self.position[0] + dx
            new_y = self.position[1] + dy
            
            if self._is_valid_position((new_x, new_y), environment):
                valid_actions.append(action)
        
        return valid_actions
    
    def _get_position_from_action(self, action: int) -> Tuple[int, int]:
        """Convert action to resulting position."""
        action_deltas = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]
        
        if 0 <= action < len(action_deltas):
            dx, dy = action_deltas[action]
            return (self.position[0] + dx, self.position[1] + dy)
        return self.position
    
    def _execute_action(self, action: int, environment) -> None:
        """Execute the chosen action (movement)."""
        new_position = self._get_position_from_action(action)
        
        if self._is_valid_position(new_position, environment):
            self._update_position(new_position)
            self.total_distance_traveled += 1
    
    def _update_learning(self, old_position: Tuple[int, int], old_battery: int,
                        old_state: RobotState, environment) -> None:
        """
        Update Q-learning based on the result of the previous action.
        """
        if self.previous_state_key is None or self.previous_action is None:
            return
        
        # Calculate reward based on what happened
        battery_used = old_battery - self.current_battery
        found_person = (self.state == RobotState.DELIVERING and 
                       old_state == RobotState.SEARCHING)
        rescued_person = (self.state == RobotState.RETURNING and 
                         old_state == RobotState.DELIVERING)
        
        # Get terrain information
        terrain_elevation = environment.get_elevation(*self.position)
        is_mountain = environment.is_mountain_area(*self.position)
        
        # Calculate reward
        reward = self.q_agent.calculate_reward(
            position=old_position,
            next_position=self.position,
            found_person=found_person,
            rescued_person=rescued_person,
            battery_used=battery_used,
            terrain_elevation=terrain_elevation,
            is_mountain_area=is_mountain
        )
        
        self.episode_rewards.append(reward)
        
        # Update success metrics
        if rescued_person:
            self.successful_rescues += 1
        
        # Get new state
        new_target = self.target_person or self.assigned_location
        new_state_key = self.q_agent.get_state_key(
            position=self.position,
            target=new_target,
            battery_level=self.current_battery,
            has_kit=self.has_first_aid_kit
        )
        
        # Get valid actions for new state
        valid_next_actions = self._get_valid_actions(environment)
        
        # Update Q-value
        self.q_agent.update_q_value(
            state_key=self.previous_state_key,
            action=self.previous_action,
            reward=reward,
            next_state_key=new_state_key,
            valid_next_actions=valid_next_actions
        )
        
        # Check for episode end (returned to base)
        if self.state == RobotState.AT_BASE and old_state != RobotState.AT_BASE:
            self._complete_learning_episode()
    
    def _complete_learning_episode(self) -> None:
        """
        Complete a learning episode when returning to base.
        """
        if self.episode_rewards:
            total_episode_reward = sum(self.episode_rewards)
            avg_reward = total_episode_reward / len(self.episode_rewards)
            
            print(f"🧠 Robot {self.robot_id} completed learning episode {self.learning_episodes_completed}:")
            print(f"   Total reward: {total_episode_reward:.2f}, Avg reward: {avg_reward:.2f}")
            print(f"   Exploration rate: {self.q_agent.exploration_rate:.3f}")
            
            # Decay exploration rate
            self.q_agent.decay_exploration_rate()
            
            # Reset episode tracking
            self.episode_rewards = []
            self.learning_episodes_completed += 1
            self.episode_start_position = self.position
    
    def _handle_at_base_state(self, environment, communication=None) -> None:
        """
        Enhanced base state handling with learning episode management.
        """
        # Complete learning episode if just returned
        if self.previous_state_key and self.episode_rewards:
            self._complete_learning_episode()
        
        # Call parent implementation
        super()._handle_at_base_state(environment, communication)
    
    def get_learning_status(self) -> dict:
        """
        Get current learning status and metrics.
        """
        base_status = self.get_status()
        
        learning_status = {
            **base_status,
            'learning_enabled': self.learning_enabled,
            'exploration_rate': self.q_agent.exploration_rate,
            'episodes_completed': self.learning_episodes_completed,
            'successful_rescues': self.successful_rescues,
            'total_distance': self.total_distance_traveled,
            'q_table_size': len(self.q_agent.q_table),
            'known_terrain_points': len(self.q_agent.terrain_difficulty),
            'known_rescue_locations': len(self.q_agent.rescue_success_map)
        }
        
        return learning_status
    
    def enable_learning(self, enabled: bool = True) -> None:
        """Enable or disable learning."""
        self.learning_enabled = enabled
        print(f"🤖 Robot {self.robot_id} learning {'enabled' if enabled else 'disabled'}")
    
    def share_knowledge_with(self, other_robot: 'LearningTerrainRobot') -> None:
        """
        Share learned knowledge with another robot.
        """
        # Share terrain difficulty knowledge
        for pos, difficulty in self.q_agent.terrain_difficulty.items():
            if pos not in other_robot.q_agent.terrain_difficulty:
                other_robot.q_agent.terrain_difficulty[pos] = difficulty
        
        # Share successful rescue locations
        for pos, count in self.q_agent.rescue_success_map.items():
            other_robot.q_agent.rescue_success_map[pos] = max(
                other_robot.q_agent.rescue_success_map.get(pos, 0), count
            )
        
        print(f"🤝 Robot {self.robot_id} shared knowledge with Robot {other_robot.robot_id}")