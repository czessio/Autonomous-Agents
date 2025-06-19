# Q-Learning implementation for intelligent terrain robot navigation

import numpy as np
import pickle
import os
from typing import Tuple, Dict, List, Optional
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning implementation for terrain robots with collective knowledge sharing.
    
    This implementation allows robots to:
    1. Learn optimal paths through the mountain terrain
    2. Remember successful rescue locations
    3. Share knowledge with other robots
    4. Adapt to terrain difficulty
    """
    
    def __init__(self, state_space_size: Tuple[int, int], action_space_size: int = 8,
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 1.0, exploration_decay: float = 0.995,
                 min_exploration_rate: float = 0.01):
        
        # Q-Learning parameters
        self.learning_rate = learning_rate  # Alpha: how much new info overrides old
        self.discount_factor = discount_factor  # Gamma: importance of future rewards
        self.exploration_rate = exploration_rate  # Epsilon: exploration vs exploitation
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # State and action spaces
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        
        # Initialize Q-table with small random values to break ties
        self.q_table = defaultdict(lambda: np.random.uniform(-0.01, 0.01, action_space_size))
        
        # Collective knowledge components
        self.terrain_difficulty = {}  # Maps positions to traversal difficulty
        self.rescue_success_map = defaultdict(int)  # Maps positions to rescue count
        self.exploration_bonus = defaultdict(int)  # Bonus for unexplored areas
        
        # Experience tracking
        self.episode_count = 0
        self.total_rewards = []
        self.learning_history = []
        
    def get_state_key(self, position: Tuple[int, int], target: Optional[Tuple[int, int]] = None,
                     battery_level: int = 100, has_kit: bool = True) -> str:
        """
        Convert current situation into a state key for Q-table lookup.
        
        State includes:
        - Current position
        - Relative direction to target (if any)
        - Battery level (discretized)
        - First-aid kit status
        """
        # Discretize battery into levels
        battery_state = "high" if battery_level > 70 else "medium" if battery_level > 30 else "low"
        
        # Calculate relative direction to target
        if target:
            dx = np.sign(target[0] - position[0])
            dy = np.sign(target[1] - position[1])
            direction = f"{dx},{dy}"
        else:
            direction = "exploring"
        
        # Create state key
        kit_status = "kit" if has_kit else "no_kit"
        return f"{position[0]},{position[1]}|{direction}|{battery_state}|{kit_status}"
    
    def choose_action(self, state_key: str, valid_actions: List[int], 
                     position: Tuple[int, int]) -> int:
        """
        Choose action using epsilon-greedy strategy with exploration bonuses.
        """
        # Exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            # Explore: choose random valid action
            return np.random.choice(valid_actions)
        else:
            # Exploit: choose best action from Q-table
            q_values = self.q_table[state_key].copy()
            
            # Add exploration bonus for less-visited positions
            for action in valid_actions:
                next_pos = self._get_next_position(position, action)
                if next_pos:
                    exploration_bonus = self.exploration_bonus[next_pos]
                    q_values[action] += exploration_bonus * 0.1
            
            # Filter to valid actions only
            valid_q_values = [(action, q_values[action]) for action in valid_actions]
            
            # Choose action with highest Q-value
            best_action = max(valid_q_values, key=lambda x: x[1])[0]
            return best_action
    
    def update_q_value(self, state_key: str, action: int, reward: float, 
                      next_state_key: str, valid_next_actions: List[int]) -> None:
        """
        Update Q-value using the Q-learning update rule:
        Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state_key][action]
        
        # Get maximum Q-value for next state
        if valid_next_actions:
            next_q_values = [self.q_table[next_state_key][a] for a in valid_next_actions]
            max_next_q = max(next_q_values)
        else:
            max_next_q = 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Track learning progress
        self.learning_history.append({
            'episode': self.episode_count,
            'state': state_key,
            'action': action,
            'reward': reward,
            'q_value': new_q
        })
    
    def calculate_reward(self, position: Tuple[int, int], next_position: Tuple[int, int],
                        found_person: bool, rescued_person: bool, battery_used: int,
                        terrain_elevation: int, is_mountain_area: bool) -> float:
        """
        Calculate sophisticated reward based on multiple factors.
        """
        reward = 0
        
        # Large positive rewards for mission success
        if rescued_person:
            reward += 100  # Major success
            self.rescue_success_map[position] += 1
        elif found_person:
            reward += 50   # Found but not rescued yet
        
        # Movement and exploration rewards
        if next_position != position:
            reward += 1  # Small reward for movement
            
            # Bonus for exploring new areas
            if self.exploration_bonus[next_position] == 0:
                reward += 5
            self.exploration_bonus[next_position] += 1
        
        # Terrain-based rewards/penalties
        if is_mountain_area:
            reward += 2  # Encourage mountain exploration
        
        # Elevation penalty (harder to climb)
        elevation_penalty = terrain_elevation / 1000 * 2
        reward -= elevation_penalty
        
        # Battery efficiency penalty
        reward -= battery_used * 0.5
        
        # Update terrain difficulty knowledge
        if position in self.terrain_difficulty:
            # Running average of terrain difficulty
            old_difficulty = self.terrain_difficulty[position]
            self.terrain_difficulty[position] = 0.9 * old_difficulty + 0.1 * elevation_penalty
        else:
            self.terrain_difficulty[position] = elevation_penalty
        
        return reward
    
    def decay_exploration_rate(self) -> None:
        """
        Decay exploration rate after each episode to shift from exploration to exploitation.
        """
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)
        self.episode_count += 1
    
    def get_learned_path_preference(self, current_pos: Tuple[int, int], 
                                   next_positions: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """
        Get preference scores for next positions based on learned knowledge.
        """
        preferences = {}
        
        for next_pos in next_positions:
            score = 0
            
            # Preference based on rescue success history
            if next_pos in self.rescue_success_map:
                score += self.rescue_success_map[next_pos] * 10
            
            # Penalty based on terrain difficulty
            if next_pos in self.terrain_difficulty:
                score -= self.terrain_difficulty[next_pos] * 5
            
            # Small bonus for less explored areas
            exploration_count = self.exploration_bonus.get(next_pos, 0)
            if exploration_count < 5:
                score += (5 - exploration_count)
            
            preferences[next_pos] = score
        
        return preferences
    
    def _get_next_position(self, position: Tuple[int, int], action: int) -> Optional[Tuple[int, int]]:
        """
        Get next position based on action.
        Actions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
        """
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
        
        if 0 <= action < len(action_deltas):
            dx, dy = action_deltas[action]
            return (position[0] + dx, position[1] + dy)
        return None
    
    def save_knowledge(self, filepath: str) -> None:
        """
        Save learned knowledge to file for persistence across simulations.
        """
        knowledge = {
            'q_table': dict(self.q_table),
            'terrain_difficulty': self.terrain_difficulty,
            'rescue_success_map': dict(self.rescue_success_map),
            'exploration_bonus': dict(self.exploration_bonus),
            'episode_count': self.episode_count,
            'exploration_rate': self.exploration_rate
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(knowledge, f)
    
    def load_knowledge(self, filepath: str) -> bool:
        """
        Load previously learned knowledge from file.
        """
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    knowledge = pickle.load(f)
                
                self.q_table = defaultdict(lambda: np.zeros(self.action_space_size), knowledge['q_table'])
                self.terrain_difficulty = knowledge['terrain_difficulty']
                self.rescue_success_map = defaultdict(int, knowledge['rescue_success_map'])
                self.exploration_bonus = defaultdict(int, knowledge['exploration_bonus'])
                self.episode_count = knowledge['episode_count']
                self.exploration_rate = knowledge['exploration_rate']
                
                return True
            except Exception as e:
                print(f"Failed to load knowledge: {e}")
                return False
        return False


class CollectiveQLearning:
    """
    Manages shared Q-learning knowledge across multiple robot agents.
    """
    
    def __init__(self, num_robots: int, environment_size: Tuple[int, int]):
        self.num_robots = num_robots
        self.environment_size = environment_size
        
        # Shared knowledge base
        self.shared_q_agent = QLearningAgent(environment_size)
        
        # Individual robot learning agents
        self.robot_agents = {}
        for i in range(num_robots):
            self.robot_agents[i] = QLearningAgent(environment_size)
        
        # Knowledge sharing parameters
        self.sharing_interval = 10  # Share knowledge every N episodes
        self.merge_weight = 0.3     # Weight for merging individual knowledge
        
    def get_robot_agent(self, robot_id: int) -> QLearningAgent:
        """Get the Q-learning agent for a specific robot."""
        return self.robot_agents.get(robot_id, self.shared_q_agent)
    
    def share_knowledge(self) -> None:
        """
        Merge individual robot knowledge into shared knowledge base.
        Uses weighted averaging to combine Q-values and terrain knowledge.
        """
        print(" Sharing collective knowledge between robots...")
        
        # Merge Q-tables
        all_states = set()
        for agent in self.robot_agents.values():
            all_states.update(agent.q_table.keys())
        
        for state in all_states:
            # Weighted average of Q-values across all robots
            combined_q_values = np.zeros(self.shared_q_agent.action_space_size)
            count = 0
            
            for agent in self.robot_agents.values():
                if state in agent.q_table:
                    combined_q_values += agent.q_table[state]
                    count += 1
            
            if count > 0:
                # Update shared Q-table
                self.shared_q_agent.q_table[state] = (
                    (1 - self.merge_weight) * self.shared_q_agent.q_table[state] +
                    self.merge_weight * (combined_q_values / count)
                )
        
        # Merge terrain difficulty knowledge
        all_positions = set()
        for agent in self.robot_agents.values():
            all_positions.update(agent.terrain_difficulty.keys())
        
        for pos in all_positions:
            difficulties = [agent.terrain_difficulty[pos] 
                          for agent in self.robot_agents.values() 
                          if pos in agent.terrain_difficulty]
            if difficulties:
                self.shared_q_agent.terrain_difficulty[pos] = np.mean(difficulties)
        
        # Merge rescue success locations
        for agent in self.robot_agents.values():
            for pos, count in agent.rescue_success_map.items():
                self.shared_q_agent.rescue_success_map[pos] += count
        
        # Update individual robots with shared knowledge
        for robot_id, agent in self.robot_agents.items():
            # Blend individual and shared Q-tables
            for state, q_values in self.shared_q_agent.q_table.items():
                if state in agent.q_table:
                    agent.q_table[state] = (
                        0.7 * agent.q_table[state] + 0.3 * q_values
                    )
                else:
                    agent.q_table[state] = q_values.copy()
            
            # Share terrain and rescue knowledge
            agent.terrain_difficulty.update(self.shared_q_agent.terrain_difficulty)
            for pos, count in self.shared_q_agent.rescue_success_map.items():
                agent.rescue_success_map[pos] = max(agent.rescue_success_map[pos], count)
        
        print(f" Knowledge shared: {len(all_states)} states, {len(all_positions)} terrain points")
    
    def save_collective_knowledge(self, filepath: str) -> None:
        """Save the collective knowledge base."""
        self.shared_q_agent.save_knowledge(filepath)
    
    def load_collective_knowledge(self, filepath: str) -> bool:
        """Load previously saved collective knowledge."""
        return self.shared_q_agent.load_knowledge(filepath)
    
    def get_learning_statistics(self) -> Dict:
        """Get statistics about the collective learning progress."""
        total_states_explored = len(self.shared_q_agent.q_table)
        total_terrain_mapped = len(self.shared_q_agent.terrain_difficulty)
        total_rescue_locations = len(self.shared_q_agent.rescue_success_map)
        avg_exploration_rate = np.mean([agent.exploration_rate for agent in self.robot_agents.values()])
        
        return {
            'total_states_explored': total_states_explored,
            'total_terrain_mapped': total_terrain_mapped,
            'total_rescue_locations': total_rescue_locations,
            'average_exploration_rate': avg_exploration_rate,
            'episodes_completed': self.shared_q_agent.episode_count
        }