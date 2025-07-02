# Enhanced Learning terrain robot with Q-learning and hierarchical coordination integration
# Integrates with mediators, coordinators, and collective intelligence systems

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Tuple, Optional, List, Dict
import numpy as np
import time

from activity1.terrain_robot import TerrainRobot, RobotState
from activity4.q_learning import QLearningAgent


class LearningTerrainRobot(TerrainRobot):
    """
    Advanced learning terrain robot with hierarchical coordination integration.
    
    Enhanced capabilities:
    1. Multi-level Q-learning with collective intelligence
    2. Integration with mediator and coordinator agents
    3. Dynamic role adaptation based on strategic assignments
    4. Advanced performance tracking and self-optimization
    5. Hierarchical decision-making (strategic → tactical → operational)
    6. Multi-objective optimization learning
    """
    
    def __init__(self, robot_id: int, start_position: Tuple[int, int], 
                 q_learning_agent: QLearningAgent,
                 max_battery: int = 120, battery_drain_rate: int = 1):
        
        # Initialize parent class
        super().__init__(robot_id, start_position, max_battery, battery_drain_rate)
        
        # Enhanced Q-learning components
        self.q_agent = q_learning_agent
        self.learning_enabled = True
        self.learning_mode = "hierarchical"  # "pure_q", "hierarchical", "cooperative"
        
        # Hierarchical integration
        self.strategic_role = "unassigned"  # Assigned by coordinator
        self.tactical_guidance = {}  # Guidance from mediators
        self.coordination_level = "autonomous"  # autonomous, guided, coordinated
        
        # Advanced learning state tracking
        self.previous_state_key = None
        self.previous_action = None
        self.previous_position = start_position
        self.episode_start_position = start_position
        self.episode_rewards = []
        
        # Multi-objective learning
        self.objective_weights = {
            'rescue_success': 0.4,
            'time_efficiency': 0.3,
            'battery_efficiency': 0.2,
            'coordination_bonus': 0.1
        }
        
        # Performance and adaptation metrics
        self.total_distance_traveled = 0
        self.successful_rescues = 0
        self.failed_attempts = 0
        self.learning_episodes_completed = 0
        self.coordination_interactions = 0
        self.strategic_compliance_score = 1.0
        
        # Advanced learning features
        self.meta_learning_enabled = True
        self.collaborative_learning_enabled = True
        self.adaptive_exploration = True
        
        # Communication and coordination tracking
        self.last_coordinator_update = 0
        self.last_mediator_interaction = 0
        self.coordination_effectiveness = {}
        
        # Performance optimization
        self.performance_history = []
        self.adaptation_triggers = {
            'low_success_rate': 0.6,
            'high_battery_usage': 0.8,
            'poor_coordination': 0.5
        }
        
        print(f"🤖 Enhanced Learning Robot {robot_id} initialized with hierarchical AI")
        print(f"   Features: Q-Learning + Coordination + Multi-Objective Optimization")
    
    def set_strategic_role(self, role: str, coordinator_id: int = None) -> None:
        """
        Set strategic role assigned by coordinator
        """
        old_role = self.strategic_role
        self.strategic_role = role
        self.last_coordinator_update = time.time()
        
        # Adapt learning parameters based on strategic role
        self._adapt_to_strategic_role(role)
        
        if old_role != role:
            print(f"🎯 Robot {self.robot_id}: Strategic role changed {old_role} → {role}")
    
    def receive_tactical_guidance(self, guidance: Dict, mediator_id: int = None) -> None:
        """
        Receive tactical guidance from mediator agents
        """
        self.tactical_guidance.update(guidance)
        self.last_mediator_interaction = time.time()
        self.coordination_interactions += 1
        
        # Integrate guidance into decision-making
        self._integrate_tactical_guidance(guidance)
        
        print(f"📋 Robot {self.robot_id}: Received tactical guidance from mediator {mediator_id}")
    
    def _adapt_to_strategic_role(self, role: str) -> None:
        """
        Adapt learning parameters and behavior based on strategic role
        """
        if role == "primary_rescuer":
            # Primary rescuers should be more exploitative (use learned knowledge)
            self.q_agent.exploration_rate = max(0.05, self.q_agent.exploration_rate * 0.8)
            self.objective_weights['rescue_success'] = 0.5
            self.objective_weights['time_efficiency'] = 0.3
            self.coordination_level = "guided"
            
        elif role == "strategic_rescuer":
            # Strategic rescuers balance learning and performance
            self.q_agent.exploration_rate = max(0.1, self.q_agent.exploration_rate * 0.9)
            self.objective_weights['coordination_bonus'] = 0.2
            self.coordination_level = "coordinated"
            
        elif role == "support_rescuer":
            # Support rescuers can explore more for learning
            self.q_agent.exploration_rate = min(0.3, self.q_agent.exploration_rate * 1.2)
            self.objective_weights['battery_efficiency'] = 0.3
            self.coordination_level = "autonomous"
            
        elif role == "emergency_rescuer":
            # Emergency rescuers prioritize speed and success
            self.q_agent.exploration_rate = 0.01  # Minimal exploration
            self.objective_weights['rescue_success'] = 0.6
            self.objective_weights['time_efficiency'] = 0.4
            self.coordination_level = "coordinated"
            
        elif role == "standby":
            # Standby robots can focus on learning and battery conservation
            self.q_agent.exploration_rate = min(0.5, self.q_agent.exploration_rate * 1.5)
            self.objective_weights['battery_efficiency'] = 0.5
            self.coordination_level = "autonomous"
    
    def _integrate_tactical_guidance(self, guidance: Dict) -> None:
        """
        Integrate tactical guidance into Q-learning decision process
        """
        # Adjust priority weights based on guidance
        if guidance.get('priority_adjustment'):
            priority_factor = guidance['priority_adjustment']
            self.objective_weights['time_efficiency'] = priority_factor
            self.objective_weights['rescue_success'] = 1 - priority_factor
        
        # Adjust resource conservation behavior
        if guidance.get('resource_conservation_mode'):
            self.objective_weights['battery_efficiency'] = 0.4
            # Increase battery threshold for actions
            self.low_battery_threshold = 30
        
        # Adjust coordination intensity
        if guidance.get('coordination_intensity'):
            coord_intensity = guidance['coordination_intensity']
            self.objective_weights['coordination_bonus'] = coord_intensity * 0.2
    
    def update(self, environment, current_step=0, communication=None) -> None:
        """
        Enhanced update method with hierarchical coordination integration
        """
        self.current_step = current_step
        
        # Store current state before update
        old_battery = self.current_battery
        old_position = self.position
        old_state = self.state
        
        # Check for coordination updates
        self._process_coordination_updates(communication)
        
        # Call parent update for basic behavior
        super().update(environment, current_step, communication)
        
        # Enhanced learning and coordination after action
        if self.learning_enabled and self.state != RobotState.AT_BASE:
            self._update_hierarchical_learning(old_position, old_battery, old_state, environment)
        
        # Update performance tracking
        self._update_performance_tracking()
        
        # Periodic self-optimization
        if current_step > 0 and current_step % 100 == 0:
            self._perform_self_optimization()
    
    def _process_coordination_updates(self, communication) -> None:
        """
        Process updates from coordination hierarchy
        """
        if not communication:
            return
        
        # Check for strategic role updates
        messages = communication.get_messages(f"robot_{self.robot_id}")
        
        for message in messages:
            if hasattr(message, 'data') and message.data:
                # Process coordinator messages
                if message.data.get('strategic_role'):
                    self.set_strategic_role(message.data['strategic_role'])
                
                # Process mediator guidance
                if message.data.get('tactical_guidance'):
                    self.receive_tactical_guidance(message.data['tactical_guidance'])
                
                # Process priority adjustments
                if message.data.get('priority_score'):
                    self._adjust_mission_priority(message.data['priority_score'])
    
    def _adjust_mission_priority(self, priority_score: float) -> None:
        """
        Adjust behavior based on mission priority score from mediators
        """
        if priority_score > 80:  # High priority mission
            # Increase focus on success and speed
            self.objective_weights['rescue_success'] = 0.5
            self.objective_weights['time_efficiency'] = 0.4
            # Reduce exploration for more direct action
            self.q_agent.exploration_rate *= 0.8
            
        elif priority_score < 40:  # Low priority mission
            # Allow more exploration and battery conservation
            self.objective_weights['battery_efficiency'] = 0.3
            self.q_agent.exploration_rate *= 1.2
    
    def _handle_searching_state(self, environment, communication=None) -> None:
        """
        Enhanced searching with hierarchical decision-making
        """
        # Check if person found at current location
        if environment.person_at_location(*self.position):
            self.target_person = self.position
            self.state = RobotState.DELIVERING
            print(f"🎯 Learning Robot {self.robot_id} found person at {self.position}!")
            return
        
        # Check for assigned location from coordination system
        if self.assigned_location:
            if communication and communication.is_robot_near_location(self.position, self.assigned_location):
                if environment.person_at_location(*self.assigned_location):
                    self.target_person = self.assigned_location
                    self.state = RobotState.DELIVERING
                    print(f"🎯 Learning Robot {self.robot_id} reached assigned person at {self.assigned_location}")
                    return
                else:
                    print(f"⚠️ Learning Robot {self.robot_id}: Person at {self.assigned_location} already rescued")
                    self.assigned_location = None
            else:
                # Use hierarchical approach to move towards assigned location
                self._move_with_hierarchical_intelligence(self.assigned_location, environment)
                return
        
        # No assigned location - explore using enhanced Q-learning
        self._explore_with_enhanced_q_learning(environment)
    
    def _move_with_hierarchical_intelligence(self, target: Tuple[int, int], environment) -> None:
        """
        Enhanced movement using hierarchical decision-making
        """
        # Strategic level: Consider role and coordination requirements
        strategic_factor = self._calculate_strategic_factor(target, environment)
        
        # Tactical level: Consider mediator guidance and situation
        tactical_factor = self._calculate_tactical_factor(target, environment)
        
        # Operational level: Use Q-learning with coordination bonuses
        operational_decision = self._make_operational_decision(target, environment, strategic_factor, tactical_factor)
        
        if operational_decision == "direct_path":
            self._move_towards_target(target, environment)
        elif operational_decision == "q_learning_path":
            self._move_with_enhanced_q_learning(target, environment)
        elif operational_decision == "coordinated_path":
            self._move_with_coordination_awareness(target, environment)
    
    def _calculate_strategic_factor(self, target: Tuple[int, int], environment) -> float:
        """
        Calculate strategic decision factor based on role and situation
        """
        factor = 1.0
        
        # Role-based strategic considerations
        if self.strategic_role == "emergency_rescuer":
            factor = 2.0  # Prioritize direct action
        elif self.strategic_role == "strategic_rescuer":
            factor = 1.5  # Balanced approach
        elif self.strategic_role == "support_rescuer":
            factor = 0.8  # More conservative
        
        # Distance and urgency considerations
        distance = abs(self.position[0] - target[0]) + abs(self.position[1] - target[1])
        if distance > 10:
            factor *= 1.2  # Increase urgency for distant targets
        
        return factor
    
    def _calculate_tactical_factor(self, target: Tuple[int, int], environment) -> float:
        """
        Calculate tactical decision factor based on mediator guidance
        """
        factor = 1.0
        
        # Tactical guidance considerations
        if self.tactical_guidance.get('resource_conservation_mode'):
            factor *= 0.8  # More conservative approach
        
        if self.tactical_guidance.get('coordination_intensity', 0) > 0.7:
            factor *= 1.3  # Increase coordination awareness
        
        # Terrain and environmental factors
        elevation = environment.get_elevation(*target)
        if elevation > 2000:  # High elevation target
            factor *= 1.1  # Slightly more urgent
        
        return factor
    
    def _make_operational_decision(self, target: Tuple[int, int], environment, 
                                  strategic_factor: float, tactical_factor: float) -> str:
        """
        Make operational-level movement decision
        """
        combined_factor = strategic_factor * tactical_factor
        
        # Decision logic based on combined factors
        if combined_factor > 1.5:
            return "direct_path"  # High urgency - direct approach
        elif combined_factor > 1.0:
            return "q_learning_path"  # Moderate - use learned knowledge
        else:
            return "coordinated_path"  # Conservative - coordinate with others
    
    def _move_with_enhanced_q_learning(self, target: Optional[Tuple[int, int]], environment) -> None:
        """
        Enhanced Q-learning movement with coordination awareness
        """
        # Get current state representation with coordination features
        state_key = self._get_enhanced_state_key(target, environment)
        
        # Get valid actions
        valid_actions = self._get_valid_actions(environment)
        
        if not valid_actions:
            return
        
        # Choose action using enhanced Q-learning with coordination bonuses
        action = self._choose_enhanced_action(state_key, valid_actions, target, environment)
        
        # Store for learning update
        self.previous_state_key = state_key
        self.previous_action = action
        self.previous_position = self.position
        
        # Execute action
        self._execute_action(action, environment)
    
    def _get_enhanced_state_key(self, target: Optional[Tuple[int, int]], environment) -> str:
        """
        Get enhanced state key including coordination features
        """
        # Base state from Q-agent
        base_state = self.q_agent.get_state_key(
            position=self.position,
            target=target,
            battery_level=self.current_battery,
            has_kit=self.has_first_aid_kit
        )
        
        # Add coordination features
        role_modifier = self.strategic_role[:3]  # First 3 chars of role
        coordination_level = self.coordination_level[:3]  # First 3 chars
        
        # Enhanced state includes coordination context
        enhanced_state = f"{base_state}|{role_modifier}|{coordination_level}"
        
        return enhanced_state
    
    def _choose_enhanced_action(self, state_key: str, valid_actions: List[int], 
                               target: Optional[Tuple[int, int]], environment) -> int:
        """
        Choose action using enhanced Q-learning with coordination bonuses
        """
        # Base Q-learning action selection
        base_action = self.q_agent.choose_action(state_key, valid_actions, self.position)
        
        # Apply coordination bonuses if in coordinated mode
        if self.coordination_level == "coordinated" and len(valid_actions) > 1:
            action_scores = {}
            
            for action in valid_actions:
                # Base Q-value
                q_value = self.q_agent.q_table[state_key][action]
                
                # Coordination bonus
                coord_bonus = self._calculate_coordination_bonus(action, target, environment)
                
                # Strategic role bonus
                role_bonus = self._calculate_role_bonus(action, target, environment)
                
                action_scores[action] = q_value + coord_bonus + role_bonus
            
            # Choose action with highest combined score
            return max(action_scores.items(), key=lambda x: x[1])[0]
        
        return base_action
    
    def _calculate_coordination_bonus(self, action: int, target: Optional[Tuple[int, int]], 
                                    environment) -> float:
        """
        Calculate coordination bonus for action selection
        """
        bonus = 0.0
        
        # Bonus for actions that align with tactical guidance
        if self.tactical_guidance.get('preferred_direction'):
            preferred_dir = self.tactical_guidance['preferred_direction']
            action_dir = self._get_action_direction(action)
            if action_dir == preferred_dir:
                bonus += 2.0
        
        # Bonus for actions that support coordination
        weight = self.objective_weights.get('coordination_bonus', 0.1)
        bonus *= weight * 10  # Scale coordination bonus
        
        return bonus
    
    def _calculate_role_bonus(self, action: int, target: Optional[Tuple[int, int]], 
                             environment) -> float:
        """
        Calculate strategic role bonus for action selection
        """
        bonus = 0.0
        
        if self.strategic_role == "primary_rescuer":
            # Primary rescuers get bonus for direct, efficient actions
            if target:
                next_pos = self._get_position_from_action(action)
                current_dist = abs(self.position[0] - target[0]) + abs(self.position[1] - target[1])
                next_dist = abs(next_pos[0] - target[0]) + abs(next_pos[1] - target[1])
                if next_dist < current_dist:
                    bonus += 1.5  # Bonus for getting closer to target
        
        elif self.strategic_role == "support_rescuer":
            # Support rescuers get bonus for energy-efficient actions
            if action in [0, 2, 4, 6]:  # Orthogonal movements (more efficient)
                bonus += 1.0
        
        elif self.strategic_role == "emergency_rescuer":
            # Emergency rescuers get bonus for fast, direct actions
            if target:
                next_pos = self._get_position_from_action(action)
                if self._is_direct_path_action(action, target):
                    bonus += 2.0
        
        return bonus
    
    def _move_with_coordination_awareness(self, target: Tuple[int, int], environment) -> None:
        """
        Move with high coordination awareness (conservative approach)
        """
        # Get current positions of other robots (if available through Q-agent memory)
        other_positions = self._get_other_robot_positions()
        
        # Calculate safe path that avoids conflicts
        safe_actions = self._get_coordination_safe_actions(target, environment, other_positions)
        
        if safe_actions:
            # Use Q-learning among safe actions
            state_key = self._get_enhanced_state_key(target, environment)
            action = self.q_agent.choose_action(state_key, safe_actions, self.position)
            self._execute_action(action, environment)
        else:
            # Fallback to standard Q-learning
            self._move_with_enhanced_q_learning(target, environment)
    
    def _get_other_robot_positions(self) -> List[Tuple[int, int]]:
        """
        Get positions of other robots (simplified - could be enhanced with communication)
        """
        # This would ideally come from shared knowledge or communication
        # For now, return empty list (could be enhanced in future)
        return []
    
    def _get_coordination_safe_actions(self, target: Tuple[int, int], environment, 
                                     other_positions: List[Tuple[int, int]]) -> List[int]:
        """
        Get actions that are safe from coordination perspective
        """
        all_valid_actions = self._get_valid_actions(environment)
        safe_actions = []
        
        for action in all_valid_actions:
            next_pos = self._get_position_from_action(action)
            
            # Check if next position conflicts with other robots
            if next_pos not in other_positions:
                safe_actions.append(action)
        
        return safe_actions if safe_actions else all_valid_actions
    
    def _update_hierarchical_learning(self, old_position: Tuple[int, int], old_battery: int,
                                    old_state: RobotState, environment) -> None:
        """
        Update Q-learning with hierarchical and multi-objective considerations
        """
        if self.previous_state_key is None or self.previous_action is None:
            return
        
        # Calculate multi-objective reward
        battery_used = old_battery - self.current_battery
        found_person = (self.state == RobotState.DELIVERING and old_state == RobotState.SEARCHING)
        rescued_person = (self.state == RobotState.RETURNING and old_state == RobotState.DELIVERING)
        
        # Base reward from Q-agent
        base_reward = self.q_agent.calculate_reward(
            position=old_position,
            next_position=self.position,
            found_person=found_person,
            rescued_person=rescued_person,
            battery_used=battery_used,
            terrain_elevation=environment.get_elevation(*self.position),
            is_mountain_area=environment.is_mountain_area(*self.position)
        )
        
        # Multi-objective enhancements
        coordination_reward = self._calculate_coordination_reward()
        strategic_compliance_reward = self._calculate_strategic_compliance_reward()
        efficiency_reward = self._calculate_efficiency_reward(battery_used, found_person, rescued_person)
        
        # Weighted combination of rewards
        total_reward = (
            base_reward * self.objective_weights['rescue_success'] +
            coordination_reward * self.objective_weights['coordination_bonus'] +
            strategic_compliance_reward * 0.1 +
            efficiency_reward * self.objective_weights['battery_efficiency']
        )
        
        self.episode_rewards.append(total_reward)
        
        # Update Q-value with enhanced reward
        new_target = self.target_person or self.assigned_location
        new_state_key = self._get_enhanced_state_key(new_target, environment)
        valid_next_actions = self._get_valid_actions(environment)
        
        self.q_agent.update_q_value(
            state_key=self.previous_state_key,
            action=self.previous_action,
            reward=total_reward,
            next_state_key=new_state_key,
            valid_next_actions=valid_next_actions
        )
        
        # Check for episode completion
        if self.state == RobotState.AT_BASE and old_state != RobotState.AT_BASE:
            self._complete_hierarchical_learning_episode()
    
    def _calculate_coordination_reward(self) -> float:
        """
        Calculate reward for coordination behavior
        """
        reward = 0.0
        
        # Reward for following tactical guidance
        if self.tactical_guidance and self.coordination_level == "coordinated":
            reward += 5.0
        
        # Reward for strategic compliance
        compliance_bonus = self.strategic_compliance_score * 3.0
        reward += compliance_bonus
        
        # Penalty for coordination conflicts (if any were detected)
        # This would be enhanced with actual conflict detection
        
        return reward
    
    def _calculate_strategic_compliance_reward(self) -> float:
        """
        Calculate reward for strategic role compliance
        """
        reward = 0.0
        
        # Check if behavior aligns with strategic role
        if self.strategic_role == "primary_rescuer" and self.state in [RobotState.SEARCHING, RobotState.DELIVERING]:
            reward += 2.0
        elif self.strategic_role == "standby" and self.state == RobotState.AT_BASE:
            reward += 1.0
        elif self.strategic_role == "emergency_rescuer" and self.state != RobotState.AT_BASE:
            reward += 3.0
        
        return reward
    
    def _calculate_efficiency_reward(self, battery_used: int, found_person: bool, 
                                   rescued_person: bool) -> float:
        """
        Calculate efficiency-based reward
        """
        reward = 0.0
        
        # Battery efficiency
        if battery_used > 0:
            if rescued_person:
                efficiency = 20 / battery_used  # High reward for efficient rescues
                reward += efficiency
            elif found_person:
                efficiency = 10 / battery_used  # Medium reward for efficient finding
                reward += efficiency
            else:
                efficiency = 2 / battery_used   # Small reward for efficient movement
                reward += efficiency
        
        return reward
    
    def _complete_hierarchical_learning_episode(self) -> None:
        """
        Complete hierarchical learning episode with enhanced analytics
        """
        if self.episode_rewards:
            total_episode_reward = sum(self.episode_rewards)
            avg_reward = total_episode_reward / len(self.episode_rewards)
            
            # Update performance tracking
            self.performance_history.append({
                'episode': self.learning_episodes_completed,
                'total_reward': total_episode_reward,
                'avg_reward': avg_reward,
                'strategic_role': self.strategic_role,
                'coordination_level': self.coordination_level,
                'rescues': self.successful_rescues,
                'exploration_rate': self.q_agent.exploration_rate,
                'timestamp': time.time()
            })
            
            print(f"🧠 Robot {self.robot_id} Episode {self.learning_episodes_completed}:")
            print(f"   Total reward: {total_episode_reward:.2f}, Avg: {avg_reward:.2f}")
            print(f"   Role: {self.strategic_role}, ε: {self.q_agent.exploration_rate:.3f}")
            
            # Decay exploration rate
            self.q_agent.decay_exploration_rate()
            
            # Reset episode tracking
            self.episode_rewards = []
            self.learning_episodes_completed += 1
            self.episode_start_position = self.position
    
    def _update_performance_tracking(self) -> None:
        """
        Update performance tracking metrics
        """
        # Update strategic compliance score
        if self.strategic_role != "unassigned":
            if self._is_complying_with_strategic_role():
                self.strategic_compliance_score = min(1.0, self.strategic_compliance_score + 0.01)
            else:
                self.strategic_compliance_score = max(0.0, self.strategic_compliance_score - 0.02)
        
        # Track coordination effectiveness
        if self.coordination_interactions > 0:
            recent_interactions = self.coordination_interactions
            # Simplified effectiveness calculation
            effectiveness = min(1.0, self.successful_rescues / max(1, recent_interactions * 0.1))
            self.coordination_effectiveness[time.time()] = effectiveness
    
    def _is_complying_with_strategic_role(self) -> bool:
        """
        Check if current behavior complies with strategic role
        """
        if self.strategic_role == "primary_rescuer":
            return self.state in [RobotState.SEARCHING, RobotState.DELIVERING, RobotState.RETURNING]
        elif self.strategic_role == "standby":
            return self.state == RobotState.AT_BASE
        elif self.strategic_role == "emergency_rescuer":
            return self.state != RobotState.AT_BASE or self.current_battery < 30
        elif self.strategic_role == "support_rescuer":
            return True  # Support rescuers have flexible compliance
        
        return True  # Default compliance for unassigned or unknown roles
    
    def _perform_self_optimization(self) -> None:
        """
        Perform periodic self-optimization based on performance
        """
        if len(self.performance_history) < 3:
            return
        
        # Analyze recent performance trends
        recent_performance = self.performance_history[-3:]
        avg_recent_reward = np.mean([p['avg_reward'] for p in recent_performance])
        
        # Check if performance is declining
        if len(self.performance_history) >= 6:
            older_performance = self.performance_history[-6:-3]
            avg_older_reward = np.mean([p['avg_reward'] for p in older_performance])
            
            if avg_recent_reward < avg_older_reward * 0.9:  # 10% decline
                print(f"📉 Robot {self.robot_id}: Performance declining, adjusting parameters")
                self._adjust_learning_parameters_for_improvement()
        
        # Check specific adaptation triggers
        current_success_rate = self.successful_rescues / max(1, len(self.performance_history))
        
        if current_success_rate < self.adaptation_triggers['low_success_rate']:
            self._adapt_for_better_success_rate()
        
        if self.strategic_compliance_score < self.adaptation_triggers['poor_coordination']:
            self._adapt_for_better_coordination()
    
    def _adjust_learning_parameters_for_improvement(self) -> None:
        """
        Adjust learning parameters to improve performance
        """
        # Increase exploration if stuck in local optimum
        if self.q_agent.exploration_rate < 0.2:
            self.q_agent.exploration_rate = min(0.3, self.q_agent.exploration_rate * 1.5)
            print(f"🔄 Robot {self.robot_id}: Increased exploration rate to {self.q_agent.exploration_rate:.3f}")
        
        # Adjust objective weights to emphasize successful patterns
        if self.successful_rescues > 0:
            self.objective_weights['rescue_success'] = min(0.6, self.objective_weights['rescue_success'] + 0.1)
    
    def _adapt_for_better_success_rate(self) -> None:
        """
        Adapt behavior for better success rate
        """
        # Focus more on exploitation and coordination
        self.q_agent.exploration_rate *= 0.8
        self.objective_weights['coordination_bonus'] += 0.05
        self.coordination_level = "coordinated"
        
        print(f"🎯 Robot {self.robot_id}: Adapting for better success rate")
    
    def _adapt_for_better_coordination(self) -> None:
        """
        Adapt behavior for better coordination
        """
        # Increase coordination awareness
        self.objective_weights['coordination_bonus'] = min(0.3, self.objective_weights['coordination_bonus'] + 0.1)
        self.coordination_level = "coordinated"
        
        print(f"🤝 Robot {self.robot_id}: Adapting for better coordination")
    
    def get_enhanced_learning_status(self) -> Dict:
        """
        Get comprehensive learning status including hierarchical features
        """
        base_status = self.get_learning_status()
        
        enhanced_status = {
            **base_status,
            'strategic_role': self.strategic_role,
            'coordination_level': self.coordination_level,
            'strategic_compliance_score': self.strategic_compliance_score,
            'coordination_interactions': self.coordination_interactions,
            'objective_weights': self.objective_weights,
            'performance_trend': self._calculate_performance_trend(),
            'last_coordinator_update': self.last_coordinator_update,
            'last_mediator_interaction': self.last_mediator_interaction,
            'meta_learning_enabled': self.meta_learning_enabled,
            'collaborative_learning_enabled': self.collaborative_learning_enabled
        }
        
        return enhanced_status
    
    def _calculate_performance_trend(self) -> str:
        """
        Calculate performance trend over recent episodes
        """
        if len(self.performance_history) < 3:
            return "insufficient_data"
        
        recent_rewards = [p['avg_reward'] for p in self.performance_history[-3:]]
        if len(self.performance_history) >= 6:
            older_rewards = [p['avg_reward'] for p in self.performance_history[-6:-3]]
            
            recent_avg = np.mean(recent_rewards)
            older_avg = np.mean(older_rewards)
            
            if recent_avg > older_avg * 1.1:
                return "improving"
            elif recent_avg < older_avg * 0.9:
                return "declining"
            else:
                return "stable"
        
        return "learning"
    
    # Helper methods
    def _get_action_direction(self, action: int) -> str:
        """Get direction name for action"""
        directions = ["north", "northeast", "east", "southeast", 
                     "south", "southwest", "west", "northwest"]
        return directions[action] if 0 <= action < len(directions) else "unknown"
    
    def _is_direct_path_action(self, action: int, target: Tuple[int, int]) -> bool:
        """Check if action moves directly toward target"""
        next_pos = self._get_position_from_action(action)
        current_dist = abs(self.position[0] - target[0]) + abs(self.position[1] - target[1])
        next_dist = abs(next_pos[0] - target[0]) + abs(next_pos[1] - target[1])
        return next_dist < current_dist