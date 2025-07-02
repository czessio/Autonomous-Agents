# Enhanced Coordinator agent for strategic mission planning and high-level resource allocation
# Provides strategic oversight for the entire rescue operation

from typing import List, Tuple, Dict, Optional
from enum import Enum
import numpy as np
import time


class CoordinatorState(Enum):
    STRATEGIC_PLANNING = "strategic_planning"
    RESOURCE_ALLOCATION = "resource_allocation"
    MISSION_MONITORING = "mission_monitoring"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    CRISIS_MANAGEMENT = "crisis_management"


class MissionPhase(Enum):
    INITIAL_DEPLOYMENT = "initial_deployment"
    ACTIVE_OPERATIONS = "active_operations"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    MISSION_COMPLETION = "mission_completion"


class CoordinatorAgent:
    """
    Strategic coordinator that provides high-level mission planning and oversight.
    
    Key responsibilities:
    1. Strategic mission planning and phase management
    2. High-level resource allocation across multiple mediators
    3. Performance monitoring and adaptive strategy adjustment
    4. Crisis management and emergency response coordination
    5. Integration with Q-learning and multi-agent systems
    6. Long-term mission optimization
    
    Integration with Novel Mode:
    - Coordinates multiple mediator agents for tactical execution
    - Integrates with collective Q-learning for strategic insights
    - Supports dynamic role assignment at strategic level
    - Provides oversight for predictive resource allocation
    """
    
    def __init__(self, agent_id: int, position: Tuple[int, int]):
        self.agent_id = agent_id
        self.position = position  # Central command position
        self.state = CoordinatorState.STRATEGIC_PLANNING
        
        # Strategic mission management
        self.current_mission_phase = MissionPhase.INITIAL_DEPLOYMENT
        self.mission_objectives = {}  # High-level objectives and KPIs
        self.strategic_priorities = []  # Long-term strategic priorities
        self.resource_allocation_strategy = "adaptive"  # Strategy mode
        
        # Performance monitoring and analytics
        self.mission_metrics = {
            'total_missions_assigned': 0,
            'successful_missions': 0,
            'failed_missions': 0,
            'average_mission_time': 0,
            'resource_efficiency': 0,
            'strategic_accuracy': 0
        }
        
        # Multi-agent coordination
        self.mediator_performance = {}  # Track mediator effectiveness
        self.robot_strategic_assignments = {}  # Strategic role assignments
        self.drone_coordination_tasks = {}  # Drone coordination
        
        # Adaptive strategy components
        self.strategy_parameters = {
            'risk_tolerance': 0.7,        # 0-1 scale
            'efficiency_priority': 0.8,   # vs speed priority
            'resource_conservation': 0.6, # Battery/resource management
            'exploration_emphasis': 0.5,  # vs exploitation
            'coordination_intensity': 0.8 # How much coordination to apply
        }
        
        # Learning and adaptation
        self.strategy_history = []
        self.performance_trends = []
        self.adaptation_triggers = {
            'low_success_rate': 0.6,
            'high_mission_time': 100,
            'low_efficiency': 0.3,
            'high_failure_rate': 0.3
        }
        
        # Crisis management
        self.crisis_threshold = {
            'multiple_failures': 3,
            'critical_resource_shortage': 0.2,
            'mission_time_exceeded': 1.5
        }
        
        # External coordination
        self.external_updates = []  # Updates from simulation controller
        self.strategic_recommendations = []  # For simulation system
        
        print(f"🏛️ Strategic Coordinator {agent_id} initialised at command position {position}")
        print(f"   Strategic oversight with adaptive learning enabled")
    
    def update(self, environment, robots: List, drones: List, communication, current_step: int = 0) -> None:
        """
        Main strategic update cycle - operates at higher level than tactical agents
        """
        self.current_step = current_step
        
        # Update mission phase based on current situation
        self._update_mission_phase(environment, robots, drones)
        
        # Strategic state machine execution
        if self.state == CoordinatorState.STRATEGIC_PLANNING:
            self._execute_strategic_planning(environment, robots, drones, communication)
        elif self.state == CoordinatorState.RESOURCE_ALLOCATION:
            self._execute_resource_allocation(robots, drones, communication)
        elif self.state == CoordinatorState.MISSION_MONITORING:
            self._execute_mission_monitoring(robots, drones, communication)
        elif self.state == CoordinatorState.PERFORMANCE_ANALYSIS:
            self._execute_performance_analysis()
        elif self.state == CoordinatorState.CRISIS_MANAGEMENT:
            self._execute_crisis_management(robots, drones, communication)
        
        # Periodic strategic adaptation
        if current_step > 0 and current_step % 50 == 0:
            self._adapt_strategy()
        
        # Update performance metrics
        self._update_strategic_metrics(robots, drones)
    
    def _update_mission_phase(self, environment, robots: List, drones: List) -> None:
        """
        Determine current mission phase based on overall situation
        """
        # Count active vs idle agents
        active_robots = sum(1 for r in robots if r.state.value not in ['at_base'])
        active_drones = sum(1 for d in drones if d.state.value not in ['at_base'])
        total_agents = len(robots) + len(drones)
        
        # Count remaining vs rescued persons
        remaining_persons = len(environment.missing_persons)
        rescued_persons = len(environment.rescued_persons)
        total_persons = remaining_persons + rescued_persons
        
        # Phase determination logic
        if total_persons == 0 or (rescued_persons == 0 and active_robots == 0):
            self.current_mission_phase = MissionPhase.INITIAL_DEPLOYMENT
        elif remaining_persons == 0:
            self.current_mission_phase = MissionPhase.MISSION_COMPLETION
        elif active_robots / len(robots) > 0.7:  # Most robots active
            self.current_mission_phase = MissionPhase.ACTIVE_OPERATIONS
        else:
            self.current_mission_phase = MissionPhase.RESOURCE_OPTIMIZATION
    
    def _execute_strategic_planning(self, environment, robots: List, drones: List, communication) -> None:
        """
        Execute strategic planning phase
        """
        # Analyse overall mission situation
        mission_analysis = self._analyse_mission_situation(environment, robots, drones)
        
        # Set strategic objectives based on analysis
        self._set_strategic_objectives(mission_analysis)
        
        # Plan resource allocation strategy
        self._plan_resource_allocation_strategy(mission_analysis, robots, drones)
        
        # Generate strategic recommendations
        self._generate_strategic_recommendations(mission_analysis)
        
        # Transition to resource allocation
        self.state = CoordinatorState.RESOURCE_ALLOCATION
        
        print(f"📋 Coordinator: Strategic planning completed for {self.current_mission_phase.value}")
    
    def _analyse_mission_situation(self, environment, robots: List, drones: List) -> Dict:
        """
        Comprehensive analysis of current mission situation
        """
        analysis = {
            'mission_phase': self.current_mission_phase,
            'persons_remaining': len(environment.missing_persons),
            'persons_rescued': len(environment.rescued_persons),
            'robot_availability': {},
            'drone_status': {},
            'resource_constraints': {},
            'performance_indicators': {},
            'risk_factors': [],
            'opportunities': []
        }
        
        # Robot availability analysis
        for robot in robots:
            robot_id = robot.robot_id
            analysis['robot_availability'][robot_id] = {
                'state': robot.state.value,
                'battery': robot.current_battery,
                'has_kit': robot.has_first_aid_kit,
                'availability_score': self._calculate_robot_availability_score(robot)
            }
        
        # Drone status analysis
        for drone in drones:
            analysis['drone_status'][drone.drone_id] = {
                'state': drone.state.value,
                'battery': drone.current_battery,
                'areas_explored': getattr(drone, 'total_areas_explored', 0),
                'persons_found': len(getattr(drone, 'found_persons', []))
            }
        
        # Resource constraints
        total_battery = sum(r.current_battery for r in robots) + sum(d.current_battery for d in drones)
        available_kits = sum(1 for r in robots if r.has_first_aid_kit)
        
        analysis['resource_constraints'] = {
            'total_battery_remaining': total_battery,
            'available_first_aid_kits': available_kits,
            'battery_per_person': total_battery / max(1, len(environment.missing_persons)),
            'kit_shortage': len(environment.missing_persons) > available_kits
        }
        
        # Performance indicators
        if hasattr(self, 'mission_start_time'):
            mission_duration = time.time() - self.mission_start_time
            analysis['performance_indicators'] = {
                'mission_duration': mission_duration,
                'rescue_rate': len(environment.rescued_persons) / max(1, mission_duration / 60),
                'efficiency_score': len(environment.rescued_persons) / max(1, (1000 - total_battery))
            }
        else:
            self.mission_start_time = time.time()
        
        # Risk factors identification
        if total_battery < 300:  # Low overall battery
            analysis['risk_factors'].append('low_battery_reserves')
        if available_kits < len(environment.missing_persons):
            analysis['risk_factors'].append('insufficient_first_aid_kits')
        if sum(1 for r in robots if r.current_battery > 50) < 2:
            analysis['risk_factors'].append('few_operational_robots')
        
        # Opportunity identification
        if len(environment.missing_persons) <= available_kits:
            analysis['opportunities'].append('sufficient_resources')
        if total_battery > 600:
            analysis['opportunities'].append('abundant_battery')
        
        return analysis
    
    def _calculate_robot_availability_score(self, robot) -> float:
        """
        Calculate availability score for strategic planning
        """
        score = 0.0
        
        # Base availability
        if robot.state.value == 'at_base':
            score += 40
        elif robot.state.value in ['searching', 'delivering']:
            score += 20  # Partially available
        
        # Battery factor
        score += (robot.current_battery / 100) * 30
        
        # Kit availability
        if robot.has_first_aid_kit:
            score += 20
        
        # Experience factor (for learning robots)
        if hasattr(robot, 'q_agent') and hasattr(robot.q_agent, 'episode_count'):
            experience_bonus = min(robot.q_agent.episode_count / 10, 10)
            score += experience_bonus
        
        return min(100, score)
    
    def _set_strategic_objectives(self, mission_analysis: Dict) -> None:
        """
        Set strategic objectives based on mission analysis
        """
        self.mission_objectives.clear()
        
        persons_remaining = mission_analysis['persons_remaining']
        available_resources = mission_analysis['resource_constraints']
        
        # Primary objective: rescue efficiency
        if available_resources['kit_shortage']:
            self.mission_objectives['primary'] = 'maximize_rescues_with_limited_kits'
            self.strategy_parameters['efficiency_priority'] = 0.9
        else:
            self.mission_objectives['primary'] = 'minimize_rescue_time'
            self.strategy_parameters['efficiency_priority'] = 0.6
        
        # Secondary objectives based on phase
        if self.current_mission_phase == MissionPhase.INITIAL_DEPLOYMENT:
            self.mission_objectives['secondary'] = 'rapid_deployment'
        elif self.current_mission_phase == MissionPhase.ACTIVE_OPERATIONS:
            self.mission_objectives['secondary'] = 'coordination_optimization'
        elif self.current_mission_phase == MissionPhase.RESOURCE_OPTIMIZATION:
            self.mission_objectives['secondary'] = 'resource_conservation'
        
        # Risk mitigation objectives
        if 'low_battery_reserves' in mission_analysis['risk_factors']:
            self.mission_objectives['risk_mitigation'] = 'battery_conservation'
            self.strategy_parameters['resource_conservation'] = 0.9
        
        print(f"🎯 Coordinator: Objectives set - Primary: {self.mission_objectives.get('primary', 'unknown')}")
    
    def _plan_resource_allocation_strategy(self, mission_analysis: Dict, robots: List, drones: List) -> None:
        """
        Plan high-level resource allocation strategy
        """
        # Determine optimal allocation strategy based on situation
        available_robots = sum(1 for r in robots if mission_analysis['robot_availability'][r.robot_id]['availability_score'] > 50)
        persons_per_robot = mission_analysis['persons_remaining'] / max(1, available_robots)
        
        # Strategy selection
        if persons_per_robot > 3:
            self.resource_allocation_strategy = "parallel_aggressive"
            print("📈 Coordinator: Strategy - Parallel aggressive deployment")
        elif mission_analysis['resource_constraints']['battery_per_person'] < 50:
            self.resource_allocation_strategy = "conservative_sequential"
            print("⚡ Coordinator: Strategy - Conservative sequential deployment")
        else:
            self.resource_allocation_strategy = "balanced_coordination"
            print("⚖️ Coordinator: Strategy - Balanced coordination")
        
        # Adjust strategy parameters
        if self.resource_allocation_strategy == "parallel_aggressive":
            self.strategy_parameters['coordination_intensity'] = 0.6
            self.strategy_parameters['efficiency_priority'] = 0.5
        elif self.resource_allocation_strategy == "conservative_sequential":
            self.strategy_parameters['resource_conservation'] = 0.9
            self.strategy_parameters['coordination_intensity'] = 0.9
        else:
            # Balanced approach - use default parameters
            pass
    
    def _generate_strategic_recommendations(self, mission_analysis: Dict) -> None:
        """
        Generate strategic recommendations for tactical agents
        """
        self.strategic_recommendations.clear()
        
        # Recommendations for mediators
        if 'insufficient_first_aid_kits' in mission_analysis['risk_factors']:
            self.strategic_recommendations.append({
                'target': 'mediators',
                'recommendation': 'prioritize_high_success_probability_rescues',
                'parameters': {'min_success_probability': 0.8}
            })
        
        # Recommendations for Q-learning robots
        if self.current_mission_phase == MissionPhase.RESOURCE_OPTIMIZATION:
            self.strategic_recommendations.append({
                'target': 'learning_robots',
                'recommendation': 'increase_exploitation_ratio',
                'parameters': {'target_exploration_rate': 0.1}
            })
        
        # Recommendations for drones
        unexplored_ratio = 1.0 - (sum(d['areas_explored'] for d in mission_analysis['drone_status'].values()) / 100)
        if unexplored_ratio > 0.3:
            self.strategic_recommendations.append({
                'target': 'drones',
                'recommendation': 'prioritize_systematic_exploration',
                'parameters': {'coverage_priority': 0.8}
            })
    
    def _execute_resource_allocation(self, robots: List, drones: List, communication) -> None:
        """
        Execute high-level resource allocation
        """
        # Strategic robot role assignments
        self._assign_strategic_robot_roles(robots)
        
        # Coordinate drone activities
        self._coordinate_drone_activities(drones)
        
        # Monitor mediator performance and provide guidance
        self._provide_mediator_guidance(communication)
        
        # Transition to monitoring
        self.state = CoordinatorState.MISSION_MONITORING
        
        print(f"📊 Coordinator: Resource allocation completed for {len(robots)} robots, {len(drones)} drones")
    
    def _assign_strategic_robot_roles(self, robots: List) -> None:
        """
        Assign strategic roles to robots based on capabilities and mission needs
        """
        # Clear previous assignments
        self.robot_strategic_assignments.clear()
        
        # Sort robots by capability score
        robots_by_capability = sorted(
            robots,
            key=lambda r: self._calculate_robot_capability_score(r),
            reverse=True
        )
        
        # Assign roles based on strategy
        for i, robot in enumerate(robots_by_capability):
            if i == 0 and len(robots) > 2:
                # Best robot becomes strategic coordinator
                self.robot_strategic_assignments[robot.robot_id] = 'strategic_rescuer'
            elif robot.current_battery > 70 and robot.has_first_aid_kit:
                # High-capability robots for active rescue
                self.robot_strategic_assignments[robot.robot_id] = 'primary_rescuer'
            elif robot.current_battery > 30:
                # Medium-capability robots for support
                self.robot_strategic_assignments[robot.robot_id] = 'support_rescuer'
            else:
                # Low-capability robots on standby
                self.robot_strategic_assignments[robot.robot_id] = 'standby'
        
        # Update mission metrics
        self.mission_metrics['total_missions_assigned'] += len([r for r in robots if r.state.value not in ['at_base']])
    
    def _calculate_robot_capability_score(self, robot) -> float:
        """
        Calculate comprehensive capability score for strategic assignment
        """
        score = 0.0
        
        # Battery capability (40% of score)
        score += (robot.current_battery / 100) * 40
        
        # Kit availability (20% of score)
        if robot.has_first_aid_kit:
            score += 20
        
        # Performance history (20% of score)
        if hasattr(robot, 'persons_rescued'):
            rescue_bonus = min(robot.persons_rescued * 5, 20)
            score += rescue_bonus
        
        # Learning capability (20% of score)
        if hasattr(robot, 'q_agent'):
            # Robots with more learned states are more capable
            learning_bonus = min(len(robot.q_agent.q_table) / 50, 20)
            score += learning_bonus
            
            # Lower exploration rate indicates more learned behavior
            exploitation_bonus = (1 - robot.q_agent.exploration_rate) * 10
            score += exploitation_bonus
        
        return score
    
    def _coordinate_drone_activities(self, drones: List) -> None:
        """
        Coordinate drone activities for optimal coverage and efficiency
        """
        self.drone_coordination_tasks.clear()
        
        for drone in drones:
            if drone.state.value == 'at_base' and drone.current_battery > 50:
                # Assign exploration tasks based on strategy
                if self.resource_allocation_strategy == "parallel_aggressive":
                    self.drone_coordination_tasks[drone.drone_id] = 'rapid_wide_search'
                else:
                    self.drone_coordination_tasks[drone.drone_id] = 'systematic_detailed_search'
            elif drone.state.value == 'exploring':
                # Continue current task but optimize based on strategy
                self.drone_coordination_tasks[drone.drone_id] = 'continue_optimized'
    
    def _provide_mediator_guidance(self, communication) -> None:
        """
        Provide strategic guidance to mediator agents
        """
        # Strategic guidance based on current strategy
        guidance = {
            'priority_adjustment': self.strategy_parameters['efficiency_priority'],
            'resource_conservation_mode': self.strategy_parameters['resource_conservation'] > 0.7,
            'coordination_intensity': self.strategy_parameters['coordination_intensity'],
            'strategic_phase': self.current_mission_phase.value
        }
        
        # Send guidance through communication system if available
        if hasattr(communication, 'send_individual_message'):
            try:
                # Would send to mediators if they were registered in communication
                pass
            except:
                pass
    
    def _execute_mission_monitoring(self, robots: List, drones: List, communication) -> None:
        """
        Monitor ongoing mission performance and adjust as needed
        """
        # Monitor robot performance against strategic assignments
        self._monitor_robot_performance(robots)
        
        # Monitor drone coordination effectiveness
        self._monitor_drone_coordination(drones)
        
        # Check for crisis conditions
        crisis_detected = self._check_for_crisis_conditions(robots, drones)
        
        if crisis_detected:
            self.state = CoordinatorState.CRISIS_MANAGEMENT
        else:
            # Periodic analysis
            if hasattr(self, 'last_analysis_time'):
                if time.time() - self.last_analysis_time > 30:  # Every 30 seconds
                    self.state = CoordinatorState.PERFORMANCE_ANALYSIS
            else:
                self.last_analysis_time = time.time()
                self.state = CoordinatorState.STRATEGIC_PLANNING
    
    def _monitor_robot_performance(self, robots: List) -> None:
        """
        Monitor robot performance against strategic expectations
        """
        for robot in robots:
            strategic_role = self.robot_strategic_assignments.get(robot.robot_id, 'unassigned')
            
            # Check if robot is fulfilling strategic role
            if strategic_role == 'primary_rescuer' and robot.state.value == 'at_base':
                # Primary rescuer should be active
                print(f"⚠️ Coordinator: Primary rescuer Robot {robot.robot_id} is idle")
            elif strategic_role == 'standby' and robot.state.value not in ['at_base', 'returning']:
                # Standby robot should not be active
                print(f"⚠️ Coordinator: Standby Robot {robot.robot_id} unexpectedly active")
    
    def _monitor_drone_coordination(self, drones: List) -> None:
        """
        Monitor drone coordination effectiveness
        """
        for drone in drones:
            coordination_task = self.drone_coordination_tasks.get(drone.drone_id, 'none')
            
            # Check drone performance against coordination expectations
            if coordination_task == 'rapid_wide_search' and drone.state.value == 'waiting':
                # Rapid search drones shouldn't wait long
                if hasattr(drone, 'waiting_time') and drone.waiting_time > 5:
                    print(f"⚠️ Coordinator: Rapid search Drone {drone.drone_id} waiting too long")
    
    def _check_for_crisis_conditions(self, robots: List, drones: List) -> bool:
        """
        Check for crisis conditions requiring immediate strategic intervention
        """
        crisis_indicators = 0
        
        # Check for multiple recent failures
        recent_failures = self.mission_metrics['failed_missions']
        if recent_failures >= self.crisis_threshold['multiple_failures']:
            crisis_indicators += 1
            print(f"🚨 Coordinator: Crisis detected - multiple failures ({recent_failures})")
        
        # Check for critical resource shortage
        operational_robots = sum(1 for r in robots if r.current_battery > 30 and r.has_first_aid_kit)
        total_robots = len(robots)
        if operational_robots / total_robots < self.crisis_threshold['critical_resource_shortage']:
            crisis_indicators += 1
            print(f"🚨 Coordinator: Crisis detected - critical resource shortage")
        
        # Check for mission time exceeded
        if hasattr(self, 'mission_start_time'):
            mission_duration = time.time() - self.mission_start_time
            expected_duration = len(robots) * 60  # Expected 60 seconds per robot
            if mission_duration > expected_duration * self.crisis_threshold['mission_time_exceeded']:
                crisis_indicators += 1
                print(f"🚨 Coordinator: Crisis detected - mission time exceeded")
        
        return crisis_indicators >= 2  # Require multiple indicators for crisis
    
    def _execute_performance_analysis(self) -> None:
        """
        Execute comprehensive performance analysis
        """
        # Calculate current performance metrics
        current_performance = self._calculate_current_performance()
        
        # Compare against historical performance
        self.performance_trends.append(current_performance)
        
        # Identify areas for improvement
        improvement_areas = self._identify_improvement_areas(current_performance)
        
        # Generate strategy adjustments
        if improvement_areas:
            self._generate_strategy_adjustments(improvement_areas)
        
        self.last_analysis_time = time.time()
        self.state = CoordinatorState.STRATEGIC_PLANNING
        
        print(f"📈 Coordinator: Performance analysis completed - Score: {current_performance.get('overall_score', 0):.2f}")
    
    def _calculate_current_performance(self) -> Dict:
        """
        Calculate current performance metrics
        """
        performance = {}
        
        # Basic metrics
        total_missions = self.mission_metrics['total_missions_assigned']
        successful_missions = self.mission_metrics['successful_missions']
        
        if total_missions > 0:
            performance['success_rate'] = successful_missions / total_missions
        else:
            performance['success_rate'] = 0
        
        performance['mission_efficiency'] = self.mission_metrics['resource_efficiency']
        performance['strategic_accuracy'] = self.mission_metrics['strategic_accuracy']
        
        # Calculate overall score
        performance['overall_score'] = (
            performance['success_rate'] * 0.4 +
            performance['mission_efficiency'] * 0.3 +
            performance['strategic_accuracy'] * 0.3
        )
        
        return performance
    
    def _identify_improvement_areas(self, current_performance: Dict) -> List[str]:
        """
        Identify areas needing improvement
        """
        improvement_areas = []
        
        if current_performance['success_rate'] < self.adaptation_triggers['low_success_rate']:
            improvement_areas.append('success_rate')
        
        if current_performance['mission_efficiency'] < self.adaptation_triggers['low_efficiency']:
            improvement_areas.append('efficiency')
        
        if current_performance['strategic_accuracy'] < 0.7:
            improvement_areas.append('strategic_planning')
        
        return improvement_areas
    
    def _generate_strategy_adjustments(self, improvement_areas: List[str]) -> None:
        """
        Generate strategy adjustments based on identified improvement areas
        """
        adjustments_made = []
        
        if 'success_rate' in improvement_areas:
            # Increase coordination intensity
            self.strategy_parameters['coordination_intensity'] = min(1.0, self.strategy_parameters['coordination_intensity'] + 0.1)
            adjustments_made.append('increased_coordination')
        
        if 'efficiency' in improvement_areas:
            # Increase efficiency priority
            self.strategy_parameters['efficiency_priority'] = min(1.0, self.strategy_parameters['efficiency_priority'] + 0.1)
            adjustments_made.append('increased_efficiency_focus')
        
        if 'strategic_planning' in improvement_areas:
            # Increase risk tolerance for more decisive planning
            self.strategy_parameters['risk_tolerance'] = min(1.0, self.strategy_parameters['risk_tolerance'] + 0.1)
            adjustments_made.append('increased_risk_tolerance')
        
        if adjustments_made:
            print(f"🔧 Coordinator: Strategy adjustments made - {', '.join(adjustments_made)}")
    
    def _execute_crisis_management(self, robots: List, drones: List, communication) -> None:
        """
        Execute crisis management procedures
        """
        print(f"🚨 Coordinator: Entering crisis management mode")
        
        # Emergency resource reallocation
        self._emergency_resource_reallocation(robots, drones)
        
        # Emergency strategic adjustments
        self.strategy_parameters['resource_conservation'] = 0.9
        self.strategy_parameters['coordination_intensity'] = 1.0
        self.strategy_parameters['risk_tolerance'] = 0.3
        
        # Force all available resources into action
        for robot in robots:
            if robot.state.value == 'at_base' and robot.current_battery > 20:
                # Override normal assignment logic
                self.robot_strategic_assignments[robot.robot_id] = 'emergency_rescuer'
        
        # Return to strategic planning with crisis parameters
        self.state = CoordinatorState.STRATEGIC_PLANNING
        
        print(f"🚨 Coordinator: Crisis management procedures activated")
    
    def _emergency_resource_reallocation(self, robots: List, drones: List) -> None:
        """
        Emergency reallocation of all available resources
        """
        # Count available resources
        available_robots = [r for r in robots if r.current_battery > 20]
        available_drones = [d for d in drones if d.current_battery > 20]
        
        print(f"🚨 Emergency reallocation: {len(available_robots)} robots, {len(available_drones)} drones available")
        
        # Assign emergency roles
        for robot in available_robots:
            if robot.has_first_aid_kit:
                self.robot_strategic_assignments[robot.robot_id] = 'emergency_rescuer'
            else:
                self.robot_strategic_assignments[robot.robot_id] = 'emergency_support'
        
        for drone in available_drones:
            self.drone_coordination_tasks[drone.drone_id] = 'emergency_search'
    
    def _adapt_strategy(self) -> None:
        """
        Periodic strategy adaptation based on performance
        """
        if len(self.performance_trends) < 2:
            return
        
        # Compare recent performance
        recent_performance = self.performance_trends[-1]
        previous_performance = self.performance_trends[-2]
        
        # Adapt based on performance trends
        if recent_performance['overall_score'] < previous_performance['overall_score']:
            # Performance declining - make conservative adjustments
            self.strategy_parameters['resource_conservation'] = min(1.0, self.strategy_parameters['resource_conservation'] + 0.05)
            print(f"📉 Coordinator: Performance declining - adopting more conservative strategy")
        else:
            # Performance improving - can be more aggressive
            self.strategy_parameters['efficiency_priority'] = min(1.0, self.strategy_parameters['efficiency_priority'] + 0.05)
            print(f"📈 Coordinator: Performance improving - increasing efficiency focus")
    
    def _update_strategic_metrics(self, robots: List, drones: List) -> None:
        """
        Update strategic performance metrics
        """
        # Update mission metrics based on current state
        active_missions = sum(1 for r in robots if r.state.value in ['searching', 'delivering'])
        
        # Update success/failure counts based on robot states
        returning_robots = [r for r in robots if r.state.value == 'returning']
        for robot in returning_robots:
            if hasattr(robot, '_mission_completed') and not robot._mission_completed:
                robot._mission_completed = True
                if robot.persons_rescued > 0:
                    self.mission_metrics['successful_missions'] += 1
                else:
                    self.mission_metrics['failed_missions'] += 1
        
        # Calculate resource efficiency
        total_battery_used = sum((r.max_battery - r.current_battery) for r in robots)
        total_rescues = sum(r.persons_rescued for r in robots)
        
        if total_battery_used > 0:
            self.mission_metrics['resource_efficiency'] = total_rescues / (total_battery_used / 100)
        
        # Calculate strategic accuracy (how well strategic assignments are working)
        correctly_assigned = 0
        total_assigned = 0
        
        for robot in robots:
            if robot.robot_id in self.robot_strategic_assignments:
                total_assigned += 1
                strategic_role = self.robot_strategic_assignments[robot.robot_id]
                
                # Check if robot is performing according to strategic role
                if strategic_role == 'primary_rescuer' and robot.state.value in ['searching', 'delivering']:
                    correctly_assigned += 1
                elif strategic_role == 'standby' and robot.state.value in ['at_base']:
                    correctly_assigned += 1
                elif strategic_role in ['support_rescuer', 'emergency_rescuer'] and robot.state.value != 'at_base':
                    correctly_assigned += 1
        
        if total_assigned > 0:
            self.mission_metrics['strategic_accuracy'] = correctly_assigned / total_assigned
    
    def get_status(self) -> Dict:
        """
        Get current coordinator status and metrics
        """
        return {
            'id': self.agent_id,
            'state': self.state.value,
            'mission_phase': self.current_mission_phase.value,
            'strategy': self.resource_allocation_strategy,
            'missions_assigned': self.mission_metrics['total_missions_assigned'],
            'successful_missions': self.mission_metrics['successful_missions'],
            'strategic_accuracy': self.mission_metrics['strategic_accuracy'],
            'resource_efficiency': self.mission_metrics['resource_efficiency'],
            'active_strategic_assignments': len(self.robot_strategic_assignments),
            'active_drone_tasks': len(self.drone_coordination_tasks)
        }
    
    def get_comprehensive_report(self) -> Dict:
        """
        Get comprehensive strategic report for analysis
        """
        return {
            'coordinator_id': self.agent_id,
            'mission_phase': self.current_mission_phase.value,
            'strategy_parameters': self.strategy_parameters,
            'mission_metrics': self.mission_metrics,
            'robot_strategic_assignments': self.robot_strategic_assignments,
            'drone_coordination_tasks': self.drone_coordination_tasks,
            'performance_trends': self.performance_trends,
            'strategic_recommendations': self.strategic_recommendations,
            'mission_objectives': self.mission_objectives,
            'strategy_history': self.strategy_history
        }