# Enhanced Mediator agent for coordinating rescue operations and resource allocation
# Fully functional integration with the Novel Mode simulation

from typing import List, Tuple, Dict, Optional
from enum import Enum
import numpy as np
import time
from typing import Optional
import matplotlib.pyplot as plt





class MediatorState(Enum):
    MONITORING = "monitoring"
    ALLOCATING = "allocating"
    COORDINATING = "coordinating"
    EMERGENCY = "emergency"
    OPTIMISING = "optimising"


class PriorityLevel(Enum):
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


class MediatorAgent:
    """
    Advanced mediator agent that provides tactical coordination for rescue operations.
    
    Key responsibilities:
    1. Real-time prioritisation of rescue targets based on multiple factors
    2. Intelligent allocation of robots to maximise efficiency
    3. Dynamic battery management and agent recall decisions
    4. Learning from rescue success patterns
    5. Coordination with other mediators and the main coordinator
    
    Integration with Novel Mode:
    - Works with Q-Learning robots to enhance their decision-making
    - Integrates with hierarchical coordination system
    - Supports dynamic role assignment
    - Provides tactical oversight for strategic coordinator
    """
    
    def __init__(self, agent_id: int, position: Tuple[int, int]):
        self.agent_id = agent_id
        self.position = position
        self.state = MediatorState.MONITORING
        
        # Mission tracking and prioritisation
        self.mission_priorities = {}  # location -> priority data
        self.active_assignments = {}  # robot_id -> assignment details
        self.rescue_queue = []        # Prioritised rescue targets
        self.emergency_situations = []
        
        # Performance tracking and learning
        self.assignment_history = []
        self.success_patterns = {}    # Learn what assignment strategies work
        self.terrain_knowledge = {}  # Build knowledge of terrain difficulty
        self.robot_performance_data = {}  # Track individual robot capabilities
        
        # Decision-making parameters
        self.urgency_weights = {
            'elevation': 0.25,        # Higher elevation = more urgent
            'distance_to_robots': 0.20,  # Closer to available robots = higher priority
            'waiting_time': 0.25,     # Longer waiting = more urgent
            'terrain_difficulty': 0.15,  # Difficult terrain = higher priority
            'robot_efficiency': 0.15     # Better robots for harder rescues
        }
        
        # Tactical coordination settings
        self.max_assignments_per_cycle = 3
        self.battery_critical_threshold = 25
        self.battery_recall_threshold = 35
        self.efficiency_learning_rate = 0.1
        
        # Coordination with other agents
        self.coordinator_updates = []
        self.mediator_communications = {}
        
        # Statistics and metrics
        self.assignments_made = 0
        self.successful_rescues = 0
        self.failed_assignments = 0
        self.average_response_time = 0.0
        self.efficiency_score = 0.0
        
        print(f"🎯 Mediator Agent {agent_id} initialised at {position}")
        print(f"   Tactical coordination with adaptive learning enabled")
    
    def update(self, environment, robots: List, drones: List, communication) -> None:
        """
        Main update cycle - integrates with simulation step timing
        """
        current_time = time.time()
        
        # Update performance tracking
        self._update_robot_performance_data(robots)
        
        # State machine execution
        if self.state == MediatorState.MONITORING:
            self._monitor_situation(environment, robots, drones, communication)
        elif self.state == MediatorState.ALLOCATING:
            self._allocate_resources(robots, communication, environment)
        elif self.state == MediatorState.COORDINATING:
            self._coordinate_active_missions(robots, communication)
        elif self.state == MediatorState.EMERGENCY:
            self._handle_emergency_situations(robots, communication)
        elif self.state == MediatorState.OPTIMISING:
            self._optimise_strategies()
        
        # Periodic learning and adaptation
        if len(self.assignment_history) > 0 and len(self.assignment_history) % 10 == 0:
            self._learn_from_experience()
    
    def _monitor_situation(self, environment, robots: List, drones: List, communication) -> None:
        """
        Monitor overall situation and identify priorities
        """
        # Scan for pending rescue operations
        pending_rescues = self._identify_pending_rescues(environment, communication)
        
        if not pending_rescues:
            # No immediate rescues - check for optimisation opportunities
            if len(self.assignment_history) > 5:
                self.state = MediatorState.OPTIMISING
            return
        
        # Analyse and prioritise rescue locations
        self.mission_priorities.clear()
        self.rescue_queue.clear()
        
        for location in pending_rescues:
            priority_data = self._calculate_comprehensive_priority(location, robots, environment)
            self.mission_priorities[location] = priority_data
        
        # Sort by priority score
        self.rescue_queue = sorted(
            self.mission_priorities.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        # Check for emergency situations
        emergency_count = sum(1 for _, data in self.rescue_queue 
                            if data['priority_level'] == PriorityLevel.CRITICAL)
        
        if emergency_count > 0:
            print(f"🚨 Mediator {self.agent_id}: {emergency_count} CRITICAL situations detected")
            self.state = MediatorState.EMERGENCY
        elif len(self.rescue_queue) > 0:
            self.state = MediatorState.ALLOCATING
        else:
            self.state = MediatorState.COORDINATING
    
    def _identify_pending_rescues(self, environment, communication) -> List[Tuple[int, int]]:
        """
        Identify all locations requiring rescue attention
        """
        pending_locations = []
        
        # From environment (unassigned missing persons)
        for location in environment.missing_persons:
            if not self._is_location_assigned(location, communication):
                pending_locations.append(location)
        
        # From communication system (reported but unassigned)
        if hasattr(communication, 'found_persons'):
            for location, info in communication.found_persons.items():
                if not info.get('assigned', False) and location not in pending_locations:
                    pending_locations.append(location)
        
        return pending_locations
    
    def _is_location_assigned(self, location: Tuple[int, int], communication) -> bool:
        """
        Check if a location is already assigned to a robot
        """
        # Check communication system assignments
        if hasattr(communication, 'assigned_rescues'):
            return location in communication.assigned_rescues.values()
        
        # Check our own tracking
        for assignment_data in self.active_assignments.values():
            if assignment_data.get('target_location') == location:
                return True
        
        return False
    
    def _calculate_comprehensive_priority(self, location: Tuple[int, int], 
                                        robots: List, environment) -> Dict:
        """
        Calculate comprehensive priority using multiple sophisticated factors
        """
        priority_data = {
            'location': location,
            'elevation_factor': 0,
            'distance_factor': 0,
            'waiting_factor': 0,
            'terrain_factor': 0,
            'robot_efficiency_factor': 0,
            'total_score': 0,
            'priority_level': PriorityLevel.MEDIUM,
            'recommended_robot': None,
            'estimated_difficulty': 1.0
        }
        
        # Factor 1: Elevation urgency (higher elevation = more urgent)
        elevation = environment.get_elevation(*location)
        priority_data['elevation_factor'] = (elevation / 3000) * 100
        
        # Factor 2: Distance to available robots (closer = higher priority for efficiency)
        available_robots = [r for r in robots 
                          if r.state.value == 'at_base' 
                          and r.has_first_aid_kit 
                          and r.current_battery > self.battery_critical_threshold]
        
        if available_robots:
            min_distance = min(
                abs(r.position[0] - location[0]) + abs(r.position[1] - location[1])
                for r in available_robots
            )
            # Inverse relationship: closer = higher priority
            priority_data['distance_factor'] = max(0, 50 - min_distance * 2)
            
            # Find recommended robot
            best_robot = min(available_robots, 
                           key=lambda r: abs(r.position[0] - location[0]) + abs(r.position[1] - location[1]))
            priority_data['recommended_robot'] = best_robot.robot_id
        else:
            priority_data['distance_factor'] = 0
        
        # Factor 3: Waiting time (if tracked)
        waiting_time = self._get_location_waiting_time(location)
        priority_data['waiting_factor'] = min(waiting_time * 2, 30)  # Cap at 30 points
        
        # Factor 4: Terrain difficulty (from learned knowledge or environment)
        terrain_difficulty = self._assess_terrain_difficulty(location, environment)
        priority_data['terrain_factor'] = terrain_difficulty * 10
        priority_data['estimated_difficulty'] = terrain_difficulty
        
        # Factor 5: Robot efficiency matching
        if available_robots and priority_data['recommended_robot'] is not None:
            robot_efficiency = self._get_robot_efficiency(priority_data['recommended_robot'])
            priority_data['robot_efficiency_factor'] = robot_efficiency * 15
        
        # Calculate weighted total score
        total_score = (
            priority_data['elevation_factor'] * self.urgency_weights['elevation'] +
            priority_data['distance_factor'] * self.urgency_weights['distance_to_robots'] +
            priority_data['waiting_factor'] * self.urgency_weights['waiting_time'] +
            priority_data['terrain_factor'] * self.urgency_weights['terrain_difficulty'] +
            priority_data['robot_efficiency_factor'] * self.urgency_weights['robot_efficiency']
        )
        
        priority_data['total_score'] = total_score
        
        # Determine priority level
        if total_score >= 80:
            priority_data['priority_level'] = PriorityLevel.CRITICAL
        elif total_score >= 60:
            priority_data['priority_level'] = PriorityLevel.HIGH
        elif total_score >= 40:
            priority_data['priority_level'] = PriorityLevel.MEDIUM
        elif total_score >= 20:
            priority_data['priority_level'] = PriorityLevel.LOW
        else:
            priority_data['priority_level'] = PriorityLevel.MINIMAL
        
        return priority_data
    
    def _get_location_waiting_time(self, location: Tuple[int, int]) -> float:
        """
        Get how long a location has been waiting for rescue
        """
        # Check if we have timing data for this location
        for assignment in self.assignment_history:
            if assignment.get('location') == location and assignment.get('status') == 'pending':
                return time.time() - assignment.get('timestamp', time.time())
        
        return 0.0  # Default if no data
    
    def _assess_terrain_difficulty(self, location: Tuple[int, int], environment) -> float:
        """
        Assess terrain difficulty based on learned knowledge and environment data
        """
        # Use learned terrain knowledge if available
        if location in self.terrain_knowledge:
            return self.terrain_knowledge[location]
        
        # Estimate based on elevation and environment features
        elevation = environment.get_elevation(*location)
        base_difficulty = elevation / 3000  # Normalize to 0-1
        
        # Add some random variation for realism
        difficulty = base_difficulty + np.random.normal(0, 0.1)
        difficulty = max(0.1, min(2.0, difficulty))  # Clamp between 0.1 and 2.0
        
        # Store for future reference
        self.terrain_knowledge[location] = difficulty
        
        return difficulty
    
    def _get_robot_efficiency(self, robot_id: int) -> float:
        """
        Get efficiency score for a specific robot based on performance history
        """
        if robot_id in self.robot_performance_data:
            perf_data = self.robot_performance_data[robot_id]
            # Calculate efficiency based on success rate and speed
            success_rate = perf_data.get('success_rate', 0.5)
            avg_rescue_time = perf_data.get('avg_rescue_time', 50)
            battery_efficiency = perf_data.get('battery_efficiency', 0.5)
            
            # Combine metrics (lower rescue time is better)
            time_score = max(0, 1 - (avg_rescue_time / 100))
            efficiency = (success_rate + time_score + battery_efficiency) / 3
            
            return min(1.0, max(0.1, efficiency))
        
        return 0.5  # Default moderate efficiency
    
    def _allocate_resources(self, robots: List, communication, environment) -> None:
        """
        Intelligent resource allocation based on priorities and capabilities
        """
        allocations_made = 0
        
        # Get available robots
        available_robots = [
            r for r in robots
            if r.state.value == 'at_base'
            and r.has_first_aid_kit
            and r.current_battery > self.battery_critical_threshold
            and r.robot_id not in self.active_assignments
        ]
        
        if not available_robots:
            print(f"⚠️ Mediator {self.agent_id}: No available robots for allocation")
            self.state = MediatorState.COORDINATING
            return
        
        # Process rescue queue up to maximum assignments per cycle
        for location, priority_data in self.rescue_queue[:self.max_assignments_per_cycle]:
            if allocations_made >= len(available_robots):
                break
            
            # Skip if already assigned
            if self._is_location_assigned(location, communication):
                continue
            
            # Find best robot for this mission
            best_robot = self._select_optimal_robot(location, priority_data, available_robots, environment)
            
            if best_robot:
                # Make assignment
                success = self._assign_robot_to_mission(best_robot, location, priority_data, communication)
                
                if success:
                    available_robots.remove(best_robot)
                    allocations_made += 1
                    
                    print(f"🎯 Mediator {self.agent_id}: Assigned Robot {best_robot.robot_id} "
                          f"to {location} (Priority: {priority_data['priority_level'].name})")
        
        self.assignments_made += allocations_made
        
        if allocations_made > 0:
            self.state = MediatorState.COORDINATING
        else:
            self.state = MediatorState.MONITORING
            
            
            
            
    
    def _select_optimal_robot(self, location: Tuple[int, int], priority_data: Dict, 
                            available_robots: List, environment) -> Optional:
        
        
        
        
        
        """
        Select the optimal robot for a specific mission using advanced matching
        """
        if not available_robots:
            return None
        
        best_robot = None
        best_score = float('-inf')
        
        for robot in available_robots:
            score = self._calculate_robot_mission_fit(robot, location, priority_data, environment)
            
            if score > best_score:
                best_score = score
                best_robot = robot
        
        return best_robot
    
    def _calculate_robot_mission_fit(self, robot, location: Tuple[int, int], 
                                   priority_data: Dict, environment) -> float:
        """
        Calculate how well a robot fits a specific mission
        """
        score = 0.0
        
        # Distance factor (closer is better)
        distance = abs(robot.position[0] - location[0]) + abs(robot.position[1] - location[1])
        score += max(0, 50 - distance)
        
        # Battery adequacy (more battery is better, but with diminishing returns)
        battery_needed = distance * 2 + 20 + priority_data['estimated_difficulty'] * 10
        battery_excess = robot.current_battery - battery_needed
        if battery_excess > 0:
            score += min(battery_excess, 30)  # Cap battery bonus
        else:
            score -= abs(battery_excess) * 2  # Penalty for insufficient battery
        
        # Robot efficiency factor
        efficiency = self._get_robot_efficiency(robot.robot_id)
        score += efficiency * 25
        
        # Mission difficulty matching (better robots for harder missions)
        difficulty = priority_data['estimated_difficulty']
        if difficulty > 1.2 and efficiency > 0.7:
            score += 15  # Bonus for assigning good robots to hard missions
        elif difficulty < 0.8 and efficiency < 0.5:
            score += 10  # Bonus for assigning weaker robots to easy missions
        
        # Q-learning integration bonus
        if hasattr(robot, 'q_agent'):
            # Bonus if robot has experience with this location or similar areas
            if location in robot.q_agent.rescue_success_map:
                score += robot.q_agent.rescue_success_map[location] * 5
            
            # Small bonus for robots with more exploration experience
            score += min(len(robot.q_agent.q_table) / 100, 10)
        
        return score
    
    def _assign_robot_to_mission(self, robot, location: Tuple[int, int], 
                               priority_data: Dict, communication) -> bool:
        """
        Execute the robot assignment with full tracking
        """
        robot_id = f"robot_{robot.robot_id}"
        
        # Send assignment through communication system
        try:
            if hasattr(communication, 'send_individual_message'):
                communication.send_individual_message(
                    sender_id=f"mediator_{self.agent_id}",
                    recipient_id=robot_id,
                    msg_type=communication.MessageType.RESCUE_REQUEST,
                    location=location,
                    urgency=priority_data['priority_level'].value,
                    data={
                        'priority_score': priority_data['total_score'],
                        'estimated_difficulty': priority_data['estimated_difficulty'],
                        'mediator_assignment': True
                    }
                )
            
            # Track assignment internally
            assignment_record = {
                'robot_id': robot.robot_id,
                'location': location,
                'priority_data': priority_data,
                'timestamp': time.time(),
                'mediator_id': self.agent_id,
                'status': 'assigned'
            }
            
            self.active_assignments[robot.robot_id] = assignment_record
            self.assignment_history.append(assignment_record)
            
            # Try to assign through communication system
            if hasattr(communication, 'assign_robot_to_rescue'):
                return communication.assign_robot_to_rescue(robot_id, location)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Mediator {self.agent_id}: Assignment failed - {e}")
            return False
    
    def _coordinate_active_missions(self, robots: List, communication) -> None:
        """
        Monitor and coordinate active rescue missions
        """
        completed_assignments = []
        
        for robot_id, assignment_data in self.active_assignments.items():
            robot = next((r for r in robots if r.robot_id == robot_id), None)
            
            if not robot:
                continue
            
            location = assignment_data['location']
            
            # Check mission status
            if robot.state.value == 'returning' or robot.state.value == 'at_base':
                # Mission likely completed
                if location not in [pos for pos in [r.position for r in robots]]:
                    # Successful rescue
                    self._record_successful_rescue(assignment_data, robot)
                    completed_assignments.append(robot_id)
                    self.successful_rescues += 1
                
            # Check for robots needing assistance
            elif robot.current_battery < self.battery_recall_threshold:
                print(f"🔋 Mediator {self.agent_id}: Robot {robot_id} low battery - considering recall")
                # Could implement recall logic here
            
            # Check for stuck robots
            elif hasattr(robot, 'stuck_counter') and robot.stuck_counter > 5:
                print(f"🚫 Mediator {self.agent_id}: Robot {robot_id} appears stuck")
                # Could implement assistance logic here
        
        # Clean up completed assignments
        for robot_id in completed_assignments:
            del self.active_assignments[robot_id]
        
        # Return to monitoring if no active assignments
        if not self.active_assignments:
            self.state = MediatorState.MONITORING
    
    def _record_successful_rescue(self, assignment_data: Dict, robot) -> None:
        """
        Record successful rescue for learning and performance tracking
        """
        location = assignment_data['location']
        rescue_time = time.time() - assignment_data['timestamp']
        
        # Update robot performance data
        robot_id = robot.robot_id
        if robot_id not in self.robot_performance_data:
            self.robot_performance_data[robot_id] = {
                'missions_completed': 0,
                'total_rescue_time': 0,
                'success_rate': 1.0,
                'battery_efficiency': 1.0
            }
        
        perf_data = self.robot_performance_data[robot_id]
        perf_data['missions_completed'] += 1
        perf_data['total_rescue_time'] += rescue_time
        perf_data['avg_rescue_time'] = perf_data['total_rescue_time'] / perf_data['missions_completed']
        
        # Update success patterns
        terrain_type = f"elevation_{assignment_data['priority_data']['elevation_factor']//20}"
        if terrain_type not in self.success_patterns:
            self.success_patterns[terrain_type] = []
        
        self.success_patterns[terrain_type].append({
            'robot_id': robot_id,
            'rescue_time': rescue_time,
            'priority_score': assignment_data['priority_data']['total_score']
        })
        
        print(f"✅ Mediator {self.agent_id}: Recorded successful rescue at {location} "
              f"by Robot {robot_id} in {rescue_time:.1f}s")
    
    def _handle_emergency_situations(self, robots: List, communication) -> None:
        """
        Handle critical/emergency rescue situations with highest priority
        """
        critical_rescues = [
            (loc, data) for loc, data in self.rescue_queue 
            if data['priority_level'] == PriorityLevel.CRITICAL
        ]
        
        if not critical_rescues:
            self.state = MediatorState.ALLOCATING
            return
        
        print(f"🚨 Mediator {self.agent_id}: Emergency mode - processing {len(critical_rescues)} critical rescues")
        
        # Emergency allocation: prioritise critical situations
        for location, priority_data in critical_rescues:
            # Find any available robot, even if not ideal
            available_robot = None
            for robot in robots:
                if (robot.state.value in ['at_base', 'searching'] 
                    and robot.current_battery > 20
                    and robot.robot_id not in self.active_assignments):
                    available_robot = robot
                    break
            
            if available_robot:
                self._assign_robot_to_mission(available_robot, location, priority_data, communication)
                print(f"🚨 Emergency assignment: Robot {available_robot.robot_id} → {location}")
        
        self.state = MediatorState.COORDINATING
    
    def _optimise_strategies(self) -> None:
        """
        Learn from experience and optimise future strategies
        """
        if len(self.assignment_history) < 5:
            self.state = MediatorState.MONITORING
            return
        
        # Analyse assignment patterns
        recent_assignments = self.assignment_history[-10:]
        
        # Calculate current efficiency metrics
        total_assignments = len(recent_assignments)
        successful_assignments = len([a for a in recent_assignments if a.get('status') == 'completed'])
        
        if total_assignments > 0:
            self.efficiency_score = successful_assignments / total_assignments
        
        # Adjust urgency weights based on performance
        if self.efficiency_score < 0.7:
            # Poor performance - adjust weights
            self.urgency_weights['distance_to_robots'] *= 1.1  # Prioritise closer rescues
            self.urgency_weights['robot_efficiency'] *= 1.1   # Weight robot capability more
            print(f"📊 Mediator {self.agent_id}: Strategy optimisation - efficiency: {self.efficiency_score:.2f}")
        
        self.state = MediatorState.MONITORING
    
    def _update_robot_performance_data(self, robots: List) -> None:
        """
        Update performance tracking for all robots
        """
        for robot in robots:
            robot_id = robot.robot_id
            
            if robot_id not in self.robot_performance_data:
                self.robot_performance_data[robot_id] = {
                    'missions_completed': 0,
                    'total_rescue_time': 0,
                    'success_rate': 0.5,
                    'battery_efficiency': 0.5,
                    'avg_rescue_time': 50
                }
            
            # Update based on current robot status
            perf_data = self.robot_performance_data[robot_id]
            
            # Simple battery efficiency calculation
            battery_used = robot.max_battery - robot.current_battery
            if battery_used > 0:
                efficiency = robot.persons_rescued / max(1, battery_used / 10)
                perf_data['battery_efficiency'] = (
                    perf_data['battery_efficiency'] * 0.9 + efficiency * 0.1
                )
    
    def _learn_from_experience(self) -> None:
        """
        Periodic learning from assignment history
        """
        if len(self.assignment_history) < 10:
            return
        
        # Analyse what types of assignments work best
        recent_history = self.assignment_history[-20:]
        
        # Learn terrain difficulty patterns
        for assignment in recent_history:
            location = assignment.get('location')
            if location and assignment.get('status') == 'completed':
                # This was a successful assignment - terrain is manageable
                if location in self.terrain_knowledge:
                    # Reduce estimated difficulty slightly
                    self.terrain_knowledge[location] *= 0.95
        
        print(f"🧠 Mediator {self.agent_id}: Learning update completed")
    
    def get_status(self) -> Dict:
        """
        Get current mediator status and performance metrics
        """
        return {
            'id': self.agent_id,
            'state': self.state.value,
            'active_assignments': len(self.active_assignments),
            'assignments_made': self.assignments_made,
            'successful_rescues': self.successful_rescues,
            'efficiency_score': self.efficiency_score,
            'queue_size': len(self.rescue_queue),
            'terrain_knowledge_points': len(self.terrain_knowledge),
            'robot_performance_tracked': len(self.robot_performance_data)
        }
    
    def get_detailed_report(self) -> Dict:
        """
        Get detailed performance report for analysis
        """
        return {
            'mediator_id': self.agent_id,
            'total_assignments': self.assignments_made,
            'successful_rescues': self.successful_rescues,
            'failed_assignments': self.failed_assignments,
            'efficiency_score': self.efficiency_score,
            'assignment_history': self.assignment_history,
            'success_patterns': self.success_patterns,
            'terrain_knowledge': self.terrain_knowledge,
            'robot_performance_data': self.robot_performance_data,
            'urgency_weights': self.urgency_weights
        }