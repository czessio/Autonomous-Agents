# Updated simulation for Activity 4 - includes Novel Mode with Q-Learning

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
from typing import List, Dict
from enum import Enum

# Import from previous activities
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from activity1.environment import Environment
from activity1.terrain_robot import TerrainRobot, RobotState
from activity1.explorer_drone import ExplorerDrone, DroneState
from activity3.communication import BasicCommunication, ExtendedCommunication, MessageType

# Import Novel Mode components
from activity4.learning_robot import LearningTerrainRobot
from activity4.q_learning import CollectiveQLearning


class OperationMode(Enum):
    BASIC = "basic"
    EXTENDED = "extended"
    NOVEL = "novel"  # Q-Learning with collective intelligence


class MountainRescueSimulation:
    def __init__(self, num_terrain_robots: int = 3, num_drones: int = 2, 
                 num_missing_persons: int = 6, max_simulation_steps: int = 400,
                 operation_mode: OperationMode = OperationMode.BASIC,
                 spawn_new_persons: bool = False, spawn_interval: int = 50,
                 enable_learning: bool = True):
        
        # Simulation parameters
        self.max_steps = max_simulation_steps
        self.current_step = 0
        self.running = True
        self.operation_mode = operation_mode
        
        # Dynamic person spawning
        self.spawn_new_persons = spawn_new_persons and (operation_mode in [OperationMode.EXTENDED, OperationMode.NOVEL])
        self.spawn_interval = spawn_interval
        self.last_spawn_step = 0
        self.total_persons_spawned = num_missing_persons
        
        # Create environment
        self.environment = Environment(
            width=20, height=15,
            num_missing_persons=num_missing_persons,
            num_terrain_robots=num_terrain_robots,
            num_drones=num_drones
        )
        
        # Create communication system based on mode
        if self.operation_mode == OperationMode.NOVEL:
            self.communication = ExtendedCommunication()
            print(" Starting in NOVEL MODE with Q-Learning and collective intelligence")
        elif self.operation_mode == OperationMode.EXTENDED:
            self.communication = ExtendedCommunication()
            print(" Starting in EXTENDED MODE with advanced communication")
        else:
            self.communication = BasicCommunication()
            print(" Starting in BASIC MODE")
        
        # Create terrain robots
        self.terrain_robots = []
        
        if self.operation_mode == OperationMode.NOVEL:
            # Create collective Q-learning system
            self.collective_learning = CollectiveQLearning(
                num_robots=num_terrain_robots,
                environment_size=(self.environment.width, self.environment.height)
            )
            
            # Load previous knowledge if available
            knowledge_file = "data/collective_knowledge.pkl"
            if self.collective_learning.load_collective_knowledge(knowledge_file):
                print(" Loaded previous collective knowledge")
            
            # Create learning robots
            for i, base_pos in enumerate(self.environment.terrain_robot_base):
                q_agent = self.collective_learning.get_robot_agent(i)
                robot = LearningTerrainRobot(
                    robot_id=i,
                    start_position=base_pos,
                    q_learning_agent=q_agent,
                    max_battery=120,
                    battery_drain_rate=1
                )
                robot.enable_learning(enable_learning)
                self.terrain_robots.append(robot)
                
                # Register with communication
                if isinstance(self.communication, ExtendedCommunication):
                    self.communication.register_agent(f"robot_{i}", "robot")
        else:
            # Create standard robots for Basic/Extended modes
            for i, base_pos in enumerate(self.environment.terrain_robot_base):
                robot = TerrainRobot(
                    robot_id=i,
                    start_position=base_pos,
                    max_battery=120,
                    battery_drain_rate=1
                )
                self.terrain_robots.append(robot)
                
                if isinstance(self.communication, ExtendedCommunication):
                    self.communication.register_agent(f"robot_{i}", "robot")
        
        # Create explorer drones
        self.drones = []
        for i, base_pos in enumerate(self.environment.drone_base):
            drone = ExplorerDrone(
                drone_id=i,
                start_position=base_pos,
                max_battery=120,
                battery_drain_rate=2
            )
            self.drones.append(drone)
            
            if isinstance(self.communication, ExtendedCommunication):
                self.communication.register_agent(f"drone_{i}", "drone")
        
        # Mode-specific initialization
        if self.operation_mode == OperationMode.BASIC:
            for robot in self.terrain_robots:
                robot.state = RobotState.SEARCHING
            for drone in self.drones:
                drone.state = DroneState.EXPLORING
                drone._generate_search_pattern(self.environment)
        else:
            # Extended and Novel modes: robots wait, drones explore
            for robot in self.terrain_robots:
                robot.state = RobotState.AT_BASE
                robot.min_leave_threshold = 50
            for drone in self.drones:
                drone.state = DroneState.EXPLORING
                drone._generate_search_pattern(self.environment)
        
        # Statistics tracking
        self.stats = {
            'total_persons_rescued': 0,
            'total_persons_spawned': num_missing_persons,
            'total_battery_consumed': 0,
            'rescue_times': [],
            'person_spawn_times': {},
            'communication_stats': {},
            'learning_stats': {},
            'mode': operation_mode.value,
            'mission_start_time': None,
            'mission_end_time': None
        }
        
        # Initialize spawn times
        for person_loc in self.environment.missing_persons:
            self.stats['person_spawn_times'][person_loc] = 0
        
        # Knowledge sharing tracking for Novel Mode
        self.knowledge_sharing_interval = 20
        self.last_knowledge_share = 0
        
        # Animation setup
        self.fig = None
        self.ax = None
        self.animation = None
    
    def _update_simulation_step(self) -> None:
        """Update all agents and environment for one time step"""
        
        # Dynamic person spawning
        if self.spawn_new_persons and self.current_step > 0:
            if (self.current_step - self.last_spawn_step) >= self.spawn_interval:
                self._spawn_new_person()
        
        # Novel Mode: Periodic knowledge sharing
        if self.operation_mode == OperationMode.NOVEL:
            if (self.current_step - self.last_knowledge_share) >= self.knowledge_sharing_interval:
                self.collective_learning.share_knowledge()
                self.last_knowledge_share = self.current_step
        
        # Update robot availability
        if isinstance(self.communication, ExtendedCommunication):
            for i, robot in enumerate(self.terrain_robots):
                available = robot.state == RobotState.AT_BASE and robot.has_first_aid_kit
                self.communication.update_robot_status(
                    f"robot_{i}", 
                    robot.current_battery,
                    available
                )
        
        # Update drones
        for drone in self.drones:
            if isinstance(self.communication, ExtendedCommunication):
                messages = self.communication.get_messages(f"drone_{drone.drone_id}")
            drone.update(self.environment, self.current_step, self.communication)
        
        # Update robots
        for robot in self.terrain_robots:
            if self.operation_mode in [OperationMode.EXTENDED, OperationMode.NOVEL]:
                self._handle_advanced_mode_robot(robot)
            else:
                robot.update(self.environment, self.current_step, self.communication)
        
        # Update statistics
        self._update_rescue_statistics()
        self._check_mission_status()
        self._update_statistics()
    
    def _handle_advanced_mode_robot(self, robot) -> None:
        """Handle robot behavior in Extended/Novel modes"""
        robot_id = f"robot_{robot.robot_id}"
        
        # Get messages
        messages = self.communication.get_messages(robot_id)
        
        # Process messages
        for message in messages:
            if message.msg_type in [MessageType.PERSON_FOUND, MessageType.RESCUE_REQUEST]:
                if robot.state == RobotState.AT_BASE and robot.has_first_aid_kit and robot.current_battery > 50:
                    if self.communication.assign_robot_to_rescue(robot_id, message.location):
                        robot.assigned_location = message.location
                        robot.state = RobotState.SEARCHING
                        
                        # Novel Mode: Log this for learning
                        if isinstance(robot, LearningTerrainRobot):
                            robot.q_agent.rescue_success_map[message.location] += 0.1  # Anticipatory reward
                        
                        print(f" Robot {robot.robot_id} responding to rescue at {message.location}")
        
        # Update robot
        robot.update(self.environment, self.current_step, self.communication)
        
        # Check rescue completion
        if hasattr(robot, '_last_rescued_location'):
            self.communication.complete_rescue(robot_id, robot._last_rescued_location)
            delattr(robot, '_last_rescued_location')
    
    def _update_statistics(self) -> None:
        """Update simulation statistics including learning metrics"""
        # Basic statistics
        self.stats['total_persons_rescued'] = len(self.environment.rescued_persons)
        self.stats['total_persons_spawned'] = self.total_persons_spawned
        
        # Battery consumption
        total_battery = 0
        for robot in self.terrain_robots:
            total_battery += (robot.max_battery - robot.current_battery)
        for drone in self.drones:
            total_battery += (drone.max_battery - drone.current_battery)
        self.stats['total_battery_consumed'] = total_battery
        
        # Communication statistics
        if isinstance(self.communication, ExtendedCommunication):
            self.stats['communication_stats'] = self.communication.get_communication_stats()
        
        # Learning statistics for Novel Mode
        if self.operation_mode == OperationMode.NOVEL:
            self.stats['learning_stats'] = self.collective_learning.get_learning_statistics()
            
            # Add individual robot learning stats
            robot_learning_stats = []
            for robot in self.terrain_robots:
                if isinstance(robot, LearningTerrainRobot):
                    robot_learning_stats.append(robot.get_learning_status())
            self.stats['robot_learning_stats'] = robot_learning_stats
    
    def _add_status_text(self) -> None:
        """Add enhanced status information to visualization"""
        status_text = f"Mode: {self.operation_mode.value.upper()} | Step: {self.current_step}/{self.max_steps}\n"
        status_text += f"Missing: {len(self.environment.missing_persons)} | "
        status_text += f"Rescued: {len(self.environment.rescued_persons)}"
        
        if self.spawn_new_persons:
            status_text += f" | Spawned: {self.total_persons_spawned}\n"
        else:
            status_text += "\n"
        
        # Communication stats
        if isinstance(self.communication, ExtendedCommunication):
            comm_stats = self.communication.get_communication_stats()
            status_text += f"Messages: {comm_stats['total_messages']} | "
            status_text += f"Active: {comm_stats['active_rescues']}\n"
        
        # Learning stats for Novel Mode
        if self.operation_mode == OperationMode.NOVEL and self.stats.get('learning_stats'):
            learning_stats = self.stats['learning_stats']
            status_text += f"States Explored: {learning_stats['total_states_explored']} | "
            status_text += f"Terrain Mapped: {learning_stats['total_terrain_mapped']}\n"
            status_text += f"Exploration Rate: {learning_stats['average_exploration_rate']:.3f}\n"
        
        status_text += "\nTerrain Robots:\n"
        for robot in self.terrain_robots:
            if isinstance(robot, LearningTerrainRobot):
                status = robot.get_learning_status()
                status_text += f"  R{status['id']}: {status['state']} (B:{status['battery']}%)"
                status_text += f" [Ep:{status['episodes_completed']} Q:{status['q_table_size']}]"
            else:
                status = robot.get_status()
                status_text += f"  R{status['id']}: {status['state']} (B:{status['battery']}%)"
            
            if hasattr(robot, 'assigned_location') and robot.assigned_location:
                status_text += f" → {robot.assigned_location}"
            status_text += "\n"
        
        status_text += "\nDrones:\n"
        for drone in self.drones:
            status = drone.get_status()
            status_text += f"  D{status['id']}: {status['state']} (B:{status['battery']}%)\n"
        
        # Legend
        status_text += "\nLegend: P=Person =Robot =Drone | "
        status_text += "Q=Q-table size Ep=Episodes"
        
        # Add text box
        self.ax.text(0.02, 0.98, status_text, transform=self.ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.9), fontfamily='monospace', fontsize=8)
    
    def _generate_final_report(self) -> dict:
        """Generate comprehensive final report including learning metrics"""
        mission_time = self.stats['mission_end_time'] - self.stats['mission_start_time']
        
        # Calculate KPIs
        avg_rescue_time = 0
        if self.stats['rescue_times']:
            avg_rescue_time = sum(r['rescue_time'] for r in self.stats['rescue_times']) / len(self.stats['rescue_times'])
        
        battery_efficiency = 0
        if self.stats['total_battery_consumed'] > 0:
            battery_efficiency = self.stats['total_persons_rescued'] / self.stats['total_battery_consumed']
        
        report = {
            'mode': self.operation_mode.value,
            'simulation_steps': self.current_step,
            'mission_time_seconds': round(mission_time, 2),
            'persons_rescued': self.stats['total_persons_rescued'],
            'persons_spawned': self.stats['total_persons_spawned'],
            'persons_remaining': len(self.environment.missing_persons),
            'total_battery_consumed': self.stats['total_battery_consumed'],
            'success_rate': self.stats['total_persons_rescued'] / max(1, self.stats['total_persons_spawned']),
            'avg_rescue_time': round(avg_rescue_time, 2),
            'battery_efficiency': round(battery_efficiency, 4),
            'rescue_times': self.stats['rescue_times'],
            'communication_stats': self.stats.get('communication_stats', {}),
            'learning_stats': self.stats.get('learning_stats', {}),
            'robot_performance': [],
            'drone_performance': []
        }
        
        # Robot performance
        for robot in self.terrain_robots:
            if isinstance(robot, LearningTerrainRobot):
                status = robot.get_learning_status()
                report['robot_performance'].append({
                    'id': status['id'],
                    'persons_rescued': status['rescued'],
                    'final_battery': status['battery'],
                    'battery_used': robot.max_battery - status['battery'],
                    'final_state': status['state'],
                    'episodes_completed': status['episodes_completed'],
                    'q_table_size': status['q_table_size'],
                    'exploration_rate': status['exploration_rate'],
                    'total_distance': status['total_distance']
                })
            else:
                status = robot.get_status()
                report['robot_performance'].append({
                    'id': status['id'],
                    'persons_rescued': status['rescued'],
                    'final_battery': status['battery'],
                    'battery_used': robot.max_battery - status['battery'],
                    'final_state': status['state']
                })
        
        # Drone performance
        for drone in self.drones:
            status = drone.get_status()
            report['drone_performance'].append({
                'id': status['id'],
                'persons_found': status['found_persons'],
                'areas_explored': status['exploring'],
                'final_battery': status['battery'],
                'battery_used': drone.max_battery - status['battery'],
                'final_state': status['state']
            })
        
        # Save collective knowledge for Novel Mode
        if self.operation_mode == OperationMode.NOVEL:
            knowledge_file = "data/collective_knowledge.pkl"
            self.collective_learning.save_collective_knowledge(knowledge_file)
            print(f" Saved collective knowledge to {knowledge_file}")
        
        self._print_final_report(report)
        return report
    
    def _print_final_report(self, report: dict) -> None:
        """Print formatted final report"""
        print("\n" + "="*70)
        print(f"FINAL REPORT - {report['mode'].upper()} MODE")
        print("="*70)
        print(f"Mission Duration: {report['simulation_steps']} steps ({report['mission_time_seconds']}s)")
        print(f"Success Rate: {report['success_rate']:.1%}")
        print(f"Persons Rescued: {report['persons_rescued']} / {report['persons_spawned']} spawned")
        
        print(f"\n KEY PERFORMANCE INDICATORS:")
        print(f"  Average Rescue Time: {report['avg_rescue_time']} steps")
        print(f"  Battery Efficiency: {report['battery_efficiency']:.3f} rescues/battery unit")
        print(f"  Total Battery Consumed: {report['total_battery_consumed']} units")
        
        if report['communication_stats']:
            print(f"\n COMMUNICATION STATISTICS:")
            comm = report['communication_stats']
            print(f"  Total Messages: {comm.get('total_messages', 0)}")
            print(f"  Message Efficiency: {comm.get('message_efficiency', 0):.2%}")
        
        if report['learning_stats']:
            print(f"\n LEARNING STATISTICS:")
            learning = report['learning_stats']
            print(f"  Total States Explored: {learning.get('total_states_explored', 0)}")
            print(f"  Terrain Points Mapped: {learning.get('total_terrain_mapped', 0)}")
            print(f"  Known Rescue Locations: {learning.get('total_rescue_locations', 0)}")
            print(f"  Average Exploration Rate: {learning.get('average_exploration_rate', 0):.3f}")
        
        print("\n TERRAIN ROBOT PERFORMANCE:")
        for robot_perf in report['robot_performance']:
            print(f"  Robot {robot_perf['id']}: {robot_perf['persons_rescued']} rescued, "
                  f"{robot_perf['battery_used']} battery used")
            if 'episodes_completed' in robot_perf:
                print(f"    Learning: {robot_perf['episodes_completed']} episodes, "
                      f"{robot_perf['q_table_size']} states learned, "
                      f"ε={robot_perf['exploration_rate']:.3f}")
        
        print("\n DRONE PERFORMANCE:")
        for drone_perf in report['drone_performance']:
            print(f"  Drone {drone_perf['id']}: {drone_perf['persons_found']} found, "
                  f"{drone_perf['areas_explored']} areas explored")
        
        print("="*70)
    
    # Inherit other methods from Activity 3 simulation
    def run_simulation(self, visualise: bool = True, step_delay: float = 0.5) -> dict:
        """Run the complete simulation"""
        print("\n" + "="*70)
        print("MOUNTAIN RESCUE SIMULATION - ACTIVITY 4")
        print("="*70)
        print(f"Mode: {self.operation_mode.value.upper()}")
        print(f"Environment: {self.environment.width}x{self.environment.height}")
        print(f"Terrain robots: {len(self.terrain_robots)}")
        print(f"Explorer drones: {len(self.drones)}")
        
        if self.operation_mode == OperationMode.NOVEL:
            print(f"Q-Learning: Enabled with collective intelligence")
            print(f"Knowledge sharing: Every {self.knowledge_sharing_interval} steps")
        
        print("-" * 70)
        
        self.stats['mission_start_time'] = time.time()
        
        try:
            if visualise:
                self._setup_visualisation()
                self._run_animated_simulation()
            else:
                self._run_batch_simulation(step_delay)
        except Exception as e:
            print(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()
            self._run_batch_simulation(step_delay)
        
        self.stats['mission_end_time'] = time.time()
        return self._generate_final_report()
    
    def _setup_visualisation(self) -> None:
        """Setup matplotlib visualisation"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        mode_str = self.operation_mode.value.upper()
        self.fig.suptitle(f'Mountain Rescue Simulation - Activity 4 ({mode_str} Mode)', 
                         fontsize=16, fontweight='bold')
        self._update_visualisation()
    
    def _update_visualisation(self) -> None:
        """Update the visualisation with current state"""
        self.ax.clear()
        self.environment.visualise(self.terrain_robots, self.drones, ax=self.ax)
        self._add_status_text()
        
        mode_str = self.operation_mode.value.capitalize()
        self.ax.set_title(f'Mountain Rescue - {mode_str} Mode - Step {self.current_step}', 
                         fontsize=14, fontweight='bold')
    
    def _run_animated_simulation(self) -> None:
        """Run simulation with matplotlib animation"""
        def animate(frame):
            if self.running and self.current_step < self.max_steps:
                self._update_simulation_step()
                self._update_visualisation()
                self.current_step += 1
                
                if self.current_step % 25 == 0:
                    self._print_progress()
            else:
                if hasattr(self, 'animation') and self.animation:
                    self.animation.event_source.stop()
            return []
        
        self.animation = animation.FuncAnimation(
            self.fig, animate, interval=350, blit=False, cache_frame_data=False
        )
        
        plt.show(block=True)
    
    def _run_batch_simulation(self, step_delay: float) -> None:
        """Run simulation without animation"""
        print(f"Running batch simulation ({self.operation_mode.value} mode)...")
        
        while self.running and self.current_step < self.max_steps:
            self._update_simulation_step()
            
            if step_delay > 0:
                time.sleep(step_delay)
            
            if self.current_step % 50 == 0:
                self._print_progress()
            
            self.current_step += 1
        
        print(f"Simulation completed after {self.current_step} steps")
    
    def _print_progress(self) -> None:
        """Print simulation progress"""
        rescued = len(self.environment.rescued_persons)
        missing = len(self.environment.missing_persons)
        
        print(f"\nStep {self.current_step}: {rescued} rescued, {missing} missing")
        
        if self.operation_mode == OperationMode.NOVEL:
            if self.stats.get('learning_stats'):
                learning = self.stats['learning_stats']
                print(f"  Learning: {learning['total_states_explored']} states, "
                      f"ε={learning['average_exploration_rate']:.3f}")
        
        active_robots = sum(1 for r in self.terrain_robots if r.current_battery > 20)
        active_drones = sum(1 for d in self.drones if d.current_battery > 20)
        print(f"  Active agents: {active_robots} robots, {active_drones} drones")
    
    def _spawn_new_person(self) -> None:
        """Spawn a new missing person in the mountain area"""
        import random
        x_start, y_start, x_end, y_end = self.environment.mountain_area['bounds']
        
        max_attempts = 20
        for _ in range(max_attempts):
            x = random.randint(x_start, x_end - 1)
            y = random.randint(y_start, y_end - 1)
            location = (x, y)
            
            if location not in self.environment.missing_persons and \
               location not in self.environment.rescued_persons:
                self.environment.missing_persons.append(location)
                self.stats['person_spawn_times'][location] = self.current_step
                self.total_persons_spawned += 1
                self.last_spawn_step = self.current_step
                
                print(f" New person appeared at {location} (step {self.current_step})")
                return
    
    def _update_rescue_statistics(self) -> None:
        """Track rescue times for KPI analysis"""
        for person_loc in self.environment.rescued_persons:
            if person_loc in self.stats['person_spawn_times']:
                spawn_time = self.stats['person_spawn_times'][person_loc]
                rescue_time = self.current_step - spawn_time
                
                if person_loc not in [r['location'] for r in self.stats['rescue_times']]:
                    self.stats['rescue_times'].append({
                        'location': person_loc,
                        'spawn_step': spawn_time,
                        'rescue_step': self.current_step,
                        'rescue_time': rescue_time
                    })
    
    def _check_mission_status(self) -> None:
        """Check if mission is complete or should be terminated"""
        # Novel/Extended modes continue while agents are operational
        if self.operation_mode in [OperationMode.EXTENDED, OperationMode.NOVEL]:
            all_inactive = True
            
            for robot in self.terrain_robots:
                if robot.current_battery > 20 or robot.state != RobotState.AT_BASE:
                    all_inactive = False
                    break
            
            if all_inactive:
                for drone in self.drones:
                    if drone.current_battery > 20 or drone.state != DroneState.AT_BASE:
                        all_inactive = False
                        break
            
            if all_inactive and len(self.environment.missing_persons) > 0:
                print(f"\n Mission ended: All agents inactive with {len(self.environment.missing_persons)} persons remaining")
                self.running = False
        else:
            # Basic Mode
            if len(self.environment.missing_persons) == 0:
                print(f"\n Mission Complete! All persons rescued in {self.current_step} steps.")
                self.running = False
            else:
                all_inactive = True
                for robot in self.terrain_robots:
                    if robot.current_battery > robot.low_battery_threshold:
                        all_inactive = False
                        break
                
                if all_inactive:
                    print(f"\n Mission terminated: All agents out of battery.")
                    self.running = False


def main():
    """Main function to run the simulation with mode selection"""
    print("\n MOUNTAIN RESCUE SIMULATION - ACTIVITY 4")
    print("="*50)
    print("Select operation mode:")
    print("1. Basic Mode")
    print("2. Extended Mode")
    print("3. Novel Mode (Q-Learning)")
    print("4. Compare All Modes")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            simulation = MountainRescueSimulation(
                num_terrain_robots=3,
                num_drones=2,
                num_missing_persons=6,
                max_simulation_steps=250,
                operation_mode=OperationMode.BASIC
            )
            viz_choice = input("Enable visualization? (y/n): ").strip().lower()
            report = simulation.run_simulation(visualise=(viz_choice == 'y'))
            
        elif choice == "2":
            simulation = MountainRescueSimulation(
                num_terrain_robots=3,
                num_drones=2,
                num_missing_persons=4,
                max_simulation_steps=300,
                operation_mode=OperationMode.EXTENDED,
                spawn_new_persons=True,
                spawn_interval=40
            )
            viz_choice = input("Enable visualization? (y/n): ").strip().lower()
            report = simulation.run_simulation(visualise=(viz_choice == 'y'))
            
        elif choice == "3":
            print("\n NOVEL MODE - Q-Learning Configuration")
            learn_choice = input("Enable learning? (y/n): ").strip().lower()
            spawn_choice = input("Enable dynamic person spawning? (y/n): ").strip().lower()
            
            simulation = MountainRescueSimulation(
                num_terrain_robots=3,
                num_drones=2,
                num_missing_persons=5,
                max_simulation_steps=400,
                operation_mode=OperationMode.NOVEL,
                spawn_new_persons=(spawn_choice == 'y'),
                spawn_interval=50,
                enable_learning=(learn_choice == 'y')
            )
            viz_choice = input("Enable visualization? (y/n): ").strip().lower()
            report = simulation.run_simulation(visualise=(viz_choice == 'y'))
            
        elif choice == "4":
            print("\n Running comparison of all modes...")
            
            reports = {}
            
            # Basic Mode
            print("\n--- BASIC MODE ---")
            basic_sim = MountainRescueSimulation(
                num_terrain_robots=3,
                num_drones=2,
                num_missing_persons=6,
                max_simulation_steps=250,
                operation_mode=OperationMode.BASIC
            )
            reports['basic'] = basic_sim.run_simulation(visualise=False, step_delay=0)
            
            # Extended Mode
            print("\n--- EXTENDED MODE ---")
            extended_sim = MountainRescueSimulation(
                num_terrain_robots=3,
                num_drones=2,
                num_missing_persons=4,
                max_simulation_steps=300,
                operation_mode=OperationMode.EXTENDED,
                spawn_new_persons=True,
                spawn_interval=40
            )
            reports['extended'] = extended_sim.run_simulation(visualise=False, step_delay=0)
            
            # Novel Mode
            print("\n--- NOVEL MODE (Q-LEARNING) ---")
            novel_sim = MountainRescueSimulation(
                num_terrain_robots=3,
                num_drones=2,
                num_missing_persons=5,
                max_simulation_steps=400,
                operation_mode=OperationMode.NOVEL,
                spawn_new_persons=True,
                spawn_interval=50,
                enable_learning=True
            )
            reports['novel'] = novel_sim.run_simulation(visualise=False, step_delay=0)
            
            # Comparison
            print("\n" + "="*80)
            print("MODE COMPARISON RESULTS")
            print("="*80)
            print(f"{'Metric':<30} {'Basic':>15} {'Extended':>15} {'Novel (Q-L)':>15}")
            print("-"*80)
            
            metrics = [
                ('Success Rate (%)', 'success_rate', lambda x: f"{x*100:.1f}"),
                ('Avg Rescue Time (steps)', 'avg_rescue_time', lambda x: f"{x:.1f}"),
                ('Battery Efficiency', 'battery_efficiency', lambda x: f"{x:.3f}"),
                ('Total Battery Used', 'total_battery_consumed', str),
                ('Persons Rescued', 'persons_rescued', str)
            ]
            
            for metric_name, metric_key, formatter in metrics:
                print(f"{metric_name:<30}", end="")
                for mode in ['basic', 'extended', 'novel']:
                    value = reports[mode].get(metric_key, 0)
                    print(f"{formatter(value):>15}", end="")
                print()
            
            # Learning-specific metrics for Novel mode
            if reports['novel'].get('learning_stats'):
                print("\n" + "-"*80)
                print("NOVEL MODE LEARNING METRICS:")
                learning = reports['novel']['learning_stats']
                print(f"  States Explored: {learning.get('total_states_explored', 0)}")
                print(f"  Terrain Mapped: {learning.get('total_terrain_mapped', 0)}")
                print(f"  Known Rescue Locations: {learning.get('total_rescue_locations', 0)}")
            
            print("="*80)
            
        else:
            print("Invalid choice. Running Novel Mode by default.")
            simulation = MountainRescueSimulation(operation_mode=OperationMode.NOVEL)
            report = simulation.run_simulation(visualise=True)
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nSimulation error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()