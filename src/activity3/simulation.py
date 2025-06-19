# Updated simulation controller for Activity 3 - includes Extended Mode operation

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random
from typing import List, Dict
from enum import Enum

# Import from activity1 (reusing existing components)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from activity1.environment import Environment
from activity1.terrain_robot import TerrainRobot, RobotState
from activity1.explorer_drone import ExplorerDrone, DroneState


# Import new communication system
from communication import BasicCommunication, ExtendedCommunication, MessageType


class OperationMode(Enum):
    BASIC = "basic"
    EXTENDED = "extended"


class MountainRescueSimulation:
    def __init__(self, num_terrain_robots: int = 3, num_drones: int = 2, 
                 num_missing_persons: int = 6, max_simulation_steps: int = 300,
                 operation_mode: OperationMode = OperationMode.BASIC,
                 spawn_new_persons: bool = False, spawn_interval: int = 50):
        
        # Simulation parameters
        self.max_steps = max_simulation_steps
        self.current_step = 0
        self.running = True
        self.operation_mode = operation_mode
        
        # Dynamic person spawning (Extended Mode)
        self.spawn_new_persons = spawn_new_persons and (operation_mode == OperationMode.EXTENDED)
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
        if self.operation_mode == OperationMode.EXTENDED:
            self.communication = ExtendedCommunication()
            print(" Starting in EXTENDED MODE with advanced communication")
        else:
            self.communication = BasicCommunication()
            print(" Starting in BASIC MODE")
        
        # Create terrain robots
        self.terrain_robots = []
        for i, base_pos in enumerate(self.environment.terrain_robot_base):
            robot = TerrainRobot(
                robot_id=i,
                start_position=base_pos,
                max_battery=120,
                battery_drain_rate=1
            )
            self.terrain_robots.append(robot)
            
            # Register with communication system
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
            
            # Register with communication system
            if isinstance(self.communication, ExtendedCommunication):
                self.communication.register_agent(f"drone_{i}", "drone")
        
        # Mode-specific initialization
        if self.operation_mode == OperationMode.BASIC:
            # Start agents immediately in Basic Mode
            for robot in self.terrain_robots:
                robot.state = RobotState.SEARCHING
            for drone in self.drones:
                drone.state = DroneState.EXPLORING
                drone._generate_search_pattern(self.environment)
        else:
            # Extended Mode: robots wait at base, drones start exploring
            for robot in self.terrain_robots:
                robot.state = RobotState.AT_BASE
                # Override minimum battery threshold for Extended Mode
                robot.min_leave_threshold = 50  # Can leave with lower battery
            for drone in self.drones:
                drone.state = DroneState.EXPLORING
                drone._generate_search_pattern(self.environment)
        
        # Statistics tracking
        self.stats = {
            'total_persons_rescued': 0,
            'total_persons_spawned': num_missing_persons,
            'total_battery_consumed': 0,
            'rescue_times': [],  # Time from spawn to rescue
            'person_spawn_times': {},  # Track when each person was spawned
            'communication_stats': {},
            'mode': operation_mode.value,
            'mission_start_time': None,
            'mission_end_time': None
        }
        
        # Initialize spawn times for initial persons
        for person_loc in self.environment.missing_persons:
            self.stats['person_spawn_times'][person_loc] = 0
        
        # Animation setup
        self.fig = None
        self.ax = None
        self.animation = None
    
    def switch_operation_mode(self, new_mode: OperationMode) -> None:
        """Switch between Basic and Extended operation modes"""
        if new_mode == self.operation_mode:
            return
        
        print(f"\n Switching from {self.operation_mode.value} to {new_mode.value} mode...")
        self.operation_mode = new_mode
        
        # Switch communication system
        if new_mode == OperationMode.EXTENDED:
            old_comm = self.communication
            self.communication = ExtendedCommunication()
            
            # Register all agents
            for i, robot in enumerate(self.terrain_robots):
                self.communication.register_agent(f"robot_{i}", "robot")
            for i, drone in enumerate(self.drones):
                self.communication.register_agent(f"drone_{i}", "drone")
            
            # Enable dynamic spawning
            self.spawn_new_persons = True
            
            print(" Extended Mode activated: Advanced communication enabled")
            
        else:
            self.communication = BasicCommunication()
            self.spawn_new_persons = False
            print(" Basic Mode activated: Simple coordination enabled")
    
    def _update_simulation_step(self) -> None:
        """Update all agents and environment for one time step"""
        
        # Extended Mode: Handle dynamic person spawning
        if self.spawn_new_persons and self.current_step > 0:
            if (self.current_step - self.last_spawn_step) >= self.spawn_interval:
                self._spawn_new_person()
        
        # Extended Mode: Update robot availability in communication system
        if isinstance(self.communication, ExtendedCommunication):
            for i, robot in enumerate(self.terrain_robots):
                available = robot.state == RobotState.AT_BASE and robot.has_first_aid_kit
                self.communication.update_robot_status(
                    f"robot_{i}", 
                    robot.current_battery,
                    available
                )
        
        # Update drones first (they find people and communicate)
        for drone in self.drones:
            # Extended Mode: Check for messages
            if isinstance(self.communication, ExtendedCommunication):
                messages = self.communication.get_messages(f"drone_{drone.drone_id}")
                # Process any messages if needed
            
            drone.update(self.environment, self.current_step, self.communication)
        
        # Update terrain robots
        for robot in self.terrain_robots:
            # Extended Mode: Check for rescue assignments
            if self.operation_mode == OperationMode.EXTENDED:
                self._handle_extended_mode_robot(robot)
            else:
                robot.update(self.environment, self.current_step, self.communication)
        
        # Track rescue times
        self._update_rescue_statistics()
        
        # Check mission completion
        self._check_mission_status()
        
        # Update statistics
        self._update_statistics()
    
    def _handle_extended_mode_robot(self, robot: TerrainRobot) -> None:
        """Handle robot behavior in Extended Mode"""
        robot_id = f"robot_{robot.robot_id}"
        
        # Get messages for this robot
        messages = self.communication.get_messages(robot_id)
        
        # Process messages
        for message in messages:
            if message.msg_type == MessageType.PERSON_FOUND:
                # Drone found a person
                if robot.state == RobotState.AT_BASE and robot.has_first_aid_kit and robot.current_battery > 50:
                    # Try to assign this robot to the rescue
                    if self.communication.assign_robot_to_rescue(robot_id, message.location):
                        robot.assigned_location = message.location
                        robot.state = RobotState.SEARCHING
                        print(f" Robot {robot.robot_id} responding to rescue at {message.location}")
            
            elif message.msg_type == MessageType.RESCUE_REQUEST:
                # Direct request from drone
                if robot.state == RobotState.AT_BASE and robot.has_first_aid_kit and robot.current_battery > 50:
                    if self.communication.assign_robot_to_rescue(robot_id, message.location):
                        robot.assigned_location = message.location
                        robot.state = RobotState.SEARCHING
                        print(f" Robot {robot.robot_id} assigned to rescue at {message.location}")
        
        # Update robot normally
        robot.update(self.environment, self.current_step, self.communication)
        
        # Check if robot completed a rescue
        if hasattr(robot, '_last_rescued_location'):
            self.communication.complete_rescue(robot_id, robot._last_rescued_location)
            delattr(robot, '_last_rescued_location')
    
    def _spawn_new_person(self) -> None:
        """Spawn a new missing person in the mountain area"""
        x_start, y_start, x_end, y_end = self.environment.mountain_area['bounds']
        
        # Try to find an empty location
        max_attempts = 20
        for _ in range(max_attempts):
            x = random.randint(x_start, x_end - 1)
            y = random.randint(y_start, y_end - 1)
            location = (x, y)
            
            # Check if location is empty
            if location not in self.environment.missing_persons and \
               location not in self.environment.rescued_persons:
                # Add new person
                self.environment.missing_persons.append(location)
                self.stats['person_spawn_times'][location] = self.current_step
                self.total_persons_spawned += 1
                self.last_spawn_step = self.current_step
                
                print(f" New person appeared at {location} (step {self.current_step})")
                return
    
    def _update_rescue_statistics(self) -> None:
        """Track rescue times for KPI analysis"""
        # Check for newly rescued persons
        for person_loc in self.environment.rescued_persons:
            if person_loc in self.stats['person_spawn_times']:
                spawn_time = self.stats['person_spawn_times'][person_loc]
                rescue_time = self.current_step - spawn_time
                
                # Only add if not already tracked
                if person_loc not in [r['location'] for r in self.stats['rescue_times']]:
                    self.stats['rescue_times'].append({
                        'location': person_loc,
                        'spawn_step': spawn_time,
                        'rescue_step': self.current_step,
                        'rescue_time': rescue_time
                    })
    
    def _check_mission_status(self) -> None:
        """Check if mission is complete or should be terminated"""
        # In Extended Mode, mission continues as long as agents are operational
        if self.operation_mode == OperationMode.EXTENDED:
            # Check if all agents are completely inactive
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
            # Basic Mode: End when all initial persons rescued or agents inactive
            if len(self.environment.missing_persons) == 0:
                print(f"\n Mission Complete! All persons rescued in {self.current_step} steps.")
                self.running = False
                return
            
            # Check battery status
            all_agents_inactive = True
            for robot in self.terrain_robots:
                if robot.current_battery > robot.low_battery_threshold or \
                   robot.state.value not in ['at_base', 'battery_low']:
                    all_agents_inactive = False
                    break
            
            if all_agents_inactive:
                print(f"\n Mission terminated: All agents out of battery. Step {self.current_step}")
                self.running = False
    
    def _update_statistics(self) -> None:
        """Update simulation statistics including communication metrics"""
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
        
        # Communication statistics (Extended Mode)
        if isinstance(self.communication, ExtendedCommunication):
            self.stats['communication_stats'] = self.communication.get_communication_stats()
    
    def _add_status_text(self) -> None:
        """Add enhanced status information to visualization"""
        status_text = f"Mode: {self.operation_mode.value.upper()} | Step: {self.current_step}/{self.max_steps}\n"
        status_text += f"Missing: {len(self.environment.missing_persons)} | "
        status_text += f"Rescued: {len(self.environment.rescued_persons)}"
        
        if self.spawn_new_persons:
            status_text += f" | Total Spawned: {self.total_persons_spawned}\n"
        else:
            status_text += "\n"
        
        # Communication stats for Extended Mode
        if isinstance(self.communication, ExtendedCommunication):
            comm_stats = self.communication.get_communication_stats()
            status_text += f"Messages: {comm_stats['total_messages']} | "
            status_text += f"Pending: {comm_stats['pending_rescues']} | "
            status_text += f"Active: {comm_stats['active_rescues']}\n"
        
        status_text += "\nTerrain Robots:\n"
        for robot in self.terrain_robots:
            status = robot.get_status()
            status_text += f"  R{status['id']}: {status['state']} (Battery: {status['battery']}%)"
            if hasattr(robot, 'assigned_location') and robot.assigned_location:
                status_text += f" → {robot.assigned_location}"
            status_text += "\n"
        
        status_text += "\nDrones:\n"
        for drone in self.drones:
            status = drone.get_status()
            status_text += f"  D{status['id']}: {status['state']} (Battery: {status['battery']}%)\n"
        
        # Add legend
        status_text += "\nLegend:\n"
        status_text += "P = Person   = Robot   = Drone\n"
        status_text += "RB = Robot Base  DB = Drone Base"
        
        # Add text box
        self.ax.text(0.02, 0.98, status_text, transform=self.ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.9), fontfamily='monospace', fontsize=8)
    
    def _update_visualisation(self) -> None:
        """Update the visualisation with current state"""
        self.ax.clear()
        
        # Use environment's visualisation method
        self.environment.visualise(self.terrain_robots, self.drones, ax=self.ax)
        
        # Add enhanced status information
        self._add_status_text()
        
        # Set title with mode indication
        mode_str = "Extended Mode" if self.operation_mode == OperationMode.EXTENDED else "Basic Mode"
        self.ax.set_title(f'Mountain Rescue Simulation - Activity 3 ({mode_str}) - Step {self.current_step}', 
                         fontsize=14, fontweight='bold')
    
    def run_simulation(self, visualise: bool = True, step_delay: float = 0.5) -> dict:
        """Run the complete simulation"""
        print("\n" + "="*60)
        print("MOUNTAIN RESCUE SIMULATION - ACTIVITY 3")
        print("="*60)
        print(f"Mode: {self.operation_mode.value.upper()}")
        print(f"Environment: {self.environment.width}x{self.environment.height}")
        print(f"Missing persons: {len(self.environment.missing_persons)}")
        print(f"Terrain robots: {len(self.terrain_robots)}")
        print(f"Explorer drones: {len(self.drones)}")
        
        if self.spawn_new_persons:
            print(f"Dynamic spawning: Every {self.spawn_interval} steps")
        
        print("-" * 60)
        
        self.stats['mission_start_time'] = time.time()
        
        try:
            if visualise:
                self._setup_visualisation()
                self._run_animated_simulation()
            else:
                self._run_batch_simulation(step_delay)
        except Exception as e:
            print(f"Simulation error: {e}")
            self._run_batch_simulation(step_delay)
        
        self.stats['mission_end_time'] = time.time()
        return self._generate_final_report()
    
    def _setup_visualisation(self) -> None:
        """Setup matplotlib visualisation"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        mode_str = "Extended Mode" if self.operation_mode == OperationMode.EXTENDED else "Basic Mode"
        self.fig.suptitle(f'Mountain Rescue Simulation - Activity 3 ({mode_str})', 
                         fontsize=16, fontweight='bold')
        self._update_visualisation()
    
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
        """Print simulation progress with mode-specific information"""
        rescued = len(self.environment.rescued_persons)
        missing = len(self.environment.missing_persons)
        
        print(f"\nStep {self.current_step}: {rescued} rescued, {missing} missing")
        
        if self.operation_mode == OperationMode.EXTENDED:
            if isinstance(self.communication, ExtendedCommunication):
                comm_stats = self.communication.get_communication_stats()
                print(f"  Communication: {comm_stats['total_messages']} messages, "
                      f"{comm_stats['active_rescues']} active rescues")
        
        active_robots = sum(1 for r in self.terrain_robots if r.current_battery > r.low_battery_threshold)
        active_drones = sum(1 for d in self.drones if d.current_battery > d.low_battery_threshold)
        print(f"  Active agents: {active_robots} robots, {active_drones} drones")
    
    def _generate_final_report(self) -> dict:
        """Generate comprehensive final simulation report"""
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
            'robot_performance': [],
            'drone_performance': []
        }
        
        # Individual agent performance
        for robot in self.terrain_robots:
            status = robot.get_status()
            report['robot_performance'].append({
                'id': status['id'],
                'persons_rescued': status['rescued'],
                'final_battery': status['battery'],
                'battery_used': robot.max_battery - status['battery'],
                'final_state': status['state']
            })
        
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
        
        self._print_final_report(report)
        return report
    
    def _print_final_report(self, report: dict) -> None:
        """Print formatted final report"""
        print("\n" + "="*60)
        print(f"FINAL REPORT - {report['mode'].upper()} MODE")
        print("="*60)
        print(f"Mission Duration: {report['simulation_steps']} steps ({report['mission_time_seconds']}s)")
        print(f"Success Rate: {report['success_rate']:.1%}")
        print(f"Persons Rescued: {report['persons_rescued']} / {report['persons_spawned']} spawned")
        print(f"Persons Remaining: {report['persons_remaining']}")
        
        print(f"\n KEY PERFORMANCE INDICATORS:")
        print(f"  Average Rescue Time: {report['avg_rescue_time']} steps")
        print(f"  Battery Efficiency: {report['battery_efficiency']:.3f} rescues/battery unit")
        print(f"  Total Battery Consumed: {report['total_battery_consumed']} units")
        
        if report['communication_stats']:
            print(f"\n COMMUNICATION STATISTICS:")
            comm = report['communication_stats']
            print(f"  Total Messages: {comm.get('total_messages', 0)}")
            print(f"  Individual Messages: {comm.get('individual_messages', 0)}")
            print(f"  Broadcast Messages: {comm.get('broadcast_messages', 0)}")
            print(f"  Message Efficiency: {comm.get('message_efficiency', 0):.2%}")
        
        print("\n TERRAIN ROBOT PERFORMANCE:")
        for robot_perf in report['robot_performance']:
            print(f"  Robot {robot_perf['id']}: {robot_perf['persons_rescued']} rescued, "
                  f"{robot_perf['battery_used']} battery used, "
                  f"{robot_perf['final_battery']}% remaining")
        
        print("\n DRONE PERFORMANCE:")
        for drone_perf in report['drone_performance']:
            print(f"  Drone {drone_perf['id']}: {drone_perf['persons_found']} found, "
                  f"{drone_perf['areas_explored']} areas explored, "
                  f"{drone_perf['battery_used']} battery used")
        
        print("="*60)


def main():
    """Main function to run the simulation with mode selection"""
    print("\n  MOUNTAIN RESCUE SIMULATION - ACTIVITY 3")
    print("="*50)
    print("Select operation mode:")
    print("1. Basic Mode (Activity 1 behavior)")
    print("2. Extended Mode (Advanced communication)")
    print("3. Mode Comparison (Run both modes)")

    report = None  # Ensure report is always defined

    try:
        choice = input("\nEnter choice (1/2/3): ").strip()

        if choice == "1":
            simulation = MountainRescueSimulation(
                num_terrain_robots=3,
                num_drones=2,
                num_missing_persons=6,
                max_simulation_steps=250,
                operation_mode=OperationMode.BASIC
            )
            viz_choice = input("Enable visualization? (y/n): ").strip().lower()
            report = simulation.run_simulation(visualise=(viz_choice == 'y'), step_delay=0.05)

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
            report = simulation.run_simulation(visualise=(viz_choice == 'y'), step_delay=0.05)

        elif choice == "3":
            print("\n Running comparison between Basic and Extended modes...")

            print("\n--- BASIC MODE ---")
            basic_sim = MountainRescueSimulation(
                num_terrain_robots=3,
                num_drones=2,
                num_missing_persons=6,
                max_simulation_steps=250,
                operation_mode=OperationMode.BASIC
            )
            basic_report = basic_sim.run_simulation(visualise=False, step_delay=0)

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
            extended_report = extended_sim.run_simulation(visualise=False, step_delay=0)

            print("\n" + "="*60)
            print("COMPARISON RESULTS")
            print("="*60)
            print(f"{'Metric':<30} {'Basic Mode':>15} {'Extended Mode':>15}")
            print("-"*60)
            print(f"{'Success Rate':<30} {basic_report['success_rate']:>14.1%} {extended_report['success_rate']:>14.1%}")
            print(f"{'Avg Rescue Time (steps)':<30} {basic_report['avg_rescue_time']:>15.1f} {extended_report['avg_rescue_time']:>15.1f}")
            print(f"{'Battery Efficiency':<30} {basic_report['battery_efficiency']:>15.3f} {extended_report['battery_efficiency']:>15.3f}")
            print(f"{'Total Battery Used':<30} {basic_report['total_battery_consumed']:>15} {extended_report['total_battery_consumed']:>15}")
            print(f"{'Persons Rescued':<30} {basic_report['persons_rescued']:>15} {extended_report['persons_rescued']:>15}")
            print("="*60)

        else:
            print("Invalid choice. Running Basic Mode by default.")
            simulation = MountainRescueSimulation()
            report = simulation.run_simulation(visualise=True)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nSimulation error: {e}")

    return report



if __name__ == "__main__":
    report = main()