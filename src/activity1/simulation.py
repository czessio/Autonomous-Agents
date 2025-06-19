# Main simulation controller for mountain rescue robots - coordinates environment, agents, and visualisation

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from typing import List


from environment import Environment
from terrain_robot import TerrainRobot, RobotState
from explorer_drone import ExplorerDrone, DroneState
from coordination import BasicCommunication


class MountainRescueSimulation:
    def __init__(self, num_terrain_robots: int = 3, num_drones: int = 2, 
                 num_missing_persons: int = 6, max_simulation_steps: int = 200):
        # Simulation parameters - increased time limit
        self.max_steps = max_simulation_steps
        self.current_step = 0
        self.running = True
        
        # Create environment
        self.environment = Environment(
            width=20, height=15,
            num_missing_persons=num_missing_persons,
            num_terrain_robots=num_terrain_robots,
            num_drones=num_drones
        )
        
        # Create terrain robots
        self.terrain_robots = []
        for i, base_pos in enumerate(self.environment.terrain_robot_base):
            robot = TerrainRobot(
                robot_id=i,
                start_position=base_pos,
                max_battery=120,  # Increased battery
                battery_drain_rate=1
            )
            self.terrain_robots.append(robot)
        
        # Create explorer drones
        self.drones = []
        for i, base_pos in enumerate(self.environment.drone_base):
            drone = ExplorerDrone(
                drone_id=i,
                start_position=base_pos,
                max_battery=120,  # Standardized battery
                battery_drain_rate=2
            )
            self.drones.append(drone)
        
        # Communication system for Basic Mode
        self.communication = BasicCommunication()  # Drone-robot communication
        
        # Start agents immediately (they have full battery)
        for robot in self.terrain_robots:
            robot.state = RobotState.SEARCHING
        for drone in self.drones:
            drone.state = DroneState.EXPLORING
            drone._generate_search_pattern(self.environment)
        
        # Statistics tracking
        self.stats = {
            'total_persons_rescued': 0,
            'total_battery_consumed': 0,
            'mission_start_time': None,
            'mission_end_time': None,
            'steps_to_rescue': {}
        }
        
        # Animation setup
        self.fig = None
        self.ax = None
        self.animation = None
    
    def run_simulation(self, visualise: bool = True, step_delay: float = 0.5) -> dict:
        """Run the complete simulation"""
        print("Starting Mountain Rescue Simulation...")
        print(f"Environment: {self.environment.width}x{self.environment.height}")
        print(f"Missing persons: {len(self.environment.missing_persons)}")
        print(f"Terrain robots: {len(self.terrain_robots)}")
        print(f"Explorer drones: {len(self.drones)}")
        print("-" * 50)
        
        # Ensure communication system is initialized (failsafe)
        if not hasattr(self, 'communication'):
            self.communication = BasicCommunication()
        
        self.stats['mission_start_time'] = time.time()
        
        print("Agents will activate automatically when ready...")
        
        try:
            if visualise:
                print("Attempting animated simulation...")
                self._setup_visualisation()
                self._run_animated_simulation()
            else:
                print("Running batch simulation...")
                self._run_batch_simulation(step_delay)
        except Exception as e:
            print(f"Simulation error: {e}")
            print("Falling back to batch mode...")
            plt.close('all')
            self._run_batch_simulation(step_delay)
        
        self.stats['mission_end_time'] = time.time()
        return self._generate_final_report()
    
    def _run_batch_simulation(self, step_delay: float) -> None:
        """Run simulation without animation"""
        print(f"Starting batch simulation with {self.max_steps} max steps...")
        
        while self.running and self.current_step < self.max_steps:
            self._update_simulation_step()
            
            if step_delay > 0:
                time.sleep(step_delay)
            
            # Print progress every 25 steps
            if self.current_step % 25 == 0:
                self._print_progress()
            
            self.current_step += 1
        
        print(f"Batch simulation completed after {self.current_step} steps")
    
    def _run_animated_simulation(self) -> None:
        """Run simulation with matplotlib animation"""
        def animate(frame):
            if self.running and self.current_step < self.max_steps:
                self._update_simulation_step()
                self._update_visualisation()
                self.current_step += 1
                
                # Print progress every 25 steps during animation
                if self.current_step % 25 == 0:
                    self._print_progress()
            else:
                # Stop animation when simulation ends
                if hasattr(self, 'animation') and self.animation:
                    self.animation.event_source.stop()
            return []
        
        print("Setting up matplotlib animation...")
        
        try:
            # Store animation in instance variable to prevent garbage collection
            self.animation = animation.FuncAnimation(
                self.fig, animate, interval=350, blit=False, cache_frame_data=False
            )
            
            print("Animation created, showing plot...")
            plt.show(block=True)  # Block until window is closed
            
        except Exception as e:
            print(f"Animation failed: {e}")
            raise e
    
    def _update_simulation_step(self) -> None:
        """Update all agents and environment for one time step"""
        # Debug print for first few steps
        if self.current_step < 3:
            print(f"Step {self.current_step}: Updating agents...")
        
        # Update drones first (they find people and communicate locations)
        for drone in self.drones:
            drone.update(self.environment, self.current_step, self.communication)
        
        # Update terrain robots (they respond to drone communications)
        for robot in self.terrain_robots:
            robot.update(self.environment, self.current_step, self.communication)
        
        # Check mission completion
        self._check_mission_status()
        
        # Update statistics
        self._update_statistics()
        
        # Debug print for first few steps
        if self.current_step < 3:
            print(f"Step {self.current_step}: Mission running = {self.running}")
            for i, robot in enumerate(self.terrain_robots):
                print(f"  Robot {i}: {robot.state.value}, battery: {robot.current_battery}%")
            for i, drone in enumerate(self.drones):
                print(f"  Drone {i}: {drone.state.value}, battery: {drone.current_battery}%")
    
    def _setup_visualisation(self) -> None:
        """Setup matplotlib visualisation"""
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.suptitle('Mountain Rescue Simulation - Activity 1 (Basic Mode)', 
                         fontsize=16, fontweight='bold')
        
        # Initial visualisation
        self._update_visualisation()
    
    def _update_visualisation(self) -> None:
        """Update the visualisation with current state"""
        self.ax.clear()
        
        # Use environment's visualisation method
        self.environment.visualise(self.terrain_robots, self.drones, ax=self.ax)
        
        # Add status information overlay
        self._add_status_text()
        
        # Set title
        self.ax.set_title(f'Mountain Rescue Simulation - Step {self.current_step}', 
                         fontsize=14, fontweight='bold')
    
    def _add_status_text(self) -> None:
        """Add status information to the visualisation"""
        status_text = f"Step: {self.current_step}/{self.max_steps}\n"
        status_text += f"Missing Persons: {len(self.environment.missing_persons)}\n"
        status_text += f"Rescued: {len(self.environment.rescued_persons)}\n\n"
        
        # Robot status
        status_text += "Terrain Robots:\n"
        for robot in self.terrain_robots:
            status = robot.get_status()
            status_text += f"  R{status['id']}: {status['state']} (Battery: {status['battery']}%)\n"
        
        status_text += "\nDrones:\n"
        for drone in self.drones:
            status = drone.get_status()
            status_text += f"  D{status['id']}: {status['state']} (Battery: {status['battery']}%)\n"
        
        # Add legend
        status_text += "\nLegend:\n"
        status_text += "P = Person  🤖 = Robot  🚁 = Drone\n"
        status_text += "RB = Robot Base  DB = Drone Base\n"
        status_text += "Colors: Gold=Search, Orange=Action\n"
        status_text += "Cyan=Explore, Red=Low Battery\n"
        status_text += "Gray=At Base"
        
        # Add text box
        self.ax.text(0.02, 0.98, status_text, transform=self.ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.9), fontfamily='monospace', fontsize=8)
    
    def _check_mission_status(self) -> None:
        """Check if mission is complete or should be terminated"""
        # Mission complete if all persons rescued
        if len(self.environment.missing_persons) == 0:
            print(f"\n🎉 Mission Complete! All persons rescued in {self.current_step} steps.")
            self.running = False
            return
        
        # Check if all agents are out of battery AND stuck at base
        all_agents_inactive = True
        
        for robot in self.terrain_robots:
            # Agent is active if it has battery OR is not stuck at base with low battery
            if robot.current_battery > robot.low_battery_threshold or robot.state.value not in ['at_base', 'battery_low']:
                all_agents_inactive = False
                break
        
        if all_agents_inactive:
            for drone in self.drones:
                # Agent is active if it has battery OR is not stuck at base with low battery  
                if drone.current_battery > drone.low_battery_threshold or drone.state.value not in ['at_base', 'battery_low']:
                    all_agents_inactive = False
                    break
        
        if all_agents_inactive:
            print(f"\n Mission terminated: All agents out of battery. Step {self.current_step}")
            self.running = False
    
    def _update_statistics(self) -> None:
        """Update simulation statistics"""
        # Count total rescued
        self.stats['total_persons_rescued'] = len(self.environment.rescued_persons)
        
        # Calculate total battery consumed
        total_battery = 0
        for robot in self.terrain_robots:
            total_battery += (robot.max_battery - robot.current_battery)
        for drone in self.drones:
            total_battery += (drone.max_battery - drone.current_battery)
        
        self.stats['total_battery_consumed'] = total_battery
    
    def _print_progress(self) -> None:
        """Print simulation progress"""
        rescued = len(self.environment.rescued_persons)
        total = rescued + len(self.environment.missing_persons)
        
        print(f"Step {self.current_step}: {rescued}/{total} persons rescued")
        
        # Print agent status
        active_robots = sum(1 for r in self.terrain_robots if r.current_battery > r.low_battery_threshold)
        active_drones = sum(1 for d in self.drones if d.current_battery > d.low_battery_threshold)
        
        print(f"  Active agents: {active_robots} robots, {active_drones} drones")
    
    def _generate_final_report(self) -> dict:
        """Generate final simulation report"""
        mission_time = self.stats['mission_end_time'] - self.stats['mission_start_time']
        
        report = {
            'simulation_steps': self.current_step,
            'mission_time_seconds': round(mission_time, 2),
            'persons_rescued': self.stats['total_persons_rescued'],
            'persons_remaining': len(self.environment.missing_persons),
            'total_battery_consumed': self.stats['total_battery_consumed'],
            'success_rate': self.stats['total_persons_rescued'] / 
                           (self.stats['total_persons_rescued'] + len(self.environment.missing_persons)),
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
                'final_state': status['state']
            })
        
        for drone in self.drones:
            status = drone.get_status()
            report['drone_performance'].append({
                'id': status['id'],
                'persons_found': status['found_persons'],
                'areas_explored': status['exploring'],
                'final_battery': status['battery'],
                'final_state': status['state']
            })
        
        self._print_final_report(report)
        return report
    
    def _print_final_report(self, report: dict) -> None:
        """Print formatted final report"""
        print("\n" + "="*60)
        print("MOUNTAIN RESCUE SIMULATION - FINAL REPORT")
        print("="*60)
        print(f"Mission Duration: {report['simulation_steps']} steps ({report['mission_time_seconds']}s)")
        print(f"Success Rate: {report['success_rate']:.1%}")
        print(f"Persons Rescued: {report['persons_rescued']}")
        print(f"Persons Remaining: {report['persons_remaining']}")
        print(f"Total Battery Consumed: {report['total_battery_consumed']} units")
        
        print("\nTerrain Robot Performance:")
        for robot_perf in report['robot_performance']:
            print(f"  Robot {robot_perf['id']}: {robot_perf['persons_rescued']} rescued, "
                  f"{robot_perf['final_battery']}% battery, {robot_perf['final_state']}")
        
        print("\nDrone Performance:")
        for drone_perf in report['drone_performance']:
            print(f"  Drone {drone_perf['id']}: {drone_perf['persons_found']} found, "
                  f"{drone_perf['areas_explored']} areas explored, "
                  f"{drone_perf['final_battery']}% battery, {drone_perf['final_state']}")
        
        print("="*60)


def main():
    """Main function to run the simulation"""
    # Create and run simulation with longer time limit
    simulation = MountainRescueSimulation(
        num_terrain_robots=3,
        num_drones=2,
        num_missing_persons=6,
        max_simulation_steps=200  # Increased time limit
    )
    
    # Ask user for mode
    print("\nSelect simulation mode:")
    print("1. Animated (with matplotlib visualisation)")
    print("2. Batch (text-based, faster)")
    
    try:
        choice = input("Enter choice (1/2) or press Enter for batch mode: ").strip()
        
        if choice == "1":
            # Run with visualisation
            final_report = simulation.run_simulation(visualise=True, step_delay=0)
        else:
            # Run in batch mode (safer)
            final_report = simulation.run_simulation(visualise=False, step_delay=0.05)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        final_report = simulation._generate_final_report()
    except Exception as e:
        print(f"\nSimulation error: {e}")
        final_report = simulation._generate_final_report()
    
    return final_report


if __name__ == "__main__":
    final_report = main()