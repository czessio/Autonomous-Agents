#!/usr/bin/env python3
# Quick test script for Novel Mode functionality

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from activity4.simulation import MountainRescueSimulation, OperationMode

def test_novel_mode():
    """Test Novel Mode with shorter duration"""
    print("\n" + "="*60)
    print("TESTING NOVEL MODE - Q-LEARNING")
    print("="*60)
    
    # Create simulation with easier parameters
    sim = MountainRescueSimulation(
        num_terrain_robots=2,
        num_drones=1,
        num_missing_persons=2,  # Start with just 2 people
        max_simulation_steps=100,  # Shorter test
        operation_mode=OperationMode.NOVEL,
        spawn_new_persons=False,  # No spawning for test
        enable_learning=True
    )
    
    # Run without visualization for speed
    report = sim.run_simulation(visualise=False, step_delay=0)
    
    # Check results
    print("\n" + "="*60)
    print("TEST RESULTS:")
    print(f"Success Rate: {report['success_rate']*100:.1f}%")
    print(f"Persons Rescued: {report['persons_rescued']}")
    print(f"Average Rescue Time: {report['avg_rescue_time']}")
    print("="*60)
    
    if report['success_rate'] > 0:
        print(" TEST PASSED - Robots are rescuing people!")
    else:
        print(" TEST FAILED - No rescues occurred")
        # Print debug info
        print("\nDEBUG INFO:")
        for robot_perf in report['robot_performance']:
            print(f"Robot {robot_perf['id']}: rescued={robot_perf['persons_rescued']}, "
                  f"battery_used={robot_perf['battery_used']}")

if __name__ == "__main__":
    test_novel_mode()