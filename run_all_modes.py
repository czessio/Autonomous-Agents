# Quick script to run all simulation modes for testing and comparison

import sys
import os
sys.path.append(os.path.abspath('src'))

from activity4.simulation import MountainRescueSimulation, OperationMode


def run_all_modes():
    """Run all three modes and display comparison"""
    print("\n" + "="*80)
    print("MOUNTAIN RESCUE ROBOTS - FULL SYSTEM DEMONSTRATION")
    print("="*80)
    
    results = {}
    
    # Run Basic Mode
    print("\n📍 Running BASIC MODE...")
    basic_sim = MountainRescueSimulation(
        num_terrain_robots=3,
        num_drones=2,
        num_missing_persons=6,
        max_simulation_steps=200,
        operation_mode=OperationMode.BASIC
    )
    results['basic'] = basic_sim.run_simulation(visualise=False, step_delay=0)
    
    # Run Extended Mode
    print("\n📡 Running EXTENDED MODE...")
    extended_sim = MountainRescueSimulation(
        num_terrain_robots=3,
        num_drones=2,
        num_missing_persons=5,
        max_simulation_steps=250,
        operation_mode=OperationMode.EXTENDED,
        spawn_new_persons=True,
        spawn_interval=40
    )
    results['extended'] = extended_sim.run_simulation(visualise=False, step_delay=0)
    
    # Run Novel Mode
    print("\n🧠 Running NOVEL MODE with Q-Learning...")
    novel_sim = MountainRescueSimulation(
        num_terrain_robots=3,
        num_drones=2,
        num_missing_persons=5,
        max_simulation_steps=300,
        operation_mode=OperationMode.NOVEL,
        spawn_new_persons=True,
        spawn_interval=50,
        enable_learning=True
    )
    results['novel'] = novel_sim.run_simulation(visualise=False, step_delay=0)
    
    # Display comprehensive comparison
    print("\n" + "="*90)
    print("COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*90)
    print(f"{'Metric':<35} {'Basic':>15} {'Extended':>15} {'Novel (Q-L)':>15}")
    print("-"*90)
    
    # Performance metrics
    metrics = [
        ('Success Rate (%)', 'success_rate', lambda x: f"{x*100:.1f}"),
        ('Average Rescue Time (steps)', 'avg_rescue_time', lambda x: f"{x:.1f}"),
        ('Battery Efficiency (rescues/unit)', 'battery_efficiency', lambda x: f"{x:.4f}"),
        ('Total Battery Consumed', 'total_battery_consumed', str),
        ('Total Persons Rescued', 'persons_rescued', str),
        ('Mission Duration (steps)', 'simulation_steps', str),
    ]
    
    for metric_name, metric_key, formatter in metrics:
        print(f"{metric_name:<35}", end="")
        for mode in ['basic', 'extended', 'novel']:
            value = results[mode].get(metric_key, 0)
            print(f"{formatter(value):>15}", end="")
        print()
    
    # Communication metrics
    if results['extended'].get('communication_stats') or results['novel'].get('communication_stats'):
        print("\n" + "-"*90)
        print("COMMUNICATION METRICS:")
        print(f"{'Total Messages Sent':<35}", end="")
        for mode in ['basic', 'extended', 'novel']:
            if mode == 'basic':
                print(f"{'N/A':>15}", end="")
            else:
                comm_stats = results[mode].get('communication_stats', {})
                print(f"{comm_stats.get('total_messages', 0):>15}", end="")
        print()
    
    # Learning metrics for Novel mode
    if results['novel'].get('learning_stats'):
        print("\n" + "-"*90)
        print("Q-LEARNING METRICS (Novel Mode Only):")
        learning = results['novel']['learning_stats']
        print(f"  Total States Explored: {learning.get('total_states_explored', 0)}")
        print(f"  Terrain Points Mapped: {learning.get('total_terrain_mapped', 0)}")
        print(f"  Known Rescue Locations: {learning.get('total_rescue_locations', 0)}")
        print(f"  Final Exploration Rate: {learning.get('average_exploration_rate', 0):.3f}")
    
    # Calculate improvements
    print("\n" + "-"*90)
    print("PERFORMANCE IMPROVEMENTS:")
    
    # Extended vs Basic
    rescue_time_improvement_1 = ((results['basic']['avg_rescue_time'] - results['extended']['avg_rescue_time']) / 
                                results['basic']['avg_rescue_time'] * 100)
    battery_eff_improvement_1 = ((results['extended']['battery_efficiency'] - results['basic']['battery_efficiency']) / 
                                results['basic']['battery_efficiency'] * 100)
    
    print(f"Extended vs Basic:")
    print(f"  - Rescue Time: {rescue_time_improvement_1:.1f}% faster")
    print(f"  - Battery Efficiency: {battery_eff_improvement_1:.1f}% better")
    
    # Novel vs Extended
    rescue_time_improvement_2 = ((results['extended']['avg_rescue_time'] - results['novel']['avg_rescue_time']) / 
                                results['extended']['avg_rescue_time'] * 100)
    battery_eff_improvement_2 = ((results['novel']['battery_efficiency'] - results['extended']['battery_efficiency']) / 
                                results['extended']['battery_efficiency'] * 100)
    
    print(f"\nNovel vs Extended:")
    print(f"  - Rescue Time: {rescue_time_improvement_2:.1f}% faster")
    print(f"  - Battery Efficiency: {battery_eff_improvement_2:.1f}% better")
    
    # Novel vs Basic (total improvement)
    rescue_time_improvement_total = ((results['basic']['avg_rescue_time'] - results['novel']['avg_rescue_time']) / 
                                    results['basic']['avg_rescue_time'] * 100)
    battery_eff_improvement_total = ((results['novel']['battery_efficiency'] - results['basic']['battery_efficiency']) / 
                                    results['basic']['battery_efficiency'] * 100)
    
    print(f"\nNovel vs Basic (Total):")
    print(f"  - Rescue Time: {rescue_time_improvement_total:.1f}% faster")
    print(f"  - Battery Efficiency: {battery_eff_improvement_total:.1f}% better")
    
    print("\n" + "="*90)
    print("✅ Full system demonstration complete!")
    print("="*90)


if __name__ == "__main__":
    run_all_modes()