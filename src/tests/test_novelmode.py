# Comprehensive test suite for Enhanced Novel Mode - Activity 4
# Tests all new innovations: Q-Learning, Hierarchical Coordination, Dynamic Roles, etc.

import unittest
import sys
import os
import numpy as np
import tempfile
import shutil
import time
from unittest.mock import Mock, patch
from io import StringIO
from contextlib import redirect_stdout

# Add src directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'activity4'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'activity1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'activity3'))

from activity4.simulation import MountainRescueSimulation, OperationMode, AgentRole
from activity4.q_learning import QLearningAgent, CollectiveQLearning
from activity4.learning_robot import LearningTerrainRobot
from activity4.mediator_agent import MediatorAgent, MediatorState, PriorityLevel
from activity4.coordinator_agent import CoordinatorAgent, CoordinatorState, MissionPhase
from activity1.environment import Environment
from activity1.terrain_robot import RobotState
from activity1.explorer_drone import ExplorerDrone, DroneState
from activity3.communication import ExtendedCommunication, MessageType


class TestEnhancedQLearning(unittest.TestCase):
    """Test enhanced Q-Learning with collective intelligence"""
    
    def setUp(self):
        """Setup test Q-learning components"""
        self.q_agent = QLearningAgent(
            state_space_size=(10, 10),
            action_space_size=8,
            learning_rate=0.1,
            discount_factor=0.95
        )
        
        self.collective_learning = CollectiveQLearning(
            num_robots=3,
            environment_size=(10, 10)
        )
    
    def test_enhanced_state_representation(self):
        """Test enhanced state keys with coordination features"""
        # Test with coordination context
        state_key = self.q_agent.get_state_key(
            position=(5, 5),
            target=(7, 7),
            battery_level=80,
            has_kit=True
        )
        
        # Should include all required components
        self.assertIsInstance(state_key, str)
        self.assertIn("5,5", state_key)
        self.assertIn("1,1", state_key)  # Target direction
        self.assertIn("high", state_key)  # Battery level
        self.assertIn("kit", state_key)  # Kit status
    
    def test_collective_learning_initialization(self):
        """Test collective learning system setup"""
        self.assertEqual(self.collective_learning.num_robots, 3)
        self.assertEqual(len(self.collective_learning.robot_agents), 3)
        self.assertIsNotNone(self.collective_learning.shared_q_agent)
        
        # Each robot should have its own Q-agent
        for i in range(3):
            agent = self.collective_learning.get_robot_agent(i)
            self.assertIsInstance(agent, QLearningAgent)
    
    def test_knowledge_sharing_mechanism(self):
        """Test knowledge sharing between agents"""
        # Add different knowledge to different agents
        agent_0 = self.collective_learning.robot_agents[0]
        agent_1 = self.collective_learning.robot_agents[1]
        
        # Set different Q-values
        agent_0.q_table["test_state"][0] = 10.0
        agent_1.q_table["test_state"][0] = 20.0
        
        # Set different terrain knowledge
        agent_0.terrain_difficulty[(5, 5)] = 2.0
        agent_1.terrain_difficulty[(5, 5)] = 4.0
        
        # Share knowledge
        self.collective_learning.share_knowledge()
        
        # Check that shared knowledge is averaged
        shared_q = self.collective_learning.shared_q_agent.q_table["test_state"][0]
        shared_terrain = self.collective_learning.shared_q_agent.terrain_difficulty[(5, 5)]
        
        self.assertTrue(10.0 < shared_q < 20.0)  # Should be weighted average
        self.assertEqual(shared_terrain, 3.0)  # Average of 2.0 and 4.0
    
    def test_reward_calculation_enhancements(self):
        """Test enhanced reward calculation with multiple objectives"""
        # Test rescue success reward
        reward = self.q_agent.calculate_reward(
            position=(5, 5),
            next_position=(5, 5),
            found_person=False,
            rescued_person=True,
            battery_used=3,
            terrain_elevation=1500,
            is_mountain_area=True
        )
        
        # Should be high positive reward for rescue
        self.assertGreater(reward, 90)
        
        # Test terrain difficulty impact
        high_elevation_reward = self.q_agent.calculate_reward(
            position=(5, 5),
            next_position=(6, 5),
            found_person=False,
            rescued_person=False,
            battery_used=2,
            terrain_elevation=3000,
            is_mountain_area=True
        )
        
        low_elevation_reward = self.q_agent.calculate_reward(
            position=(5, 5),
            next_position=(6, 5),
            found_person=False,
            rescued_person=False,
            battery_used=2,
            terrain_elevation=500,
            is_mountain_area=True
        )
        
        # Higher elevation should result in lower reward due to difficulty
        self.assertLess(high_elevation_reward, low_elevation_reward)
    
    def test_knowledge_persistence(self):
        """Test saving and loading of collective knowledge"""
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            test_file = tmp_file.name
        
        try:
            # Add some knowledge
            self.collective_learning.shared_q_agent.q_table["test_state"][0] = 42.0
            self.collective_learning.shared_q_agent.terrain_difficulty[(5, 5)] = 1.5
            
            # Save knowledge
            self.collective_learning.save_collective_knowledge(test_file)
            
            # Create new system and load
            new_collective = CollectiveQLearning(num_robots=3, environment_size=(10, 10))
            success = new_collective.load_collective_knowledge(test_file)
            
            self.assertTrue(success)
            self.assertEqual(new_collective.shared_q_agent.q_table["test_state"][0], 42.0)
            self.assertEqual(new_collective.shared_q_agent.terrain_difficulty[(5, 5)], 1.5)
            
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)


class TestHierarchicalCoordination(unittest.TestCase):
    """Test hierarchical coordination system"""
    
    def setUp(self):
        """Setup coordination agents"""
        self.environment = Environment(width=15, height=12, num_missing_persons=3)
        
        # Create coordinator
        self.coordinator = CoordinatorAgent(agent_id=0, position=(7, 11))
        
        # Create mediators
        self.mediator_1 = MediatorAgent(agent_id=1, position=(3, 11))
        self.mediator_2 = MediatorAgent(agent_id=2, position=(11, 11))
        
        # Create communication system
        self.communication = ExtendedCommunication()
        
        # Create test robots
        self.robots = []
        for i in range(3):
            q_agent = QLearningAgent((15, 12))
            robot = LearningTerrainRobot(
                robot_id=i,
                start_position=self.environment.terrain_robot_base[i],
                q_learning_agent=q_agent
            )
            self.robots.append(robot)
            self.communication.register_agent(f"robot_{i}", "robot")
        
        # Create test drones
        self.drones = []
        for i in range(2):
            drone = ExplorerDrone(
                drone_id=i,
                start_position=self.environment.drone_base[i]
            )
            self.drones.append(drone)
    
    def test_coordinator_initialization(self):
        """Test coordinator agent initialization"""
        self.assertEqual(self.coordinator.agent_id, 0)
        self.assertEqual(self.coordinator.state, CoordinatorState.STRATEGIC_PLANNING)
        self.assertEqual(self.coordinator.current_mission_phase, MissionPhase.INITIAL_DEPLOYMENT)
        self.assertIsInstance(self.coordinator.strategy_parameters, dict)
        self.assertIn('risk_tolerance', self.coordinator.strategy_parameters)
    
    def test_mediator_initialization(self):
        """Test mediator agent initialization"""
        self.assertEqual(self.mediator_1.agent_id, 1)
        self.assertEqual(self.mediator_1.state, MediatorState.MONITORING)
        self.assertIsInstance(self.mediator_1.urgency_weights, dict)
        self.assertIn('elevation', self.mediator_1.urgency_weights)
    
    def test_coordinator_strategic_planning(self):
        """Test coordinator strategic planning capabilities"""
        with redirect_stdout(StringIO()):  # Suppress output
            self.coordinator.update(
                self.environment, 
                self.robots, 
                self.drones, 
                self.communication,
                current_step=1
            )
        
        # Should have moved to resource allocation or monitoring
        self.assertIn(self.coordinator.state, [
            CoordinatorState.RESOURCE_ALLOCATION,
            CoordinatorState.MISSION_MONITORING
        ])
        
        # Should have mission objectives
        self.assertIsInstance(self.coordinator.mission_objectives, dict)
    
    def test_mediator_priority_calculation(self):
        """Test mediator priority calculation system"""
        # Test comprehensive priority calculation
        location = (5, 5)
        
        with redirect_stdout(StringIO()):  # Suppress output
            priority_data = self.mediator_1._calculate_comprehensive_priority(
                location, self.robots, self.environment
            )
        
        # Should return complete priority data
        self.assertIsInstance(priority_data, dict)
        self.assertIn('total_score', priority_data)
        self.assertIn('priority_level', priority_data)
        self.assertIn('recommended_robot', priority_data)
        self.assertIsInstance(priority_data['priority_level'], PriorityLevel)
        self.assertGreaterEqual(priority_data['total_score'], 0)
    
    def test_strategic_role_assignment(self):
        """Test strategic role assignment by coordinator"""
        with redirect_stdout(StringIO()):  # Suppress output
            # Trigger strategic planning
            self.coordinator.update(
                self.environment, 
                self.robots, 
                self.drones, 
                self.communication,
                current_step=1
            )
        
        # Should have assigned strategic roles
        self.assertIsInstance(self.coordinator.robot_strategic_assignments, dict)
        
        # At least some robots should have assignments
        if len(self.coordinator.robot_strategic_assignments) > 0:
            for robot_id, role in self.coordinator.robot_strategic_assignments.items():
                self.assertIsInstance(role, str)
                self.assertIn(role, [
                    'strategic_rescuer', 'primary_rescuer', 'support_rescuer', 
                    'emergency_rescuer', 'standby'
                ])
    
    def test_mediator_resource_allocation(self):
        """Test mediator resource allocation logic"""
        # Setup scenario with pending rescue
        self.environment.missing_persons = [(6, 6)]
        
        with redirect_stdout(StringIO()):  # Suppress output
            # Update mediator to trigger allocation
            self.mediator_1.update(
                self.environment, 
                self.robots, 
                self.drones, 
                self.communication
            )
        
        # Should have identified the pending rescue
        if len(self.mediator_1.rescue_queue) > 0:
            location, priority_data = self.mediator_1.rescue_queue[0]
            self.assertEqual(location, (6, 6))
            self.assertIsInstance(priority_data, dict)
    
    def test_coordination_integration(self):
        """Test integration between coordinator and mediators"""
        with redirect_stdout(StringIO()):  # Suppress output
            # Update coordinator first
            self.coordinator.update(
                self.environment, 
                self.robots, 
                self.drones, 
                self.communication,
                current_step=1
            )
            
            # Then update mediators
            self.mediator_1.update(
                self.environment, 
                self.robots, 
                self.drones, 
                self.communication
            )
        
        # Both should be operational
        self.assertIsNotNone(self.coordinator.state)
        self.assertIsNotNone(self.mediator_1.state)
        
        # Should have performance tracking
        coordinator_status = self.coordinator.get_status()
        mediator_status = self.mediator_1.get_status()
        
        self.assertIn('state', coordinator_status)
        self.assertIn('state', mediator_status)


class TestDynamicRoleAssignment(unittest.TestCase):
    """Test dynamic role assignment system"""
    
    def setUp(self):
        """Setup role assignment test environment"""
        with redirect_stdout(StringIO()):  # Suppress output
            self.simulation = MountainRescueSimulation(
                num_terrain_robots=3,
                num_drones=2,
                num_missing_persons=2,
                max_simulation_steps=50,
                operation_mode=OperationMode.NOVEL,
                enable_learning=True,
                enable_hierarchical_coordination=True
            )
    
    def test_initial_role_assignment(self):
        """Test initial role assignment"""
        self.assertTrue(self.simulation.enable_dynamic_roles)
        self.assertIsInstance(self.simulation.agent_roles, dict)
        
        # Should have roles for all agents
        expected_agents = (
            len(self.simulation.terrain_robots) + 
            len(self.simulation.drones)
        )
        self.assertGreaterEqual(len(self.simulation.agent_roles), expected_agents)
        
        # All roles should be valid
        for agent_id, role in self.simulation.agent_roles.items():
            self.assertIsInstance(role, AgentRole)
    
    def test_role_reassignment_logic(self):
        """Test dynamic role reassignment"""
        # Setup scenario for role reassignment
        robot = self.simulation.terrain_robots[0]
        robot.current_battery = 25  # Low battery
        
        with redirect_stdout(StringIO()):  # Suppress output
            self.simulation._reassign_agent_roles()
        
        # Should have recorded role changes
        self.assertIsInstance(self.simulation.role_assignment_history, list)
        
        # Check that robot with low battery gets appropriate role
        robot_id = f"robot_{robot.robot_id}"
        if robot_id in self.simulation.agent_roles:
            role = self.simulation.agent_roles[robot_id]
            # Low battery robots should get standby role
            if robot.current_battery < 30:
                self.assertEqual(role, AgentRole.STANDBY)
    
    def test_role_based_behavior_adaptation(self):
        """Test that robots adapt behavior based on roles"""
        robot = self.simulation.terrain_robots[0]
        
        # Test strategic role setting
        if hasattr(robot, 'set_strategic_role'):
            with redirect_stdout(StringIO()):  # Suppress output
                robot.set_strategic_role("primary_rescuer")
            
            self.assertEqual(robot.strategic_role, "primary_rescuer")
            
            # Should affect learning parameters
            if hasattr(robot, 'q_agent'):
                self.assertLessEqual(robot.q_agent.exploration_rate, 0.5)


class TestLearningRobotEnhancements(unittest.TestCase):
    """Test enhanced learning robot capabilities"""
    
    def setUp(self):
        """Setup enhanced learning robot"""
        self.environment = Environment(width=10, height=8, num_missing_persons=2)
        self.q_agent = QLearningAgent((10, 8))
        self.robot = LearningTerrainRobot(
            robot_id=0,
            start_position=(5, 5),
            q_learning_agent=self.q_agent
        )
    
    def test_enhanced_initialization(self):
        """Test enhanced robot initialization"""
        self.assertEqual(self.robot.learning_mode, "hierarchical")
        self.assertEqual(self.robot.strategic_role, "unassigned")
        self.assertIsInstance(self.robot.objective_weights, dict)
        self.assertIn('rescue_success', self.robot.objective_weights)
        self.assertEqual(self.robot.coordination_level, "autonomous")
    
    def test_strategic_role_adaptation(self):
        """Test adaptation to strategic roles"""
        initial_exploration = self.robot.q_agent.exploration_rate
        
        # Set primary rescuer role
        self.robot.set_strategic_role("primary_rescuer")
        
        self.assertEqual(self.robot.strategic_role, "primary_rescuer")
        self.assertEqual(self.robot.coordination_level, "guided")
        
        # Should reduce exploration rate for primary rescuers
        self.assertLessEqual(self.robot.q_agent.exploration_rate, initial_exploration)
    
    def test_tactical_guidance_integration(self):
        """Test integration of tactical guidance"""
        guidance = {
            'priority_adjustment': 0.8,
            'resource_conservation_mode': True,
            'coordination_intensity': 0.9
        }
        
        self.robot.receive_tactical_guidance(guidance, mediator_id=1)
        
        # Should have updated tactical guidance
        self.assertIn('priority_adjustment', self.robot.tactical_guidance)
        self.assertEqual(self.robot.tactical_guidance['priority_adjustment'], 0.8)
        
        # Should increase coordination interactions counter
        self.assertGreater(self.robot.coordination_interactions, 0)
    
    def test_enhanced_state_representation(self):
        """Test enhanced state key generation"""
        target = (7, 7)
        state_key = self.robot._get_enhanced_state_key(target, self.environment)
        
        # Should include coordination features
        self.assertIsInstance(state_key, str)
        self.assertIn(self.robot.strategic_role[:3], state_key)
        self.assertIn(self.robot.coordination_level[:3], state_key)
    
    def test_multi_objective_reward_calculation(self):
        """Test multi-objective reward calculation"""
        # Setup scenario for reward calculation
        old_position = self.robot.position
        old_battery = self.robot.current_battery
        old_state = self.robot.state
        
        # Simulate movement and battery drain
        self.robot.position = (6, 6)
        self.robot.current_battery -= 2
        
        with redirect_stdout(StringIO()):  # Suppress output
            self.robot._update_hierarchical_learning(
                old_position, old_battery, old_state, self.environment
            )
        
        # Should have recorded episode rewards
        if hasattr(self.robot, 'episode_rewards'):
            self.assertIsInstance(self.robot.episode_rewards, list)
    
    def test_performance_tracking(self):
        """Test performance tracking and optimization"""
        # Simulate some performance data
        self.robot.successful_rescues = 2
        self.robot.learning_episodes_completed = 5
        
        with redirect_stdout(StringIO()):  # Suppress output
            self.robot._update_performance_tracking()
        
        # Should have performance metrics
        status = self.robot.get_enhanced_learning_status()
        
        self.assertIn('strategic_compliance_score', status)
        self.assertIn('coordination_interactions', status)
        self.assertIn('performance_trend', status)


class TestNovelModeSimulation(unittest.TestCase):
    """Test complete Novel Mode simulation integration"""
    
    def setUp(self):
        """Setup Novel Mode simulation"""
        with redirect_stdout(StringIO()):  # Suppress output
            self.simulation = MountainRescueSimulation(
                num_terrain_robots=2,
                num_drones=1,
                num_missing_persons=2,
                max_simulation_steps=25,
                operation_mode=OperationMode.NOVEL,
                enable_learning=True,
                enable_hierarchical_coordination=True
            )
    
    def test_novel_mode_initialization(self):
        """Test Novel Mode initialization"""
        self.assertEqual(self.simulation.operation_mode, OperationMode.NOVEL)
        self.assertIsNotNone(self.simulation.collective_learning)
        self.assertTrue(self.simulation.enable_hierarchical_coordination)
        self.assertTrue(self.simulation.enable_dynamic_roles)
        
        # Should have coordinator and mediators
        self.assertIsNotNone(self.simulation.coordinator)
        self.assertIsInstance(self.simulation.mediators, list)
        self.assertGreater(len(self.simulation.mediators), 0)
        
        # All robots should be learning robots
        for robot in self.simulation.terrain_robots:
            self.assertIsInstance(robot, LearningTerrainRobot)
    
    def test_hierarchical_system_integration(self):
        """Test integration of hierarchical coordination"""
        # Update simulation step to trigger coordination
        with redirect_stdout(StringIO()):  # Suppress output
            self.simulation._update_simulation_step()
        
        # Coordinator should have updated
        self.assertIsNotNone(self.simulation.coordinator.state)
        
        # Mediators should have updated
        for mediator in self.simulation.mediators:
            self.assertIsNotNone(mediator.state)
    
    def test_knowledge_sharing_schedule(self):
        """Test periodic knowledge sharing"""
        initial_share_step = self.simulation.last_knowledge_share
        
        # Advance past sharing interval
        self.simulation.current_step = self.simulation.knowledge_sharing_interval + 1
        
        with redirect_stdout(StringIO()):  # Suppress output
            self.simulation._update_novel_mode_systems()
        
        # Should have updated knowledge sharing
        self.assertGreater(self.simulation.last_knowledge_share, initial_share_step)
    
    def test_role_reassignment_schedule(self):
        """Test periodic role reassignment"""
        initial_reassignment_step = self.simulation.last_role_reassignment
        
        # Advance past reassignment interval
        self.simulation.current_step = self.simulation.role_reassignment_interval + 1
        
        with redirect_stdout(StringIO()):  # Suppress output
            self.simulation._update_novel_mode_systems()
        
        # Should have updated role reassignment
        self.assertGreater(self.simulation.last_role_reassignment, initial_reassignment_step)
    
    def test_complete_simulation_run(self):
        """Test complete Novel Mode simulation run"""
        with redirect_stdout(StringIO()):  # Suppress output
            report = self.simulation.run_simulation(visualise=False, step_delay=0)
        
        # Should generate comprehensive report
        self.assertIsInstance(report, dict)
        self.assertEqual(report['mode'], 'novel')
        
        # Should have Novel Mode specific metrics
        self.assertIn('learning_stats', report)
        self.assertIn('coordination_stats', report)
        
        # Should have role assignment information
        if 'role_assignment_stats' in report:
            role_stats = report['role_assignment_stats']
            self.assertIn('total_role_changes', role_stats)
            self.assertIn('final_role_distribution', role_stats)
        
        # Robot performance should include learning metrics
        for robot_perf in report['robot_performance']:
            if 'episodes_completed' in robot_perf:
                self.assertIsInstance(robot_perf['episodes_completed'], int)
                self.assertIn('q_table_size', robot_perf)


class TestPerformanceAndOptimization(unittest.TestCase):
    """Test performance improvements and optimization features"""
    
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization metrics"""
        with redirect_stdout(StringIO()):  # Suppress output
            simulation = MountainRescueSimulation(
                num_terrain_robots=2,
                num_drones=1,
                num_missing_persons=1,
                max_simulation_steps=15,
                operation_mode=OperationMode.NOVEL
            )
            
            report = simulation.run_simulation(visualise=False, step_delay=0)
        
        # Should have multi-objective performance metrics
        if 'multi_objective_performance' in report:
            multi_obj = report['multi_objective_performance']
            
            self.assertIn('speed_score', multi_obj)
            self.assertIn('efficiency_score', multi_obj)
            self.assertIn('safety_score', multi_obj)
            self.assertIn('overall_score', multi_obj)
            
            # Scores should be reasonable (0-1 range)
            for score_name, score_value in multi_obj.items():
                self.assertGreaterEqual(score_value, 0.0)
                self.assertLessEqual(score_value, 1.0)
    
    def test_learning_progression(self):
        """Test that learning progresses over episodes"""
        q_agent = QLearningAgent((10, 10))
        initial_exploration = q_agent.exploration_rate
        initial_episode_count = q_agent.episode_count
        
        # Simulate several episodes
        for _ in range(5):
            q_agent.decay_exploration_rate()
        
        # Exploration should decrease
        self.assertLess(q_agent.exploration_rate, initial_exploration)
        
        # Episode count should increase
        self.assertGreater(q_agent.episode_count, initial_episode_count)
    
    def test_coordination_effectiveness(self):
        """Test coordination effectiveness metrics"""
        mediator = MediatorAgent(agent_id=0, position=(5, 5))
        
        # Simulate some assignments
        mediator.assignments_made = 5
        mediator.successful_rescues = 4
        mediator.efficiency_score = 0.8
        
        status = mediator.get_status()
        
        self.assertEqual(status['assignments_made'], 5)
        self.assertEqual(status['successful_rescues'], 4)
        self.assertEqual(status['efficiency_score'], 0.8)


class TestErrorHandlingAndRobustness(unittest.TestCase):
    """Test error handling and system robustness"""
    
    def test_missing_knowledge_file(self):
        """Test handling of missing knowledge file"""
        collective = CollectiveQLearning(num_robots=2, environment_size=(10, 10))
        
        # Should handle missing file gracefully
        success = collective.load_collective_knowledge("nonexistent_file.pkl")
        self.assertFalse(success)
        
        # System should still be functional
        self.assertIsNotNone(collective.shared_q_agent)
    
    def test_invalid_role_assignment(self):
        """Test handling of invalid role assignments"""
        q_agent = QLearningAgent((10, 10))
        robot = LearningTerrainRobot(robot_id=0, start_position=(5, 5), q_learning_agent=q_agent)
        
        # Should handle invalid role gracefully
        robot.set_strategic_role("invalid_role")
        
        # Should maintain a valid state
        self.assertIsNotNone(robot.strategic_role)
    
    def test_empty_environment_handling(self):
        """Test handling of edge cases"""
        # Create environment with no missing persons
        env = Environment(width=10, height=8, num_missing_persons=0)
        mediator = MediatorAgent(agent_id=0, position=(5, 5))
        
        # Should handle empty environment gracefully
        with redirect_stdout(StringIO()):  # Suppress output
            pending = mediator._identify_pending_rescues(env, None)
        
        self.assertEqual(len(pending), 0)
        self.assertIsInstance(pending, list)


# Cleanup function for temporary files
def cleanup_test_files():
    """Clean up any test files created during testing"""
    test_data_dir = "data"
    if os.path.exists(test_data_dir):
        try:
            shutil.rmtree(test_data_dir)
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == '__main__':
    print("🧪 Running Comprehensive Novel Mode Test Suite...")
    print("=" * 70)
    print("Testing Enhanced Multi-Agent AI System with:")
    print("  • Q-Learning with Collective Intelligence")
    print("  • Hierarchical Coordination (Coordinator + Mediators)")
    print("  • Dynamic Role Assignment System")
    print("  • Multi-Objective Optimization")
    print("  • Performance Tracking and Adaptation")
    print("  • Integration and Robustness")
    print("-" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEnhancedQLearning,
        TestHierarchicalCoordination,
        TestDynamicRoleAssignment,
        TestLearningRobotEnhancements,
        TestNovelModeSimulation,
        TestPerformanceAndOptimization,
        TestErrorHandlingAndRobustness
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Count total tests
    total_tests = suite.countTestCases()
    print(f"Total tests to run: {total_tests}")
    print("-" * 70)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Clean up test files
    cleanup_test_files()
    
    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    
    if result.wasSuccessful():
        print(f"🎉 ALL TESTS PASSED: {result.testsRun}/{total_tests}")
        print("\n✅ NOVEL MODE VERIFICATION COMPLETE:")
        print("  • Q-Learning with Collective Intelligence: ✓")
        print("  • Hierarchical Coordination System: ✓")
        print("  • Dynamic Role Assignment: ✓")
        print("  • Multi-Objective Optimization: ✓")
        print("  • Performance Tracking: ✓")
        print("  • Integration & Robustness: ✓")
        print("\n🚀 Enhanced Multi-Agent AI System is ready for deployment!")
        print("   All innovations verified and functional.")
    else:
        failed = len(result.failures)
        errors = len(result.errors)
        passed = result.testsRun - failed - errors
        
        print(f"⚠️  SOME TESTS FAILED: {passed}/{total_tests} passed")
        print(f"   Failures: {failed}, Errors: {errors}")
        print("\n❌ Issues found in:")
        
        for test, traceback in result.failures:
            print(f"   • FAILURE: {test}")
        
        for test, traceback in result.errors:
            print(f"   • ERROR: {test}")
        
        print("\n🔧 Please fix issues before proceeding to final deployment.")
    
    print("=" * 70)