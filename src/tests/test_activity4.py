# Unit tests for Activity 4 - Novel Mode with Q-Learning

import unittest
import sys
import os
import numpy as np
import shutil

# Add src directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'activity4'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'activity1'))

from activity4.q_learning import QLearningAgent, CollectiveQLearning
from activity4.learning_robot import LearningTerrainRobot
from activity4.simulation import MountainRescueSimulation, OperationMode
from activity1.environment import Environment
from activity1.terrain_robot import RobotState


class TestQLearning(unittest.TestCase):
    """Test Q-Learning implementation"""
    
    def setUp(self):
        """Setup test Q-learning agent"""
        self.q_agent = QLearningAgent(
            state_space_size=(10, 10),
            action_space_size=8,
            learning_rate=0.1,
            discount_factor=0.95
        )
    
    def test_initialization(self):
        """Test Q-learning agent initialization"""
        self.assertEqual(self.q_agent.learning_rate, 0.1)
        self.assertEqual(self.q_agent.discount_factor, 0.95)
        self.assertEqual(self.q_agent.exploration_rate, 1.0)
        self.assertEqual(self.q_agent.action_space_size, 8)
        self.assertEqual(len(self.q_agent.q_table), 0)  # Empty initially
    
    def test_state_key_generation(self):
        """Test state key creation for Q-table"""
        # Test with target
        state_key = self.q_agent.get_state_key(
            position=(5, 5),
            target=(7, 7),
            battery_level=80,
            has_kit=True
        )
        self.assertEqual(state_key, "5,5|1,1|high|kit")
        
        # Test without target (exploring)
        state_key = self.q_agent.get_state_key(
            position=(3, 3),
            target=None,
            battery_level=25,
            has_kit=False
        )
        self.assertEqual(state_key, "3,3|exploring|low|no_kit")
    
    def test_action_selection(self):
        """Test epsilon-greedy action selection"""
        state_key = "5,5|exploring|high|kit"
        valid_actions = [0, 1, 2, 3, 4]
        
        # Test exploration (high epsilon)
        self.q_agent.exploration_rate = 1.0
        actions_chosen = []
        for _ in range(100):
            action = self.q_agent.choose_action(state_key, valid_actions, (5, 5))
            actions_chosen.append(action)
        
        # Should explore randomly
        unique_actions = set(actions_chosen)
        self.assertTrue(len(unique_actions) > 1)  # Multiple different actions
        
        # Test exploitation (low epsilon)
        self.q_agent.exploration_rate = 0.0
        # Set one action to have high Q-value
        self.q_agent.q_table[state_key][2] = 10.0
        
        action = self.q_agent.choose_action(state_key, valid_actions, (5, 5))
        self.assertEqual(action, 2)  # Should choose highest Q-value
    
    def test_q_value_update(self):
        """Test Q-learning update rule"""
        state_key = "5,5|1,1|high|kit"
        next_state_key = "6,6|1,1|high|kit"
        action = 2
        reward = 10.0
        valid_next_actions = [0, 1, 2, 3]
        
        # Initial Q-value should be near 0
        initial_q = self.q_agent.q_table[state_key][action]
        self.assertTrue(-0.01 <= initial_q <= 0.01)
        
        # Update Q-value
        self.q_agent.update_q_value(
            state_key, action, reward, next_state_key, valid_next_actions
        )
        
        # Q-value should increase after positive reward
        updated_q = self.q_agent.q_table[state_key][action]
        self.assertGreater(updated_q, initial_q)
        self.assertAlmostEqual(updated_q, 1.0, places=2)  # α * r = 0.1 * 10
    
    def test_reward_calculation(self):
        """Test reward calculation logic"""
        # Test rescue reward
        reward = self.q_agent.calculate_reward(
            position=(5, 5),
            next_position=(5, 5),
            found_person=False,
            rescued_person=True,
            battery_used=2,
            terrain_elevation=1000,
            is_mountain_area=True
        )
        self.assertGreater(reward, 90)  # Should be around 100 - penalties
        
        # Test movement reward
        reward = self.q_agent.calculate_reward(
            position=(5, 5),
            next_position=(5, 6),
            found_person=False,
            rescued_person=False,
            battery_used=1,
            terrain_elevation=0,
            is_mountain_area=True
        )
        self.assertGreater(reward, 0)  # Positive for movement + mountain area
    
    def test_exploration_decay(self):
        """Test exploration rate decay"""
        initial_rate = self.q_agent.exploration_rate
        self.q_agent.decay_exploration_rate()
        
        self.assertLess(self.q_agent.exploration_rate, initial_rate)
        self.assertEqual(self.q_agent.episode_count, 1)
        
        # Test minimum exploration rate
        for _ in range(1000):
            self.q_agent.decay_exploration_rate()
        
        self.assertGreaterEqual(self.q_agent.exploration_rate, 
                               self.q_agent.min_exploration_rate)
    
    def test_knowledge_persistence(self):
        """Test saving and loading knowledge"""
        # Add some knowledge
        self.q_agent.q_table["test_state"][0] = 5.0
        self.q_agent.terrain_difficulty[(5, 5)] = 2.5
        self.q_agent.rescue_success_map[(7, 7)] = 3
        
        # Save knowledge
        test_file = "test_knowledge.pkl"
        self.q_agent.save_knowledge(test_file)
        
        # Create new agent and load
        new_agent = QLearningAgent((10, 10))
        success = new_agent.load_knowledge(test_file)
        
        self.assertTrue(success)
        self.assertEqual(new_agent.q_table["test_state"][0], 5.0)
        self.assertEqual(new_agent.terrain_difficulty[(5, 5)], 2.5)
        self.assertEqual(new_agent.rescue_success_map[(7, 7)], 3)
        
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists("data"):
            shutil.rmtree("data")


class TestCollectiveQLearning(unittest.TestCase):
    """Test collective Q-learning system"""
    
    def setUp(self):
        """Setup collective learning system"""
        self.collective = CollectiveQLearning(
            num_robots=3,
            environment_size=(10, 10)
        )
    
    def test_initialization(self):
        """Test collective system setup"""
        self.assertEqual(self.collective.num_robots, 3)
        self.assertEqual(len(self.collective.robot_agents), 3)
        self.assertIsNotNone(self.collective.shared_q_agent)
    
    def test_knowledge_sharing(self):
        """Test knowledge sharing between robots"""
        # Give different robots different knowledge
        self.collective.robot_agents[0].q_table["state1"][0] = 10.0
        self.collective.robot_agents[1].q_table["state1"][0] = 20.0
        self.collective.robot_agents[0].terrain_difficulty[(5, 5)] = 3.0
        self.collective.robot_agents[1].terrain_difficulty[(5, 5)] = 5.0
        
        # Share knowledge
        self.collective.share_knowledge()
        
        # Check shared knowledge is averaged
        shared_q = self.collective.shared_q_agent.q_table["state1"][0]
        self.assertTrue(10.0 < shared_q < 20.0)  # Should be weighted average
        
        shared_terrain = self.collective.shared_q_agent.terrain_difficulty[(5, 5)]
        self.assertEqual(shared_terrain, 4.0)  # Average of 3.0 and 5.0
    
    def test_learning_statistics(self):
        """Test statistics collection"""
        # Add some data
        self.collective.shared_q_agent.q_table["state1"] = np.zeros(8)
        self.collective.shared_q_agent.terrain_difficulty[(1, 1)] = 1.0
        self.collective.shared_q_agent.rescue_success_map[(2, 2)] = 2
        
        stats = self.collective.get_learning_statistics()
        
        self.assertEqual(stats['total_states_explored'], 1)
        self.assertEqual(stats['total_terrain_mapped'], 1)
        self.assertEqual(stats['total_rescue_locations'], 1)
        self.assertIn('average_exploration_rate', stats)


class TestLearningRobot(unittest.TestCase):
    """Test learning terrain robot"""
    
    def setUp(self):
        """Setup test learning robot"""
        self.env = Environment(width=10, height=10, num_missing_persons=2)
        self.q_agent = QLearningAgent((10, 10))
        self.robot = LearningTerrainRobot(
            robot_id=0,
            start_position=(5, 5),
            q_learning_agent=self.q_agent,
            max_battery=100
        )
    
    def test_initialization(self):
        """Test learning robot initialization"""
        self.assertEqual(self.robot.robot_id, 0)
        self.assertTrue(self.robot.learning_enabled)
        self.assertIsNotNone(self.robot.q_agent)
        self.assertEqual(self.robot.learning_episodes_completed, 0)
    
    def test_valid_actions(self):
        """Test getting valid movement actions"""
        valid_actions = self.robot._get_valid_actions(self.env)
        
        # Should have 8 possible directions (or fewer at boundaries)
        self.assertGreater(len(valid_actions), 0)
        self.assertLessEqual(len(valid_actions), 8)
        
        # All actions should be in range 0-7
        for action in valid_actions:
            self.assertTrue(0 <= action <= 7)
    
    def test_learning_status(self):
        """Test learning status reporting"""
        status = self.robot.get_learning_status()
        
        self.assertIn('learning_enabled', status)
        self.assertIn('exploration_rate', status)
        self.assertIn('episodes_completed', status)
        self.assertIn('q_table_size', status)
        self.assertTrue(status['learning_enabled'])
    
    def test_position_from_action(self):
        """Test action to position conversion"""
        # Test each direction
        self.robot.position = (5, 5)
        
        # North (action 0)
        pos = self.robot._get_position_from_action(0)
        self.assertEqual(pos, (5, 4))
        
        # East (action 2)
        pos = self.robot._get_position_from_action(2)
        self.assertEqual(pos, (6, 5))
        
        # Southeast (action 3)
        pos = self.robot._get_position_from_action(3)
        self.assertEqual(pos, (6, 6))
    
    def test_learning_episode_completion(self):
        """Test learning episode tracking"""
        # Simulate some rewards
        self.robot.episode_rewards = [1.0, 2.0, 100.0, -1.0]
        initial_episodes = self.robot.learning_episodes_completed
        initial_exploration = self.robot.q_agent.exploration_rate
        
        # Complete episode
        self.robot._complete_learning_episode()
        
        # Check episode count increased
        self.assertEqual(self.robot.learning_episodes_completed, initial_episodes + 1)
        
        # Check exploration rate decreased
        self.assertLess(self.robot.q_agent.exploration_rate, initial_exploration)
        
        # Check rewards cleared
        self.assertEqual(len(self.robot.episode_rewards), 0)


class TestNovelModeSimulation(unittest.TestCase):
    """Test Novel Mode simulation integration"""
    
    def setUp(self):
        """Setup test simulation"""
        self.simulation = MountainRescueSimulation(
            num_terrain_robots=2,
            num_drones=1,
            num_missing_persons=2,
            max_simulation_steps=50,
            operation_mode=OperationMode.NOVEL,
            enable_learning=True
        )
    
    def test_novel_mode_initialization(self):
        """Test Novel Mode setup"""
        self.assertEqual(self.simulation.operation_mode, OperationMode.NOVEL)
        self.assertIsNotNone(self.simulation.collective_learning)
        
        # Check all robots are learning robots
        for robot in self.simulation.terrain_robots:
            self.assertIsInstance(robot, LearningTerrainRobot)
            self.assertTrue(robot.learning_enabled)
    
    def test_knowledge_sharing_schedule(self):
        """Test periodic knowledge sharing"""
        initial_share_step = self.simulation.last_knowledge_share
        
        # Advance simulation past sharing interval
        self.simulation.current_step = self.simulation.knowledge_sharing_interval + 1
        self.simulation._update_simulation_step()
        
        # Check knowledge was shared
        self.assertGreater(self.simulation.last_knowledge_share, initial_share_step)
    
    def test_learning_statistics_tracking(self):
        """Test that learning stats are tracked"""
        # Run a few steps
        for _ in range(5):
            self.simulation._update_simulation_step()
            self.simulation.current_step += 1
        
        # Update statistics
        self.simulation._update_statistics()
        
        # Check learning stats exist
        self.assertIn('learning_stats', self.simulation.stats)
        learning_stats = self.simulation.stats['learning_stats']
        
        self.assertIn('total_states_explored', learning_stats)
        self.assertIn('total_terrain_mapped', learning_stats)
        self.assertIn('average_exploration_rate', learning_stats)
    
    def test_short_simulation_run(self):
        """Test running a short Novel Mode simulation"""
        report = self.simulation.run_simulation(visualise=False, step_delay=0)
        
        # Check report structure
        self.assertEqual(report['mode'], 'novel')
        self.assertIn('learning_stats', report)
        self.assertIn('robot_performance', report)
        
        # Check robot performance includes learning metrics
        for robot_perf in report['robot_performance']:
            self.assertIn('episodes_completed', robot_perf)
            self.assertIn('q_table_size', robot_perf)
            self.assertIn('exploration_rate', robot_perf)
    
    def tearDown(self):
        """Cleanup after tests"""
        # Remove any created knowledge files
        if os.path.exists("data/collective_knowledge.pkl"):
            os.remove("data/collective_knowledge.pkl")
        if os.path.exists("data"):
            shutil.rmtree("data", ignore_errors=True)


if __name__ == '__main__':
    print("Running Novel Mode Tests for Activity 4...")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    # Count tests
    total_tests = suite.countTestCases()
    print(f"Total tests to run: {total_tests}")
    print("-" * 50)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print(f" ALL TESTS PASSED: {result.testsRun}/{total_tests}")
        print("Novel Mode Q-Learning implementation verified!")
    else:
        print(f" SOME TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        print("Please fix issues before proceeding.")
    print("=" * 50)