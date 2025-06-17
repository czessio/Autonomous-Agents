# Unit tests for Activity 3 - Extended Mode functionality

import unittest
import sys
import os
import io
from contextlib import redirect_stdout

# Add src directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'activity3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'activity1'))

from activity3.communication import BasicCommunication, ExtendedCommunication, MessageType, Message
from activity3.simulation import MountainRescueSimulation, OperationMode
from activity1.environment import Environment
from activity1.terrain_robot import TerrainRobot, RobotState
from activity1.explorer_drone import ExplorerDrone, DroneState



class TestExtendedCommunication(unittest.TestCase):
    """Test the Extended Communication system"""
    
    def setUp(self):
        """Setup test communication system"""
        self.comm = ExtendedCommunication()
        # Register test agents
        self.comm.register_agent("robot_0", "robot")
        self.comm.register_agent("robot_1", "robot")
        self.comm.register_agent("drone_0", "drone")
        self.comm.register_agent("drone_1", "drone")
    
    def test_agent_registration(self):
        """Test agent registration in communication system"""
        # Check agents are registered
        self.assertIn("robot_0", self.comm.message_queues)
        self.assertIn("drone_0", self.comm.message_queues)
        
        # Check robot status tracking
        self.assertIn("robot_0", self.comm.robot_status)
        self.assertTrue(self.comm.robot_status["robot_0"]['available'])
        self.assertEqual(self.comm.robot_status["robot_0"]['battery'], 100)
    
    def test_individual_messaging(self):
        """Test individual message sending"""
        # Send individual message
        self.comm.send_individual_message(
            sender_id="drone_0",
            recipient_id="robot_0",
            msg_type=MessageType.RESCUE_REQUEST,
            location=(5, 5),
            urgency=8
        )
        
        # Check message received
        messages = self.comm.get_messages("robot_0")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].msg_type, MessageType.RESCUE_REQUEST)
        self.assertEqual(messages[0].location, (5, 5))
        self.assertEqual(messages[0].urgency, 8)
        
        # Check metrics
        self.assertEqual(self.comm.total_messages_sent, 1)
        self.assertEqual(self.comm.individual_messages, 1)
    
    def test_broadcast_messaging(self):
        """Test broadcast message functionality"""
        # Send broadcast
        self.comm.broadcast_message(
            sender_id="drone_0",
            msg_type=MessageType.PERSON_FOUND,
            location=(3, 3),
            urgency=5
        )
        
        # Check all agents except sender received message
        for agent_id in ["robot_0", "robot_1", "drone_1"]:
            messages = self.comm.get_messages(agent_id)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].msg_type, MessageType.PERSON_FOUND)
        
        # Sender should not receive own broadcast
        sender_messages = self.comm.get_messages("drone_0")
        self.assertEqual(len(sender_messages), 0)
        
        # Check metrics
        self.assertEqual(self.comm.broadcast_messages_count, 1)
        self.assertEqual(self.comm.total_messages_sent, 3)  # 3 recipients
    
    def test_rescue_assignment(self):
        """Test robot assignment to rescue locations"""
        location = (4, 4)
        
        # Drone finds person
        self.comm.drone_found_person("drone_0", location, urgency=7, broadcast=False)
        
        # Assign robot to rescue
        success = self.comm.assign_robot_to_rescue("robot_0", location)
        self.assertTrue(success)
        
        # Check assignment tracking
        self.assertIn("robot_0", self.comm.assigned_rescues)
        self.assertEqual(self.comm.assigned_rescues["robot_0"], location)
        self.assertFalse(self.comm.robot_status["robot_0"]['available'])
        
        # Try to assign another robot to same location
        success2 = self.comm.assign_robot_to_rescue("robot_1", location)
        self.assertFalse(success2)  # Should fail - already assigned
    
    def test_rescue_completion(self):
        """Test rescue completion workflow"""
        location = (6, 6)
        
        # Setup rescue scenario
        self.comm.drone_found_person("drone_0", location, urgency=9)
        self.comm.assign_robot_to_rescue("robot_0", location)
        
        # Complete rescue
        self.comm.complete_rescue("robot_0", location)
        
        # Check cleanup
        self.assertNotIn("robot_0", self.comm.assigned_rescues)
        self.assertNotIn(location, self.comm.found_persons)
        self.assertTrue(self.comm.robot_status["robot_0"]['available'])
        self.assertEqual(len(self.comm.completed_rescues), 1)
    
    def test_communication_stats(self):
        """Test communication statistics tracking"""
        # Generate some communication activity
        self.comm.drone_found_person("drone_0", (2, 2), urgency=5)
        self.comm.drone_found_person("drone_1", (3, 3), urgency=7)
        self.comm.assign_robot_to_rescue("robot_0", (2, 2))
        self.comm.complete_rescue("robot_0", (2, 2))
        
        # Get stats
        stats = self.comm.get_communication_stats()
        
        self.assertGreater(stats['total_messages'], 0)
        self.assertEqual(stats['completed_rescues'], 1)
        self.assertEqual(stats['pending_rescues'], 1)  # One location still pending
        self.assertIn('message_efficiency', stats)


class TestOperationModes(unittest.TestCase):
    """Test operation mode functionality"""
    
    def test_mode_initialization(self):
        """Test simulation starts in correct mode"""
        # Basic mode
        with redirect_stdout(io.StringIO()):
            basic_sim = MountainRescueSimulation(
                num_terrain_robots=2,
                num_drones=1,
                num_missing_persons=3,
                operation_mode=OperationMode.BASIC
            )
        
        self.assertEqual(basic_sim.operation_mode, OperationMode.BASIC)
        self.assertIsInstance(basic_sim.communication, BasicCommunication)
        self.assertFalse(basic_sim.spawn_new_persons)
        
        # Extended mode
        with redirect_stdout(io.StringIO()):
            extended_sim = MountainRescueSimulation(
                num_terrain_robots=2,
                num_drones=1,
                num_missing_persons=3,
                operation_mode=OperationMode.EXTENDED
            )
        
        self.assertEqual(extended_sim.operation_mode, OperationMode.EXTENDED)
        self.assertIsInstance(extended_sim.communication, ExtendedCommunication)
        self.assertTrue(extended_sim.spawn_new_persons)
    
    def test_mode_switching(self):
        """Test switching between operation modes"""
        with redirect_stdout(io.StringIO()):
            sim = MountainRescueSimulation(
                operation_mode=OperationMode.BASIC
            )
            
            # Switch to Extended
            sim.switch_operation_mode(OperationMode.EXTENDED)
            
            self.assertEqual(sim.operation_mode, OperationMode.EXTENDED)
            self.assertIsInstance(sim.communication, ExtendedCommunication)
            self.assertTrue(sim.spawn_new_persons)
            
            # Switch back to Basic
            sim.switch_operation_mode(OperationMode.BASIC)
            
            self.assertEqual(sim.operation_mode, OperationMode.BASIC)
            self.assertIsInstance(sim.communication, BasicCommunication)
            self.assertFalse(sim.spawn_new_persons)
    
    def test_robot_behavior_extended_mode(self):
        """Test robots wait at base in Extended Mode"""
        with redirect_stdout(io.StringIO()):
            sim = MountainRescueSimulation(
                num_terrain_robots=2,
                num_drones=1,
                operation_mode=OperationMode.EXTENDED
            )
        
        # In Extended Mode, robots should start at base
        for robot in sim.terrain_robots:
            self.assertEqual(robot.state, RobotState.AT_BASE)
            self.assertEqual(robot.min_leave_threshold, 50)  # Lower threshold
        
        # Drones should still start exploring
        for drone in sim.drones:
            self.assertEqual(drone.state, DroneState.EXPLORING)


class TestDynamicPersonSpawning(unittest.TestCase):
    """Test dynamic person spawning in Extended Mode"""
    
    def test_person_spawning(self):
        """Test new persons spawn at intervals"""
        with redirect_stdout(io.StringIO()):
            sim = MountainRescueSimulation(
                num_missing_persons=2,
                operation_mode=OperationMode.EXTENDED,
                spawn_new_persons=True,
                spawn_interval=10,
                max_simulation_steps=50
            )
        
        initial_persons = len(sim.environment.missing_persons)
        initial_spawned = sim.total_persons_spawned
        
        # Run simulation for enough steps to trigger spawning
        sim.current_step = 15
        sim._spawn_new_person()
        
        # Check new person was spawned
        self.assertEqual(len(sim.environment.missing_persons), initial_persons + 1)
        self.assertEqual(sim.total_persons_spawned, initial_spawned + 1)
        
        # Check spawn location is in mountain area
        new_person = sim.environment.missing_persons[-1]
        self.assertTrue(sim.environment.is_mountain_area(*new_person))
    
    def test_spawn_tracking(self):
        """Test spawn time tracking for KPI analysis"""
        with redirect_stdout(io.StringIO()):
            sim = MountainRescueSimulation(
                num_missing_persons=1,
                operation_mode=OperationMode.EXTENDED
            )
        
        # Check initial persons have spawn time 0
        for person in sim.environment.missing_persons:
            self.assertIn(person, sim.stats['person_spawn_times'])
            self.assertEqual(sim.stats['person_spawn_times'][person], 0)
        
        # Spawn new person at step 20
        sim.current_step = 20
        sim._spawn_new_person()
        
        # Find the new person
        for person in sim.environment.missing_persons:
            if person not in [(p['location'] for p in sim.stats['rescue_times'])]:
                if sim.stats['person_spawn_times'].get(person, 0) == 20:
                    # Found the new person
                    self.assertEqual(sim.stats['person_spawn_times'][person], 20)
                    return
        
        self.fail("New person spawn time not tracked correctly")


class TestMessageProcessing(unittest.TestCase):
    """Test message processing by agents in Extended Mode"""
    
    def setUp(self):
        """Setup test environment"""
        with redirect_stdout(io.StringIO()):
            self.sim = MountainRescueSimulation(
                num_terrain_robots=2,
                num_drones=1,
                operation_mode=OperationMode.EXTENDED
            )
    
    def test_robot_message_handling(self):
        """Test robots respond to rescue messages"""
        robot = self.sim.terrain_robots[0]
        robot_id = f"robot_{robot.robot_id}"
        
        # Ensure robot is at base with kit and battery
        robot.state = RobotState.AT_BASE
        robot.has_first_aid_kit = True
        robot.current_battery = 80
        
        # Send rescue request
        self.sim.communication.send_individual_message(
            sender_id="drone_0",
            recipient_id=robot_id,
            msg_type=MessageType.PERSON_FOUND,
            location=(5, 5),
            urgency=8
        )
        
        # Process messages
        with redirect_stdout(io.StringIO()):
            self.sim._handle_extended_mode_robot(robot)
        
        # Robot should have assigned location
        self.assertEqual(robot.assigned_location, (5, 5))
        self.assertEqual(robot.state, RobotState.SEARCHING)
    
    def test_broadcast_response(self):
        """Test multiple robots respond to broadcast"""
        # Setup robots at base
        for robot in self.sim.terrain_robots:
            robot.state = RobotState.AT_BASE
            robot.has_first_aid_kit = True
            robot.current_battery = 90
        
        # Broadcast person found
        self.sim.communication.broadcast_message(
            sender_id="drone_0",
            msg_type=MessageType.PERSON_FOUND,
            location=(4, 4),
            urgency=6
        )
        
        # Process messages for first robot
        with redirect_stdout(io.StringIO()):
            self.sim._handle_extended_mode_robot(self.sim.terrain_robots[0])
        
        # First robot should be assigned
        self.assertEqual(self.sim.terrain_robots[0].state, RobotState.SEARCHING)
        
        # Process messages for second robot
        with redirect_stdout(io.StringIO()):
            self.sim._handle_extended_mode_robot(self.sim.terrain_robots[1])
        
        # Second robot should still be at base (location already assigned)
        self.assertEqual(self.sim.terrain_robots[1].state, RobotState.AT_BASE)


class TestPerformanceMetrics(unittest.TestCase):
    """Test KPI tracking and statistics"""
    
    def test_rescue_time_tracking(self):
        """Test rescue time calculation"""
        with redirect_stdout(io.StringIO()):
            sim = MountainRescueSimulation(
                num_missing_persons=2,
                operation_mode=OperationMode.EXTENDED
            )
        
        # Setup scenario
        person_loc = sim.environment.missing_persons[0]
        sim.stats['person_spawn_times'][person_loc] = 10
        sim.current_step = 50
        
        # Simulate rescue
        sim.environment.rescue_person(*person_loc)
        sim._update_rescue_statistics()
        
        # Check rescue time recorded
        self.assertEqual(len(sim.stats['rescue_times']), 1)
        rescue_record = sim.stats['rescue_times'][0]
        self.assertEqual(rescue_record['location'], person_loc)
        self.assertEqual(rescue_record['spawn_step'], 10)
        self.assertEqual(rescue_record['rescue_step'], 50)
        self.assertEqual(rescue_record['rescue_time'], 40)
    
    def test_battery_efficiency_calculation(self):
        """Test battery efficiency metrics"""
        with redirect_stdout(io.StringIO()):
            sim = MountainRescueSimulation(operation_mode=OperationMode.EXTENDED)
            
            # Simulate battery usage and rescues
            for robot in sim.terrain_robots:
                robot.current_battery = 70  # Used 50 battery
            
            sim.stats['total_persons_rescued'] = 3
            sim._update_statistics()
            
            # Calculate expected efficiency
            total_battery_used = len(sim.terrain_robots) * 50
            expected_efficiency = 3 / total_battery_used
            
            # Generate report
            report = sim._generate_final_report()
            
            self.assertAlmostEqual(report['battery_efficiency'], expected_efficiency, places=4)


class TestIntegrationExtended(unittest.TestCase):
    """Integration tests for Extended Mode"""
    
    def test_full_extended_mode_workflow(self):
        """Test complete Extended Mode rescue workflow"""
        with redirect_stdout(io.StringIO()):
            sim = MountainRescueSimulation(
                num_terrain_robots=1,
                num_drones=1,
                num_missing_persons=1,
                max_simulation_steps=30,
                operation_mode=OperationMode.EXTENDED
            )
            
            # Run short simulation
            report = sim.run_simulation(visualise=False, step_delay=0)
        
        # Verify Extended Mode features
        self.assertEqual(report['mode'], 'extended')
        self.assertIn('communication_stats', report)
        self.assertIn('avg_rescue_time', report)
        self.assertIn('battery_efficiency', report)
        
        # Should have some communication activity
        if report['communication_stats']:
            self.assertGreater(report['communication_stats']['total_messages'], 0)


if __name__ == '__main__':
    print("Running Extended Mode Tests for Activity 3...")
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
        print(f"✅ ALL TESTS PASSED: {result.testsRun}/{total_tests}")
        print("Extended Mode implementation verified!")
    else:
        print(f"❌ SOME TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        print("Please fix issues before proceeding.")
    print("=" * 50)