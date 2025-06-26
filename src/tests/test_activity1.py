# Unit tests for Activity 1 - Mountain Rescue Simulation Basic Mode

import unittest
import sys
import os
import io
from contextlib import redirect_stdout

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'activity1'))

from activity1.environment import Environment
from activity1.terrain_robot import TerrainRobot, RobotState
from activity1.explorer_drone import ExplorerDrone, DroneState
from activity1.simulation import MountainRescueSimulation


class TestEnvironment(unittest.TestCase):
    """Test the simulation environment setup and functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.env = Environment(width=10, height=8, num_missing_persons=3, 
                              num_terrain_robots=2, num_drones=1)
    
    def test_environment_creation(self):
        """Test environment initialisation with correct dimensions and agent counts"""
        self.assertEqual(self.env.width, 10)
        self.assertEqual(self.env.height, 8)
        self.assertEqual(len(self.env.missing_persons), 3)
        self.assertEqual(len(self.env.terrain_robot_base), 2)
        self.assertEqual(len(self.env.drone_base), 1)
    
    def test_mountain_area_definition(self):
        """Test mountain area boundaries are properly defined"""
        self.assertIn('bounds', self.env.mountain_area)
        bounds = self.env.mountain_area['bounds']
        self.assertEqual(len(bounds), 4)  # x_start, y_start, x_end, y_end
    
    def test_elevation_zones(self):
        """Test elevation zones are set up with multiple height levels"""
        # Test that different elevations exist
        elevations = set()
        for y in range(self.env.height):
            for x in range(self.env.width):
                elevations.add(self.env.get_elevation(x, y))
        
        # Should have multiple elevation levels
        self.assertGreater(len(elevations), 1)
        self.assertIn(0, elevations)  # Base level should exist
    
    def test_person_placement(self):
        """Test missing persons are correctly placed within mountain boundaries"""
        x_start, y_start, x_end, y_end = self.env.mountain_area['bounds']
        
        for person_x, person_y in self.env.missing_persons:
            self.assertTrue(x_start <= person_x < x_end)
            self.assertTrue(y_start <= person_y < y_end)
    
    def test_person_rescue(self):
        """Test person rescue mechanism works correctly"""
        if self.env.missing_persons:
            person_pos = self.env.missing_persons[0]
            
            # Confirm person exists at location
            self.assertTrue(self.env.person_at_location(*person_pos))
            
            # Rescue person
            success = self.env.rescue_person(*person_pos)
            self.assertTrue(success)
            
            # Confirm person is rescued and moved to rescued list
            self.assertFalse(self.env.person_at_location(*person_pos))
            self.assertIn(person_pos, self.env.rescued_persons)
            self.assertNotIn(person_pos, self.env.missing_persons)
    
    def test_invalid_rescue(self):
        """Test rescuing at empty location returns False"""
        success = self.env.rescue_person(0, 0)
        self.assertFalse(success)


class TestTerrainRobot(unittest.TestCase):
    """Test terrain robot agent behavior and state management"""
    
    def setUp(self):
        """Setup test robot and environment"""
        self.env = Environment(width=10, height=8, num_missing_persons=2)
        self.robot = TerrainRobot(robot_id=1, start_position=(5, 5), 
                                 max_battery=100, battery_drain_rate=2)
    
    def test_robot_initialisation(self):
        """Test robot starts with correct initial values"""
        self.assertEqual(self.robot.robot_id, 1)
        self.assertEqual(self.robot.position, (5, 5))
        self.assertEqual(self.robot.base_position, (5, 5))
        self.assertEqual(self.robot.state, RobotState.AT_BASE)
        self.assertEqual(self.robot.current_battery, 100)
        self.assertTrue(self.robot.has_first_aid_kit)
    
    def test_battery_drain(self):
        """Test battery decreases when robot operates away from base"""
        initial_battery = self.robot.current_battery
        self.robot._drain_battery(self.env)
        self.assertLess(self.robot.current_battery, initial_battery)
    
    def test_state_transitions(self):
        """Test robot state machine transitions work correctly"""
        # Start at base
        self.assertEqual(self.robot.state, RobotState.AT_BASE)
        
        # Should transition to searching when battery is sufficient
        with redirect_stdout(io.StringIO()):  # Suppress print output
            self.robot.current_battery = 95  # Above min_leave_threshold (90)
            self.robot._handle_at_base_state(self.env)
            self.assertEqual(self.robot.state, RobotState.SEARCHING)
            
            # Test low battery state - robot must be AWAY from base
            self.robot.position = (3, 3)  # Move away from base position
            self.robot.current_battery = 10  # Very low battery
            self.robot.state = RobotState.SEARCHING  # Must be in active state
            
            self.robot.update(self.env)
            self.assertEqual(self.robot.state, RobotState.BATTERY_LOW)
    
    def test_valid_position_check(self):
        """Test position validation for environment boundaries"""
        # Valid positions
        self.assertTrue(self.robot._is_valid_position((0, 0), self.env))
        self.assertTrue(self.robot._is_valid_position((5, 5), self.env))
        
        # Invalid positions (outside boundaries)
        self.assertFalse(self.robot._is_valid_position((-1, 0), self.env))
        self.assertFalse(self.robot._is_valid_position((0, -1), self.env))
        self.assertFalse(self.robot._is_valid_position((10, 8), self.env))
    
    def test_movement_tracking(self):
        """Test robot tracks its movement history correctly"""
        initial_position = self.robot.position
        new_position = (6, 6)
        
        self.robot._update_position(new_position)
        
        self.assertEqual(self.robot.position, new_position)
        self.assertIn(initial_position, self.robot.last_positions)
    
    def test_rescue_capability(self):
        """Test robot can successfully rescue persons"""
        if self.env.missing_persons:
            person_pos = self.env.missing_persons[0]
            self.robot.position = person_pos
            self.robot.state = RobotState.DELIVERING
            self.robot.target_person = person_pos
            
            initial_rescued_count = self.robot.persons_rescued
            
            with redirect_stdout(io.StringIO()):  # Suppress print output
                self.robot._handle_delivering_state(self.env)
            
            # Check rescue occurred
            self.assertEqual(self.robot.persons_rescued, initial_rescued_count + 1)
            self.assertFalse(self.robot.has_first_aid_kit)
            self.assertEqual(self.robot.state, RobotState.RETURNING)


class TestExplorerDrone(unittest.TestCase):
    """Test explorer drone agent behavior and capabilities"""
    
    def setUp(self):
        """Setup test drone and environment"""
        self.env = Environment(width=10, height=8, num_missing_persons=2)
        self.drone = ExplorerDrone(drone_id=1, start_position=(7, 7), 
                                  max_battery=120, battery_drain_rate=3)
    
    def test_drone_initialisation(self):
        """Test drone starts with correct initial values"""
        self.assertEqual(self.drone.drone_id, 1)
        self.assertEqual(self.drone.position, (7, 7))
        self.assertEqual(self.drone.base_position, (7, 7))
        self.assertEqual(self.drone.state, DroneState.AT_BASE)
        self.assertEqual(self.drone.current_battery, 120)
        self.assertEqual(len(self.drone.movement_directions), 6)
    
    def test_six_directional_movement(self):
        """Test drone has 6-directional movement capability as required"""
        expected_directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
        self.assertEqual(self.drone.movement_directions, expected_directions)
    
    def test_battery_drain(self):
        """Test drone battery decreases during flight operations"""
        initial_battery = self.drone.current_battery
        self.drone._drain_battery()
        self.assertEqual(self.drone.current_battery, initial_battery - self.drone.battery_drain_rate)
    
    def test_search_pattern_generation(self):
        """Test drone generates systematic search patterns"""
        with redirect_stdout(io.StringIO()):  # Suppress print output
            self.drone._generate_search_pattern(self.env)
        
        # Should have generated some search pattern
        self.assertGreater(len(self.drone.search_pattern), 0)
        
        # All pattern positions should be valid
        for pos in self.drone.search_pattern:
            self.assertTrue(self.drone._is_valid_position(pos, self.env))
    
    def test_person_detection(self):
        """Test drone detects persons and changes state appropriately"""
        if self.env.missing_persons:
            person_pos = self.env.missing_persons[0]
            self.drone.position = person_pos
            self.drone.state = DroneState.EXPLORING
            
            with redirect_stdout(io.StringIO()):  # Suppress print output
                self.drone._handle_exploring_state(self.env)
            
            # Should detect person and change state
            self.assertEqual(self.drone.state, DroneState.FOUND_PERSON)
            self.assertIn(person_pos, self.drone.found_persons)
    
    def test_position_validation(self):
        """Test drone position validation works correctly"""
        # Valid positions
        self.assertTrue(self.drone._is_valid_position((0, 0), self.env))
        self.assertTrue(self.drone._is_valid_position((5, 5), self.env))
        
        # Invalid positions
        self.assertFalse(self.drone._is_valid_position((-1, 0), self.env))
        self.assertFalse(self.drone._is_valid_position((15, 15), self.env))
    
    def test_mission_reset(self):
        """Test drone properly resets mission data when returning to base"""
        # Set some mission data
        self.drone.found_persons = [(1, 1), (2, 2)]
        self.drone.explored_positions = {(3, 3), (4, 4)}
        self.drone.waiting_time = 5
        
        # Reset mission
        self.drone._reset_mission_data()
        
        # Check everything is cleared
        self.assertEqual(len(self.drone.found_persons), 0)
        self.assertEqual(len(self.drone.explored_positions), 0)
        self.assertEqual(self.drone.waiting_time, 0)


class TestSimulation(unittest.TestCase):
    """Test overall simulation coordination and management"""
    
    def setUp(self):
        """Setup test simulation"""
        with redirect_stdout(io.StringIO()):  # Suppress setup output
            self.simulation = MountainRescueSimulation(
                num_terrain_robots=2, num_drones=1, 
                num_missing_persons=3, max_simulation_steps=50
            )
    
    def test_simulation_initialisation(self):
        """Test simulation creates correct number of agents and environment"""
        self.assertEqual(len(self.simulation.terrain_robots), 2)
        self.assertEqual(len(self.simulation.drones), 1)
        self.assertEqual(len(self.simulation.environment.missing_persons), 3)
        self.assertEqual(self.simulation.max_steps, 50)
    
    def test_agent_creation(self):
        """Test agents are created at their designated base positions"""
        # Check robots are at their base positions
        for i, robot in enumerate(self.simulation.terrain_robots):
            expected_pos = self.simulation.environment.terrain_robot_base[i]
            self.assertEqual(robot.position, expected_pos)
            self.assertEqual(robot.base_position, expected_pos)
        
        # Check drones are at their base positions
        for i, drone in enumerate(self.simulation.drones):
            expected_pos = self.simulation.environment.drone_base[i]
            self.assertEqual(drone.position, expected_pos)
            self.assertEqual(drone.base_position, expected_pos)
    
    def test_simulation_step(self):
        """Test single simulation step executes without errors"""
        initial_step = self.simulation.current_step
        
        with redirect_stdout(io.StringIO()):  # Suppress step output
            self.simulation._update_simulation_step()
        
        # Step counter managed by run loop, agents should have status
        for robot in self.simulation.terrain_robots:
            self.assertIsNotNone(robot.get_status())
        
        for drone in self.simulation.drones:
            self.assertIsNotNone(drone.get_status())
    
    def test_statistics_tracking(self):
        """Test simulation properly tracks performance statistics"""
        self.simulation._update_statistics()
        
        # Check statistics structure
        self.assertIn('total_persons_rescued', self.simulation.stats)
        self.assertIn('total_battery_consumed', self.simulation.stats)
        self.assertGreaterEqual(self.simulation.stats['total_battery_consumed'], 0)
    
    def test_mission_completion_check(self):
        """Test simulation detects mission completion correctly"""
        # Simulate all persons rescued
        self.simulation.environment.missing_persons = []
        self.simulation.environment.rescued_persons = [(1, 1), (2, 2), (3, 3)]
        
        with redirect_stdout(io.StringIO()):  # Suppress completion message
            self.simulation._check_mission_status()
        
        # Mission should be marked as complete
        self.assertFalse(self.simulation.running)


class TestIntegration(unittest.TestCase):
    """Test complete system integration"""
    
    def test_short_simulation_run(self):
        """Test complete simulation runs without errors"""
        with redirect_stdout(io.StringIO()):  # Suppress all output
            simulation = MountainRescueSimulation(
                num_terrain_robots=1, num_drones=1, 
                num_missing_persons=1, max_simulation_steps=10
            )
            
            # Run simulation without visualisation
            report = simulation.run_simulation(visualise=False, step_delay=0)
        
        # Check report structure
        self.assertIn('simulation_steps', report)
        self.assertIn('persons_rescued', report)
        self.assertIn('success_rate', report)
        self.assertIn('robot_performance', report)
        self.assertIn('drone_performance', report)
        
        # Check performance data
        self.assertEqual(len(report['robot_performance']), 1)
        self.assertEqual(len(report['drone_performance']), 1)


if __name__ == '__main__':
    print("Running Mountain Rescue Simulation Tests...")
    print("=" * 50)
    
    # Create test suite and runner
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    # Count total tests
    total_tests = suite.countTestCases()
    
    # Print all tests being run
    test_count = 0
    for test_group in suite:
        for test in test_group:
            test_count += 1
            test_name = test._testMethodName
            test_class = test.__class__.__name__
            test_doc = getattr(test, test._testMethodName).__doc__
            if test_doc:
                # Clean up docstring - remove extra whitespace
                test_description = test_doc.strip()
            else:
                test_description = f"Test {test_name}"
            
            print(f"{test_count:2d}. {test_description}")
    
    # Run tests with suppressed default output
    stream = io.StringIO()  # Capture the default unittest output
    runner = unittest.TextTestRunner(verbosity=0, buffer=True, stream=stream)
    result = runner.run(suite)
    
    # Print clean summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    failed_tests = len(result.failures)
    error_tests = len(result.errors)
    passed_tests = result.testsRun - failed_tests - error_tests
    
    if result.wasSuccessful():
        print(f"ALL TESTS PASSED: {passed_tests}/{total_tests}")
        print("Mountain Rescue System is ready for operation!")
    else:
        print(f"SOME TESTS FAILED: {passed_tests}/{total_tests} passed")
        if failed_tests > 0:
            print(f"   {failed_tests} test failures")
        if error_tests > 0:
            print(f"   {error_tests} test errors")
        print("Please fix issues before proceeding.")
        
        # Show failure details if any
        if result.failures or result.errors:
            print("\nFAILURE DETAILS:")
            for test, trace in result.failures:
                print(f"FAILED: {test}")
                print(trace)
            for test, trace in result.errors:
                print(f"ERROR: {test}")
                print(trace)
    
    print("=" * 50)