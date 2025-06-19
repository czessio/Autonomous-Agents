# Communication system for mountain rescue simulation - Basic Mode coordination

from typing import Tuple, List, Dict, Optional
import time

class BasicCommunication:
    """Simple communication system for Basic Mode - drones broadcast person locations"""
    
    def __init__(self):
        self.found_persons = {}  # Maps location to drone_id that found it
        self.waiting_drones = {}  # Maps location to drone waiting there
        self.robot_targets = {}  # Maps robot_id to target location
        
    def drone_found_person(self, drone_id: str, location: Tuple[int, int], 
                        urgency: int, broadcast: bool = True) -> bool:
        """Drone reports finding a person (compatible with extended logic)"""
        if location not in self.found_persons:
            self.found_persons[location] = {
                'drone_id': drone_id,
                'urgency': urgency,
                'timestamp': time.time(),
                'assigned': False
            }
            self.pending_locations.append(location)

            if broadcast:
                print(f"📡 Drone {drone_id} broadcasts: Person found at {location} (urgency: {urgency})")
            return True
        return False
    
    def drone_waiting_at(self, location, drone_id):
        """Drone reports it's waiting at a person location"""
        self.waiting_drones[location] = drone_id
    
    def drone_stopped_waiting(self, location):
        """Drone stops waiting at location"""
        if location in self.waiting_drones:
            del self.waiting_drones[location]
    
    def get_nearest_person_location(self, robot_position):
        """Get nearest unassigned person location for a robot"""
        available_locations = []
        for location in self.found_persons.keys():
            # Check if no robot is already assigned to this location
            if location not in self.robot_targets.values():
                available_locations.append(location)
        
        if not available_locations:
            return None
        
        # Find nearest location
        def distance(loc):
            return abs(loc[0] - robot_position[0]) + abs(loc[1] - robot_position[1])
        
        return min(available_locations, key=distance)
    
    def assign_robot_to_location(self, robot_id, location):
        """Assign a robot to go to a specific location"""
        self.robot_targets[robot_id] = location
        print(f"📡 Robot {robot_id} assigned to rescue person at {location}")
    
    def robot_completed_rescue(self, robot_id, location):
        """Robot reports completing a rescue"""
        if robot_id in self.robot_targets:
            del self.robot_targets[robot_id]
        if location in self.found_persons:
            del self.found_persons[location]
        if location in self.waiting_drones:
            del self.waiting_drones[location]
        print(f"📡 Robot {robot_id} completed rescue at {location}")
    
    def is_robot_near_location(self, robot_position, target_location, threshold=1):
        """Check if robot is near target location"""
        if not target_location:
            return False
        distance = abs(robot_position[0] - target_location[0]) + abs(robot_position[1] - target_location[1])
        return distance <= threshold