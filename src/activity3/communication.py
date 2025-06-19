# Extended communication system for mountain rescue simulation - handles individual and broadcast messaging

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import time


class MessageType(Enum):
    PERSON_FOUND = "person_found"
    RESCUE_REQUEST = "rescue_request"
    RESCUE_ASSIGNED = "rescue_assigned"
    RESCUE_COMPLETED = "rescue_completed"
    STATUS_UPDATE = "status_update"


@dataclass
class Message:
    """Represents a communication message between agents"""
    msg_type: MessageType
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    location: Optional[Tuple[int, int]]
    urgency: Optional[int]
    timestamp: float
    data: Optional[Dict] = None


class ExtendedCommunication:
    """Advanced communication system for Extended Mode operation"""
    
    def __init__(self):
        # Message queues for each agent
        self.message_queues = {}
        self.broadcast_messages = []
        
        # Tracking systems
        self.found_persons = {}  # location -> {drone_id, urgency, timestamp}
        self.assigned_rescues = {}  # robot_id -> location
        self.completed_rescues = []
        self.pending_locations = []  # Locations awaiting robot assignment
        self.detected_locations = []
        self.waiting_drones = {}  # location -> drone_id (for Basic Mode compatibility)
        
        # Communication metrics
        self.total_messages_sent = 0
        self.individual_messages = 0
        self.broadcast_messages_count = 0
        self.message_history = []
        
        # Robot availability tracking
        self.robot_status = {}  # robot_id -> {'available': bool, 'battery': int}
        self.robot_targets = {}  # robot_id -> target location (Basic Mode compatibility)
    
    # Basic Mode compatibility methods
    def drone_waiting_at(self, location: Tuple[int, int], drone_id: str) -> None:
        """Drone reports it's waiting at a person location (Basic Mode compatibility)"""
        self.waiting_drones[location] = drone_id
    
    def drone_stopped_waiting(self, location: Tuple[int, int]) -> None:
        """Drone stops waiting at location (Basic Mode compatibility)"""
        if location in self.waiting_drones:
            del self.waiting_drones[location]
    
    def is_robot_near_location(self, robot_position: Tuple[int, int], 
                              target_location: Tuple[int, int], threshold: int = 1) -> bool:
        """Check if robot is near target location (Basic Mode compatibility)"""
        if not target_location:
            return False
        distance = abs(robot_position[0] - target_location[0]) + abs(robot_position[1] - target_location[1])
        return distance <= threshold
    
    def assign_robot_to_location(self, robot_id: str, location: Tuple[int, int]) -> None:
        """Assign a robot to go to a specific location (Basic Mode compatibility)"""
        self.robot_targets[robot_id] = location
        self.assign_robot_to_rescue(f"robot_{robot_id}", location)
        print(f" Robot {robot_id} assigned to rescue person at {location}")
    
    def robot_completed_rescue(self, robot_id: str, location: Tuple[int, int]) -> None:
        """Robot reports completing a rescue (Basic Mode compatibility)"""
        if robot_id in self.robot_targets:
            del self.robot_targets[robot_id]
        self.complete_rescue(f"robot_{robot_id}", location)
        print(f" Robot {robot_id} completed rescue at {location}")
    
    def get_nearest_person_location(self, robot_position: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Get nearest unassigned person location for a robot"""
        available_locations = []
        
        # Check pending locations first
        for location in self.pending_locations:
            if location not in self.assigned_rescues.values():
                available_locations.append(location)
        
        # Also check found persons
        for location in self.found_persons.keys():
            if location not in self.assigned_rescues.values() and location not in available_locations:
                available_locations.append(location)
        
        if not available_locations:
            return None
        
        # Find nearest location using Manhattan distance
        def distance(loc):
            return abs(loc[0] - robot_position[0]) + abs(loc[1] - robot_position[1])
        
        return min(available_locations, key=distance)
    
    # Extended Mode methods
    def register_agent(self, agent_id: str, agent_type: str) -> None:
        """Register an agent in the communication system"""
        self.message_queues[agent_id] = []
        if agent_type == "robot":
            self.robot_status[agent_id] = {'available': True, 'battery': 100}
    
    def send_individual_message(self, sender_id: str, recipient_id: str, 
                               msg_type: MessageType, location: Optional[Tuple[int, int]] = None,
                               urgency: Optional[int] = None, data: Optional[Dict] = None) -> None:
        """Send a message to a specific agent"""
        message = Message(
            msg_type=msg_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            location=location,
            urgency=urgency,
            timestamp=time.time(),
            data=data
        )
        
        if recipient_id in self.message_queues:
            self.message_queues[recipient_id].append(message)
            self.total_messages_sent += 1
            self.individual_messages += 1
            self.message_history.append(message)
            
            print(f" {sender_id} → {recipient_id}: {msg_type.value} at {location}")
    
    def broadcast_message(self, sender_id: str, msg_type: MessageType,
                         location: Optional[Tuple[int, int]] = None,
                         urgency: Optional[int] = None, data: Optional[Dict] = None) -> None:
        """Broadcast a message to all agents"""
        message = Message(
            msg_type=msg_type,
            sender_id=sender_id,
            recipient_id=None,  # Broadcast
            location=location,
            urgency=urgency,
            timestamp=time.time(),
            data=data
        )
        
        # Add to all agent queues except sender
        for agent_id in self.message_queues:
            if agent_id != sender_id:
                self.message_queues[agent_id].append(message)
        
        self.broadcast_messages.append(message)
        self.total_messages_sent += len(self.message_queues) - 1
        self.broadcast_messages_count += 1
        self.message_history.append(message)
        
        print(f" {sender_id} broadcasts: {msg_type.value} at {location}")
    
    def drone_found_person(self, drone_id: str, location: Tuple[int, int], 
                          urgency: int, broadcast: bool = True) -> None:
        """Drone reports finding a person - Extended Mode"""
        # Store person information
        self.found_persons[location] = {
            'drone_id': drone_id,
            'urgency': urgency,
            'timestamp': time.time(),
            'assigned': False
        }
        
        if location not in self.pending_locations:
            self.pending_locations.append(location)
        
        if location not in self.detected_locations:
            self.detected_locations.append(location)
        
        if broadcast:
            # Broadcast to all robots
            self.broadcast_message(
                sender_id=f"drone_{drone_id}",
                msg_type=MessageType.PERSON_FOUND,
                location=location,
                urgency=urgency
            )
        else:
            # Find nearest available robot
            nearest_robot = self._find_nearest_available_robot(location)
            if nearest_robot:
                self.send_individual_message(
                    sender_id=f"drone_{drone_id}",
                    recipient_id=nearest_robot,
                    msg_type=MessageType.RESCUE_REQUEST,
                    location=location,
                    urgency=urgency
                )
    
    def get_messages(self, agent_id: str) -> List[Message]:
        """Get all pending messages for an agent"""
        if agent_id in self.message_queues:
            messages = self.message_queues[agent_id].copy()
            self.message_queues[agent_id].clear()
            return messages
        return []
    
    def assign_robot_to_rescue(self, robot_id: str, location: Tuple[int, int]) -> bool:
        """Assign a robot to rescue at a specific location"""
        # Check if location is still available
        if location in self.found_persons and not self.found_persons[location]['assigned']:
            self.assigned_rescues[robot_id] = location
            self.found_persons[location]['assigned'] = True
            self.robot_status[robot_id]['available'] = False
            
            if location in self.pending_locations:
                self.pending_locations.remove(location)
            
            # Send confirmation
            self.send_individual_message(
                sender_id="system",
                recipient_id=robot_id,
                msg_type=MessageType.RESCUE_ASSIGNED,
                location=location
            )
            
            return True
        elif location in self.pending_locations:
            # Handle case where person info isn't in found_persons yet
            self.assigned_rescues[robot_id] = location
            if robot_id in self.robot_status:
                self.robot_status[robot_id]['available'] = False
            self.pending_locations.remove(location)
            return True
        return False
    
    def complete_rescue(self, robot_id: str, location: Tuple[int, int]) -> None:
        """Mark a rescue as completed"""
        if robot_id in self.assigned_rescues:
            del self.assigned_rescues[robot_id]
        
        if location in self.found_persons:
            rescue_info = self.found_persons[location].copy()
            rescue_info['completed_time'] = time.time()
            rescue_info['robot_id'] = robot_id
            self.completed_rescues.append(rescue_info)
            del self.found_persons[location]
        
        if location in self.waiting_drones:
            del self.waiting_drones[location]
        
        if location in self.detected_locations:
            self.detected_locations.remove(location)
        
        if robot_id in self.robot_status:
            self.robot_status[robot_id]['available'] = True
        
        # Broadcast completion
        self.broadcast_message(
            sender_id=robot_id,
            msg_type=MessageType.RESCUE_COMPLETED,
            location=location
        )
    
    def update_robot_status(self, robot_id: str, battery: int, available: bool = None) -> None:
        """Update robot status for intelligent assignment"""
        if robot_id in self.robot_status:
            self.robot_status[robot_id]['battery'] = battery
            if available is not None:
                self.robot_status[robot_id]['available'] = available
    
    def _find_nearest_available_robot(self, location: Tuple[int, int]) -> Optional[str]:
        """Find the nearest available robot with sufficient battery"""
        available_robots = [
            robot_id for robot_id, status in self.robot_status.items()
            if status['available'] and status['battery'] > 30
        ]
        
        if available_robots:
            # For now, return first available robot
            # In a full implementation, would calculate actual distances
            return available_robots[0]
        return None
    
    def get_pending_rescues(self) -> List[Tuple[int, int]]:
        """Get list of locations with people awaiting rescue"""
        return self.pending_locations.copy()
    
    def get_communication_stats(self) -> Dict:
        """Get communication performance statistics"""
        return {
            'total_messages': self.total_messages_sent,
            'individual_messages': self.individual_messages,
            'broadcast_messages': self.broadcast_messages_count,
            'pending_rescues': len(self.pending_locations),
            'active_rescues': len(self.assigned_rescues),
            'completed_rescues': len(self.completed_rescues),
            'message_efficiency': self.individual_messages / max(1, self.total_messages_sent)
        }
    
    def reset_metrics(self) -> None:
        """Reset communication metrics for new simulation run"""
        self.total_messages_sent = 0
        self.individual_messages = 0
        self.broadcast_messages_count = 0
        self.message_history.clear()


class BasicCommunication:
    """Legacy communication system for Basic Mode - maintained for compatibility"""
    
    def __init__(self):
        self.found_persons = {}
        self.waiting_drones = {}
        self.robot_targets = {}
        self.pending_locations = []  # Add this for compatibility
    
    def drone_found_person(self, drone_id: str, location: Tuple[int, int], 
                        urgency: int, broadcast: bool = True) -> bool:
        """Basic mode: simple location reporting with standard signature"""
        if location not in self.found_persons:
            self.found_persons[location] = {
                'drone_id': drone_id,
                'urgency': urgency,
                'timestamp': time.time(),
                'assigned': False
            }
            if location not in self.pending_locations:
                self.pending_locations.append(location)
            print(f"📡 Drone {drone_id} found person at {location} (urgency: {urgency})")
            return True
        return False
    
    def drone_waiting_at(self, location, drone_id):
        """Drone reports waiting at location"""
        self.waiting_drones[location] = drone_id
    
    def drone_stopped_waiting(self, location):
        """Drone stops waiting"""
        if location in self.waiting_drones:
            del self.waiting_drones[location]
    
    def get_nearest_person_location(self, robot_position):
        """Get nearest unassigned person for robot"""
        available_locations = []
        for location in self.found_persons.keys():
            if location not in self.robot_targets.values():
                available_locations.append(location)
        
        if not available_locations:
            return None
        
        def distance(loc):
            return abs(loc[0] - robot_position[0]) + abs(loc[1] - robot_position[1])
        
        return min(available_locations, key=distance)
    
    def assign_robot_to_location(self, robot_id, location):
        """Assign robot to location"""
        self.robot_targets[robot_id] = location
        print(f"📡 Robot {robot_id} assigned to {location}")
    
    def robot_completed_rescue(self, robot_id, location):
        """Mark rescue complete"""
        if robot_id in self.robot_targets:
            del self.robot_targets[robot_id]
        if location in self.found_persons:
            del self.found_persons[location]
        if location in self.waiting_drones:
            del self.waiting_drones[location]
        if location in self.pending_locations:
            self.pending_locations.remove(location)
    
    def is_robot_near_location(self, robot_position, target_location, threshold=1):
        """Check if robot near target"""
        if not target_location:
            return False
        distance = abs(robot_position[0] - target_location[0]) + abs(robot_position[1] - target_location[1])
        return distance <= threshold