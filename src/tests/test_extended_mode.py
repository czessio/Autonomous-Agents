# tests/test_extended_mode.py

import pytest
from src.activity3.communication import ExtendedCommunication, MessageType
from src.activity4.coordinator_agent import CoordinatorAgent
from src.activity1.terrain_robot import TerrainRobot

@pytest.fixture
def comm():
    return ExtendedCommunication()

@pytest.fixture
def robots():
    # create two dummy robots at base
    return [TerrainRobot(robot_id=i) for i in range(2)]

def test_message_send_receive(comm, robots):
    # robot_0 sends a rescue request to robot_1
    sender = "robot_0"
    receiver = "robot_1"
    payload = (5,5)
    comm.send(sender, receiver, MessageType.RESCUE_REQUEST, payload)
    msg = comm.receive(receiver)
    assert msg.sender == sender
    assert msg.receiver == receiver
    assert msg.msg_type == MessageType.RESCUE_REQUEST
    assert msg.payload == payload

def test_coordinator_allocates(comm, robots):
    coord = CoordinatorAgent(agent_id=0, base_pos=(0,0))
    # should not throw
    coord._allocate_resources(robots, comm)
    # check that each robot got a message
    for i in range(len(robots)):
        m = comm.receive(f"robot_{i}")
        assert m.msg_type in (MessageType.RESCUE_REQUEST, MessageType.RESCUE_ASSIGNED)
