# tests/test_basic_mode.py

import pytest
from src.activity1.environment import Environment
from src.activity1.terrain_robot import TerrainRobot
from src.activity1.simulation import Simulation  # or whatever your basic-mode class is called

def test_environment_initialisation():
    env = Environment(width=10, height=8, num_robots=2, num_drones=1, num_persons=3)
    assert env.width == 10
    assert env.height == 8
    assert len(env.robots) == 2
    assert len(env.drones) == 1
    assert len(env.persons) == 3

def test_basic_simulation_runs_to_completion(tmp_path):
    sim = Simulation(width=5, height=5, num_robots=1, num_drones=1, max_steps=20)
    report = sim.run(visualise=False)
    # report should be a dict containing these keys
    for key in ("Persons Rescued", "Mission Duration", "Success Rate"):
        assert key in report
    assert isinstance(report["Persons Rescued"], int)
    assert report["Mission Duration"] <= 20
