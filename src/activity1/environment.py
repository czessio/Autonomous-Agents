# Environment module for mountain rescue simulation - defines the world layout, missing persons, and robot base stations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict
import random


class Environment:
    def __init__(self, width: int = 20, height: int = 15, num_missing_persons: int = 6, 
                 num_terrain_robots: int = 3, num_drones: int = 2):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        
        # Define mountain area and elevation zones
        self.mountain_area = self._define_mountain_area()
        self._setup_elevation_zones()
        
        # Place missing persons
        self.missing_persons = self._place_missing_persons(num_missing_persons)
        
        # Define base station locations
        self.terrain_robot_base = self._define_terrain_robot_base(num_terrain_robots)
        self.drone_base = self._define_drone_base(num_drones)
        
        # Track rescued persons
        self.rescued_persons = []
        
    def _define_mountain_area(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Define the mountain area boundaries based on the environment image"""
        # Mountain occupies roughly upper-left quadrant
        mountain_x_start = 2
        mountain_x_end = 8
        mountain_y_start = 2
        mountain_y_end = 8
        
        return {
            'bounds': (mountain_x_start, mountain_y_start, mountain_x_end, mountain_y_end)
        }
    
    def _setup_elevation_zones(self):
        """Setup elevation zones within the mountain area"""
        x_start, y_start, x_end, y_end = self.mountain_area['bounds']
        
        # Create elevation zones as concentric rectangles
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                # Distance from centre determines elevation
                centre_x = (x_start + x_end) // 2
                centre_y = (y_start + y_end) // 2
                distance = max(abs(x - centre_x), abs(y - centre_y))
                
                if distance == 0:
                    self.grid[y][x] = 3000  # 3K MASL (centre)
                elif distance == 1:
                    self.grid[y][x] = 2000  # 2K MASL
                elif distance == 2:
                    self.grid[y][x] = 1000  # 1K MASL
                else:
                    self.grid[y][x] = 0     # 0 MASL (base level)
    
    def _place_missing_persons(self, num_persons: int) -> List[Tuple[int, int]]:
        """Place missing persons randomly across the mountain area"""
        x_start, y_start, x_end, y_end = self.mountain_area['bounds']
        persons = []
        
        for _ in range(num_persons):
            x = random.randint(x_start, x_end - 1)
            y = random.randint(y_start, y_end - 1)
            persons.append((x, y))
        
        return persons
    
    def _define_terrain_robot_base(self, num_robots: int) -> List[Tuple[int, int]]:
        """Define starting positions for terrain robots (right side of environment)"""
        base_positions = []
        start_x = self.width - 4
        start_y = 6
        
        for i in range(num_robots):
            x = start_x + (i % 2)
            y = start_y + (i // 2)
            base_positions.append((x, y))
        
        return base_positions
    
    def _define_drone_base(self, num_drones: int) -> List[Tuple[int, int]]:
        """Define starting positions for drones (bottom right of environment)"""
        base_positions = []
        start_x = self.width - 4
        start_y = self.height - 3
        
        for i in range(num_drones):
            x = start_x + i
            y = start_y
            base_positions.append((x, y))
        
        return base_positions
    
    def get_elevation(self, x: int, y: int) -> int:
        """Get elevation at specific coordinates"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.grid[y][x])
        return 0
    
    def is_mountain_area(self, x: int, y: int) -> bool:
        """Check if coordinates are within mountain area"""
        x_start, y_start, x_end, y_end = self.mountain_area['bounds']
        return x_start <= x < x_end and y_start <= y < y_end
    
    def person_at_location(self, x: int, y: int) -> bool:
        """Check if there's a missing person at the given location"""
        return (x, y) in self.missing_persons
    
    def rescue_person(self, x: int, y: int) -> bool:
        """Rescue a person at the given location"""
        if (x, y) in self.missing_persons:
            self.missing_persons.remove((x, y))
            self.rescued_persons.append((x, y))
            return True
        return False
    
    def visualise(self, terrain_robots=None, drones=None, ax=None):
        """Visualise the environment with robots and drones"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw elevation zones
        x_start, y_start, x_end, y_end = self.mountain_area['bounds']
        
        # 0 MASL (lightest green, outermost)
        rect_0 = patches.Rectangle((x_start-0.5, y_start-0.5), x_end-x_start, y_end-y_start, 
                                  linewidth=2, edgecolor='black', facecolor='lightgreen')
        ax.add_patch(rect_0)
        
        # 1K MASL
        centre_x = (x_start + x_end) // 2
        centre_y = (y_start + y_end) // 2
        rect_1k = patches.Rectangle((centre_x-2.5, centre_y-2.5), 5, 5, 
                                   linewidth=1, edgecolor='black', facecolor='mediumseagreen')
        ax.add_patch(rect_1k)
        
        # 2K MASL
        rect_2k = patches.Rectangle((centre_x-1.5, centre_y-1.5), 3, 3, 
                                   linewidth=1, edgecolor='black', facecolor='forestgreen')
        ax.add_patch(rect_2k)
        
        # 3K MASL (darkest green, centre)
        rect_3k = patches.Rectangle((centre_x-0.5, centre_y-0.5), 1, 1, 
                                   linewidth=1, edgecolor='black', facecolor='darkgreen')
        ax.add_patch(rect_3k)
        
        # Draw missing persons
        for x, y in self.missing_persons:
            ax.plot(x, y, 'ko', markersize=10, markerfacecolor='red')
            ax.text(x-0.2, y+0.3, '♂', fontsize=14, fontweight='bold', color='white')
        
        # Draw rescued persons (if any)
        for x, y in self.rescued_persons:
            ax.plot(x, y, 'go', markersize=8, markerfacecolor='green', alpha=0.7)
            ax.text(x-0.2, y+0.3, '✓', fontsize=12, fontweight='bold', color='white')
        
        # Draw terrain robots
        if terrain_robots:
            for robot in terrain_robots:
                x, y = robot.position
                # Color based on state
                if robot.state.value == 'searching':
                    color = 'yellow'
                elif robot.state.value == 'delivering':
                    color = 'orange'
                elif robot.state.value == 'returning':
                    color = 'lightblue'
                elif robot.state.value == 'battery_low':
                    color = 'red'
                else:
                    color = 'white'
                
                rect = patches.Rectangle((x-0.3, y-0.3), 0.6, 0.6, 
                                       linewidth=2, edgecolor='black', facecolor=color)
                ax.add_patch(rect)
                ax.text(x, y, '+', fontsize=12, fontweight='bold', 
                       ha='center', va='center', color='red')
        
        # Draw drones
        if drones:
            for drone in drones:
                x, y = drone.position
                # Color based on state
                if drone.state.value == 'exploring':
                    color = 'cyan'
                elif drone.state.value == 'found_person':
                    color = 'orange'
                elif drone.state.value == 'waiting':
                    color = 'yellow'
                elif drone.state.value == 'battery_low':
                    color = 'red'
                else:
                    color = 'black'
                
                ax.plot(x, y, 's', markersize=12, markerfacecolor=color, 
                       markeredgecolor='white', markeredgewidth=2)
        
        # Draw base station areas (lightly)
        for x, y in self.terrain_robot_base:
            ax.plot(x, y, 'rs', markersize=8, alpha=0.3)
        
        for x, y in self.drone_base:
            ax.plot(x, y, 'bs', markersize=8, alpha=0.3)
        
        # Add elevation labels
        ax.text(centre_x, centre_y, '3K', ha='center', va='center', 
               fontweight='bold', color='white', fontsize=10)
        ax.text(centre_x, centre_y-1, '2K', ha='center', va='center', 
               fontweight='bold', color='white', fontsize=10)
        ax.text(centre_x, centre_y-2, '1K', ha='center', va='center', 
               fontweight='bold', color='white', fontsize=10)
        ax.text(x_start+0.5, y_start+0.5, '0', ha='left', va='bottom', 
               fontweight='bold', fontsize=10)
        
        # Set up the plot
        ax.set_xlim(-1, self.width)
        ax.set_ylim(-1, self.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        return ax