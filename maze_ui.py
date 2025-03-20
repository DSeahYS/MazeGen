#!/usr/bin/env python3
"""
Magnificent Maze Generator - Optimized version with custom path drawing
"""

import pygame
import sys
import random
import time
import argparse
from typing import List, Tuple, Dict, Any, Set
from enum import Enum
from datetime import datetime
import heapq  # For A* pathfinding

# Initialize pygame
pygame.init()

# Constants
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
SIDEBAR_WIDTH = 300
MIN_CELL_SIZE = 4
MAX_CELL_SIZE = 100
DEFAULT_CELL_SIZE = 15
FPS = 60

# --- Maze Generation Algorithms ---
class MazeAlgorithm(Enum):
    ELLERS = "Eller's Algorithm"
    BACKTRACKER = "Recursive Backtracker"
    HYBRID = "Hybrid Algorithm"

class BaseMaze:
    """Base class for maze generators."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.maze_width = 2 * width + 1
        self.maze_height = 2 * height + 1
        self.maze = [['#' for _ in range(self.maze_width)] for _ in range(self.maze_height)]
        self.generation_time = 0
        self.generation_steps = []
        
    def generate(self) -> List[List[str]]:
        raise NotImplementedError
    
    def create_entrance_exit(self) -> None:
        self.maze[1][0] = ' '  # Entrance at top-left
        self.maze[self.maze_height-2][self.maze_width-1] = ' '  # Exit at bottom-right
    
    def save_to_file(self, filename: str) -> None:
        with open(filename, 'w') as f:
            for row in self.maze:
                f.write(''.join(row) + '\n')


class EllerMaze(BaseMaze):
    def __init__(self, width, height, density=0.5):
        super().__init__(width, height)
        self.density = density
        
    def generate(self):
        start_time = time.time()
        self.maze = [['#' for _ in range(self.maze_width)] for _ in range(self.maze_height)]
        self.generation_steps = []
        
        # Process each row with Eller's algorithm
        row = list(range(self.width))  # Each cell starts in its own set
        next_set = self.width
        
        for y in range(self.height):
            # Set cells in the current row
            for x in range(self.width):
                self.maze[2*y+1][2*x+1] = ' '
            
            self.generation_steps.append([row[:] for row in self.maze])
            
            # Randomly connect cells horizontally
            for x in range(self.width - 1):
                if row[x] != row[x+1] and random.random() < self.density:
                    self.maze[2*y+1][2*x+2] = ' '  # Remove wall
                    
                    # Merge sets
                    old_set, new_set = row[x+1], row[x]
                    for i in range(self.width):
                        if row[i] == old_set:
                            row[i] = new_set
                    
                    self.generation_steps.append([row[:] for row in self.maze])
            
            # Last row: connect all different sets
            if y == self.height - 1:
                for x in range(self.width - 1):
                    if row[x] != row[x+1]:
                        self.maze[2*y+1][2*x+2] = ' '
                        self.generation_steps.append([row[:] for row in self.maze])
                continue
            
            # Group cells by set
            sets = {}
            for x, set_id in enumerate(row):
                if set_id not in sets:
                    sets[set_id] = []
                sets[set_id].append(x)
            
            next_row = [-1] * self.width
            
            # Connect some cells to row below
            for set_id, cells in sets.items():
                connect_count = max(1, min(len(cells), int(len(cells) * (1 - self.density) + 0.5)))
                for x in random.sample(cells, connect_count):
                    self.maze[2*y+2][2*x+1] = ' '  # Remove wall below
                    next_row[x] = set_id
                    self.generation_steps.append([row[:] for row in self.maze])
            
            # Assign new set IDs to unconnected cells
            for x in range(self.width):
                if next_row[x] == -1:
                    next_row[x] = next_set
                    next_set += 1
            
            row = next_row
        
        self.create_entrance_exit()
        self.generation_steps.append([row[:] for row in self.maze])
        self.generation_time = time.time() - start_time
        return self.maze


class RecursiveBacktracker(BaseMaze):
    def generate(self):
        start_time = time.time()
        self.maze = [['#' for _ in range(self.maze_width)] for _ in range(self.maze_height)]
        self.generation_steps = []
        
        # Create cells
        for y in range(self.height):
            for x in range(self.width):
                self.maze[2*y+1][2*x+1] = ' '
        
        self.generation_steps.append([row[:] for row in self.maze])
        
        # Execute recursive backtracker algorithm
        stack = []
        visited = set()
        
        # Start at random cell
        start_x, start_y = random.randint(0, self.width-1), random.randint(0, self.height-1)
        stack.append((start_x, start_y))
        visited.add((start_x, start_y))
        
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        
        while stack:
            x, y = stack[-1]
            
            # Mark current cell for visualization
            current_maze = [row[:] for row in self.maze]
            current_maze[2*y+1][2*x+1] = 'C'
            self.generation_steps.append(current_maze)
            
            # Find unvisited neighbors
            neighbors = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                self.maze[2*y+1+dy][2*x+1+dx] = ' '  # Remove wall
                stack.append((nx, ny))
                visited.add((nx, ny))
            else:
                stack.pop()  # Backtrack
                if stack:
                    bx, by = stack[-1]
                    current_maze = [row[:] for row in self.maze]
                    current_maze[2*by+1][2*bx+1] = 'B'
                    self.generation_steps.append(current_maze)
        
        # Clean up temporary markers
        for y in range(self.height):
            for x in range(self.width):
                if self.maze[2*y+1][2*x+1] in ['C', 'B']:
                    self.maze[2*y+1][2*x+1] = ' '
        
        self.create_entrance_exit()
        self.generation_steps.append([row[:] for row in self.maze])
        self.generation_time = time.time() - start_time
        return self.maze


class HybridMaze(BaseMaze):
    def __init__(self, width, height, density=0.6, section_size=5):
        super().__init__(width, height)
        self.density = density
        self.section_size = section_size
        
    def generate(self):
        start_time = time.time()
        self.generation_steps = []
        
        # Generate base maze using Eller's Algorithm
        eller = EllerMaze(self.width, self.height, self.density)
        self.maze = eller.generate()
        self.generation_steps = eller.generation_steps
        
        # Apply recursive backtracker to random sections
        sections_x = self.width // self.section_size
        sections_y = self.height // self.section_size
        
        section_count = max(1, min(10, sections_x * sections_y // 3))
        for _ in range(section_count):
            section_x = random.randint(0, max(0, sections_x - 1))
            section_y = random.randint(0, max(0, sections_y - 1))
            
            start_x = section_x * self.section_size
            start_y = section_y * self.section_size
            end_x = min(start_x + self.section_size, self.width)
            end_y = min(start_y + self.section_size, self.height)
            
            # Highlight section
            highlight_maze = [row[:] for row in self.maze]
            for y in range(start_y, end_y):
                for x in range(start_x, end_x):
                    if highlight_maze[2*y+1][2*x+1] == ' ':
                        highlight_maze[2*y+1][2*x+1] = 'H'
            self.generation_steps.append(highlight_maze)
            
            self._rework_section(start_x, start_y, end_x, end_y)
        
        self.create_entrance_exit()
        self.generation_steps.append([row[:] for row in self.maze])
        self.generation_time = time.time() - start_time
        return self.maze
    
    def _rework_section(self, start_x, start_y, end_x, end_y):
        # Reset this section
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                self.maze[2*y+1][2*x+1] = ' '  # Keep cells
                if x < end_x - 1:
                    self.maze[2*y+1][2*x+2] = '#'  # Reset horizontal wall
                if y < end_y - 1:
                    self.maze[2*y+2][2*x+1] = '#'  # Reset vertical wall
        
        self.generation_steps.append([row[:] for row in self.maze])
        
        # Apply recursive backtracker to the section
        stack = []
        visited = set()
        
        # Start at random cell within section
        x = random.randint(start_x, end_x - 1)
        y = random.randint(start_y, end_y - 1)
        stack.append((x, y))
        visited.add((x, y))
        
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        
        while stack:
            x, y = stack[-1]
            
            current_maze = [row[:] for row in self.maze]
            current_maze[2*y+1][2*x+1] = 'C'
            self.generation_steps.append(current_maze)
            
            # Find unvisited neighbors within the section
            neighbors = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (start_x <= nx < end_x and start_y <= ny < end_y and 
                    (nx, ny) not in visited):
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                self.maze[2*y+1+dy][2*x+1+dx] = ' '  # Remove wall
                stack.append((nx, ny))
                visited.add((nx, ny))
                self.generation_steps.append([row[:] for row in self.maze])
            else:
                stack.pop()
                if stack:
                    bx, by = stack[-1]
                    current_maze = [row[:] for row in self.maze]
                    current_maze[2*by+1][2*bx+1] = 'B'
                    self.generation_steps.append(current_maze)
        
        # Clean up markers
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                if self.maze[2*y+1][2*x+1] in ['C', 'B', 'H']:
                    self.maze[2*y+1][2*x+1] = ' '


class CustomPathMaze(BaseMaze):
    """Maze generator that incorporates a user-defined path."""
    
    def __init__(self, width, height, path, algorithm, density=0.5, section_size=5):
        super().__init__(width, height)
        self.path = path
        self.algorithm = algorithm
        self.density = density
        self.section_size = section_size
    
    def generate(self):
        start_time = time.time()
        self.maze = [['#' for _ in range(self.maze_width)] for _ in range(self.maze_height)]
        self.generation_steps = []
        
        # Create user path
        self._create_user_path()
        self.generation_steps.append([row[:] for row in self.maze])
        
        # Generate rest of maze
        if self.algorithm == MazeAlgorithm.ELLERS:
            self._generate_around_path_eller()
        elif self.algorithm == MazeAlgorithm.BACKTRACKER:
            self._generate_around_path_backtracker()
        else:  # Hybrid
            self._generate_around_path_hybrid()
        
        self.create_entrance_exit()
        self.generation_steps.append([row[:] for row in self.maze])
        
        self.generation_time = time.time() - start_time
        return self.maze
    
    def _create_user_path(self):
        # Create cells at path points
        for x, y in self.path:
            self.maze[2*y+1][2*x+1] = ' '
        
        # Connect adjacent points
        for i in range(len(self.path) - 1):
            x1, y1 = self.path[i]
            x2, y2 = self.path[i+1]
            
            dx, dy = x2 - x1, y2 - y1
            
            if abs(dx) + abs(dy) == 1:  # Directly adjacent
                wall_x = 2*x1 + 1 + dx
                wall_y = 2*y1 + 1 + dy
                self.maze[wall_y][wall_x] = ' '
            else:  # Create path between non-adjacent points
                steps = max(abs(dx), abs(dy))
                for step in range(1, steps + 1):
                    t = step / steps
                    ix, iy = int(x1 + dx * t), int(y1 + dy * t)
                    
                    # Create cell
                    self.maze[2*iy+1][2*ix+1] = ' '
                    
                    # Connect to previous cell
                    if step > 1:
                        prev_x = int(x1 + dx * (step - 1) / steps)
                        prev_y = int(y1 + dy * (step - 1) / steps)
                        
                        # Calculate wall position
                        wall_x = 2*prev_x + 1 + (1 if ix > prev_x else (-1 if ix < prev_x else 0))
                        wall_y = 2*prev_y + 1 + (1 if iy > prev_y else (-1 if iy < prev_y else 0))
                        
                        self.maze[wall_y][wall_x] = ' '
    
    def _generate_around_path_eller(self):
        # Create set of path cells
        path_cells = set(self.path)
        
        # Same algorithm as EllerMaze but preserving the user path
        row = list(range(self.width))
        next_set = self.width
        
        # Initialize sets based on user path connections
        for y in range(self.height):
            # Check horizontal connections
            for x in range(self.width - 1):
                if self.maze[2*y+1][2*x+2] == ' ':  # Already connected
                    old_set, new_set = row[x+1], row[x]
                    for i in range(self.width):
                        if row[i] == old_set:
                            row[i] = new_set
            
            self.generation_steps.append([row[:] for row in self.maze])
            
            # Connect remaining cells
            for x in range(self.width - 1):
                if (row[x] != row[x+1] and random.random() < self.density and
                    self.maze[2*y+1][2*x+2] == '#'):  # Wall exists
                    
                    self.maze[2*y+1][2*x+2] = ' '  # Remove wall
                    
                    # Merge sets
                    old_set, new_set = row[x+1], row[x]
                    for i in range(self.width):
                        if row[i] == old_set:
                            row[i] = new_set
                    
                    self.generation_steps.append([row[:] for row in self.maze])
            
            # Last row connects all sets
            if y == self.height - 1:
                for x in range(self.width - 1):
                    if row[x] != row[x+1]:
                        self.maze[2*y+1][2*x+2] = ' '
                        self.generation_steps.append([row[:] for row in self.maze])
                continue
            
            # Group by set
            sets = {}
            for x, set_id in enumerate(row):
                if set_id not in sets:
                    sets[set_id] = []
                sets[set_id].append(x)
            
            next_row = [-1] * self.width
            
            # Check existing vertical connections
            for x in range(self.width):
                if self.maze[2*y+2][2*x+1] == ' ':  # Already connected
                    next_row[x] = row[x]
            
            # Connect some cells vertically
            for set_id, cells in sets.items():
                unconnected = [x for x in cells if next_row[x] == -1]
                
                if unconnected:
                    connect_count = max(1, min(len(unconnected), 
                                           int(len(unconnected) * (1 - self.density) + 0.5)))
                    
                    for x in random.sample(unconnected, connect_count):
                        self.maze[2*y+2][2*x+1] = ' '  # Remove wall below
                        next_row[x] = set_id
                        self.generation_steps.append([row[:] for row in self.maze])
            
            # Assign new set IDs
            for x in range(self.width):
                if next_row[x] == -1:
                    next_row[x] = next_set
                    next_set += 1
            
            row = next_row
    
    def _generate_around_path_backtracker(self):
        # Set of all path cells
        path_cells = set(self.path)
        visited = set(path_cells)
        
        # Find frontier cells (path cells with unvisited neighbors)
        frontier = []
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W
        
        for x, y in path_cells:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    (nx, ny) not in visited):
                    frontier.append((x, y))
                    break
        
        # If no frontier, pick any unvisited cell
        if not frontier and len(visited) < self.width * self.height:
            for y in range(self.height):
                for x in range(self.width):
                    if (x, y) not in visited:
                        self.maze[2*y+1][2*x+1] = ' '
                        frontier.append((x, y))
                        visited.add((x, y))
                        break
                if frontier:
                    break
        
        # Recursive backtracker from frontier
        while frontier:
            x, y = frontier[-1]
            
            # Mark for visualization
            current_maze = [row[:] for row in self.maze]
            current_maze[2*y+1][2*x+1] = 'C'
            self.generation_steps.append(current_maze)
            
            # Find unvisited neighbors
            neighbors = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    (nx, ny) not in visited):
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                
                # Create cell and remove wall
                self.maze[2*ny+1][2*nx+1] = ' '
                self.maze[2*y+1+dy][2*x+1+dx] = ' '
                
                frontier.append((nx, ny))
                visited.add((nx, ny))
            else:
                frontier.pop()
                
                if frontier:
                    bx, by = frontier[-1]
                    current_maze = [row[:] for row in self.maze]
                    current_maze[2*by+1][2*bx+1] = 'B'
                    self.generation_steps.append(current_maze)
    
    def _generate_around_path_hybrid(self):
        # First use Eller's
        self._generate_around_path_eller()
        
        # Then rework sections with backtracker
        sections_x = self.width // self.section_size
        sections_y = self.height // self.section_size
        section_count = max(1, min(10, sections_x * sections_y // 3))
        
        for _ in range(section_count):
            section_x = random.randint(0, max(0, sections_x - 1))
            section_y = random.randint(0, max(0, sections_y - 1))
            
            start_x = section_x * self.section_size
            start_y = section_y * self.section_size
            end_x = min(start_x + self.section_size, self.width)
            end_y = min(start_y + self.section_size, self.height)
            
            # Count path cells in section
            path_cells = sum(1 for x, y in self.path 
                          if start_x <= x < end_x and start_y <= y < end_y)
            section_area = (end_x - start_x) * (end_y - start_y)
            
            # Skip if too many path cells
            if path_cells / section_area < 0.3:
                # Highlight section
                highlight_maze = [row[:] for row in self.maze]
                for y in range(start_y, end_y):
                    for x in range(start_x, end_x):
                        if highlight_maze[2*y+1][2*x+1] == ' ':
                            highlight_maze[2*y+1][2*x+1] = 'H'
                self.generation_steps.append(highlight_maze)
                
                self._rework_section_preserving_path(start_x, start_y, end_x, end_y)
    
    def _rework_section_preserving_path(self, start_x, start_y, end_x, end_y):
        # Find path cells and walls in section
        path_cells = set()
        path_walls = set()
        
        for i in range(len(self.path)):
            x, y = self.path[i]
            if start_x <= x < end_x and start_y <= y < end_y:
                path_cells.add((x, y))
                
                # Mark walls between path cells
                if i < len(self.path) - 1:
                    nx, ny = self.path[i+1]
                    if abs(nx - x) + abs(ny - y) == 1:
                        wall_x = 2*x + 1 + (nx - x)
                        wall_y = 2*y + 1 + (ny - y)
                        path_walls.add((wall_x, wall_y))
        
        # Reset section walls preserving path
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                self.maze[2*y+1][2*x+1] = ' '  # Keep cells
                
                # Reset walls unless part of path
                if x < end_x - 1:
                    wall_pos = (2*x+2, 2*y+1)
                    if wall_pos not in path_walls:
                        self.maze[wall_pos[1]][wall_pos[0]] = '#'
                
                if y < end_y - 1:
                    wall_pos = (2*x+1, 2*y+2)
                    if wall_pos not in path_walls:
                        self.maze[wall_pos[1]][wall_pos[0]] = '#'
        
        self.generation_steps.append([row[:] for row in self.maze])
        
        # Apply recursive backtracker
        stack = []
        visited = set(path_cells)
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        # Start with path cell if available
        if path_cells:
            x, y = random.choice(list(path_cells))
            stack.append((x, y))
        else:
            x = random.randint(start_x, end_x - 1)
            y = random.randint(start_y, end_y - 1)
            stack.append((x, y))
            visited.add((x, y))
        
        while stack:
            x, y = stack[-1]
            
            # Visualization
            current_maze = [row[:] for row in self.maze]
            current_maze[2*y+1][2*x+1] = 'C'
            self.generation_steps.append(current_maze)
            
            # Find unvisited neighbors
            neighbors = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (start_x <= nx < end_x and start_y <= ny < end_y and 
                    (nx, ny) not in visited):
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                
                # Check if wall is part of path
                wall_x = 2*x + 1 + dx
                wall_y = 2*y + 1 + dy
                
                if (wall_x, wall_y) not in path_walls:
                    self.maze[wall_y][wall_x] = ' '  # Remove wall
                
                stack.append((nx, ny))
                visited.add((nx, ny))
                self.generation_steps.append([row[:] for row in self.maze])
            else:
                stack.pop()
                
                if stack:
                    bx, by = stack[-1]
                    current_maze = [row[:] for row in self.maze]
                    current_maze[2*by+1][2*bx+1] = 'B'
                    self.generation_steps.append(current_maze)
        
        # Clean up markers
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                if self.maze[2*y+1][2*x+1] in ['C', 'B', 'H']:
                    self.maze[2*y+1][2*x+1] = ' '


def create_maze(width, height, algorithm, density=0.5, section_size=5):
    """Create maze with specified algorithm."""
    if algorithm == MazeAlgorithm.ELLERS:
        generator = EllerMaze(width, height, density)
    elif algorithm == MazeAlgorithm.BACKTRACKER:
        generator = RecursiveBacktracker(width, height)
    else:  # Hybrid
        generator = HybridMaze(width, height, density, section_size)
    
    generator.generate()
    return generator


class MazeSolver:
    """Solves mazes using various algorithms."""
    
    def __init__(self, maze):
        self.maze = maze
        self.height = len(maze)
        self.width = len(maze[0]) if self.height > 0 else 0
        self.solution = []
        self.solving_steps = []
    
    def solve(self, algorithm="astar"):
        # Find entrance and exit
        entrance = exit = None
        
        for y in range(self.height):
            if self.maze[y][0] == ' ':
                entrance = (0, y)
            if self.maze[y][self.width-1] == ' ':
                exit = (self.width-1, y)
        
        if not entrance:
            for x in range(self.width):
                if self.maze[0][x] == ' ':
                    entrance = (x, 0)
                if self.maze[self.height-1][x] == ' ':
                    exit = (x, self.height-1)
        
        if not entrance or not exit:
            return []
        
        # Reset state
        self.solution = []
        self.solving_steps = []
        
        # Choose algorithm
        if algorithm == "dfs":
            return self._solve_dfs(entrance, exit)
        elif algorithm == "bfs":
            return self._solve_bfs(entrance, exit)
        else:  # A* (default)
            return self._solve_astar(entrance, exit)
    
    def _solve_dfs(self, start, end):
        """Depth-first search."""
        stack = [(start, [start])]
        visited = {start}
        
        while stack:
            (x, y), path = stack.pop()
            
            # Record step
            current_maze = [row[:] for row in self.maze]
            current_maze[y][x] = 'X'
            self.solving_steps.append((current_maze, path))
            
            if (x, y) == end:
                self.solution = path
                return path
            
            # Check all directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    self.maze[ny][nx] == ' ' and (nx, ny) not in visited):
                    stack.append(((nx, ny), path + [(nx, ny)]))
                    visited.add((nx, ny))
        
        return []
    
    def _solve_bfs(self, start, end):
        """Breadth-first search."""
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            (x, y), path = queue.pop(0)
            
            # Record step
            current_maze = [row[:] for row in self.maze]
            current_maze[y][x] = 'X'
            self.solving_steps.append((current_maze, path))
            
            if (x, y) == end:
                self.solution = path
                return path
            
            # Check all directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    self.maze[ny][nx] == ' ' and (nx, ny) not in visited):
                    queue.append(((nx, ny), path + [(nx, ny)]))
                    visited.add((nx, ny))
        
        return []
    
    def _solve_astar(self, start, end):
        """A* algorithm."""
        # Heuristic (Manhattan distance)
        def h(pos):
            return abs(pos[0] - end[0]) + abs(pos[1] - end[1])
        
        open_set = [(h(start), 0, start, [start])]  # (f, g, pos, path)
        closed_set = set()
        g_score = {start: 0}
        
        while open_set:
            f, g, (x, y), path = heapq.heappop(open_set)
            
            # Record step
            current_maze = [row[:] for row in self.maze]
            current_maze[y][x] = 'X'
            self.solving_steps.append((current_maze, path))
            
            if (x, y) == end:
                self.solution = path
                return path
            
            if (x, y) in closed_set:
                continue
            
            closed_set.add((x, y))
            
            # Check all directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    self.maze[ny][nx] == ' '):
                    
                    tentative_g = g + 1
                    
                    if (nx, ny) in closed_set and tentative_g >= g_score.get((nx, ny), float('inf')):
                        continue
                    
                    if tentative_g < g_score.get((nx, ny), float('inf')):
                        g_score[(nx, ny)] = tentative_g
                        f_score = tentative_g + h((nx, ny))
                        heapq.heappush(open_set, (f_score, tentative_g, (nx, ny), path + [(nx, ny)]))
        
        return []


# --- Visualization and UI ---
class MazeTheme:
    """Color themes for maze display."""
    
    THEMES = {
        "Cosmic": {
            "background": (5, 5, 15),
            "wall": [(25, 25, 112), (138, 43, 226)],  # Navy to purple
            "cell": [(0, 0, 0), (10, 10, 30)],
            "current": (255, 215, 0),  # Gold
            "entrance": (0, 255, 127),  # Spring green
            "exit": (220, 20, 60),  # Crimson
            "solution": [(255, 105, 180), (255, 20, 147)],
            "highlight": (255, 255, 255),
            "text": (240, 240, 255),
            "button": (70, 70, 120),
            "button_hover": (90, 90, 150),
            "slider": (70, 70, 120),
            "slider_handle": (150, 150, 200),
        },
        "Forest": {
            "background": (10, 30, 15),
            "wall": [(34, 139, 34), (0, 100, 0)],  # Forest green
            "cell": [(0, 0, 0), (10, 20, 10)],
            "current": (255, 215, 0),  # Gold
            "entrance": (152, 251, 152),  # Pale green
            "exit": (220, 20, 60),  # Crimson
            "solution": [(255, 165, 0), (255, 140, 0)],
            "highlight": (255, 255, 255),
            "text": (220, 255, 220),
            "button": (40, 80, 40),
            "button_hover": (60, 100, 60),
            "slider": (40, 80, 40),
            "slider_handle": (100, 170, 100),
        },
        "Neon": {
            "background": (5, 5, 5),
            "wall": [(0, 255, 255), (0, 150, 255)],  # Cyan to blue
            "cell": [(0, 0, 0), (5, 5, 5)],
            "current": (255, 215, 0),  # Gold
            "entrance": (0, 255, 127),  # Spring green
            "exit": (255, 20, 147),  # Deep pink
            "solution": [(255, 0, 255), (200, 0, 255)],
            "highlight": (255, 255, 255),
            "text": (180, 255, 255),
            "button": (0, 100, 100),
            "button_hover": (0, 130, 130),
            "slider": (0, 100, 100),
            "slider_handle": (0, 200, 200),
        }
    }
    
    @classmethod
    def get_theme(cls, name):
        return cls.THEMES.get(name, cls.THEMES["Cosmic"])
    
    @classmethod
    def get_theme_names(cls):
        return list(cls.THEMES.keys())


class MazeUI:
    """Magnificent Maze UI class."""
    
    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        self.screen_width = width
        self.screen_height = height
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Magnificent Maze Generator")
        
        # Maze parameters
        self.maze_width = 30
        self.maze_height = 30
        self.cell_size = DEFAULT_CELL_SIZE
        self.offset_x = 0
        self.offset_y = 0
        self.algorithm = MazeAlgorithm.HYBRID
        self.density = 0.5
        self.section_size = 5
        
        # UI state
        self.clock = pygame.time.Clock()
        self.dragging = False
        self.drag_start = (0, 0)
        self.mouse_pos = (0, 0)
        self.generating = False
        self.solving = False
        self.show_solution = False
        self.drawing_mode = False
        self.user_path = []
        self.animation_speed = 10
        self.current_step = 0
        
        # Fonts
        self.font = pygame.font.SysFont('Arial', 16)
        self.font_large = pygame.font.SysFont('Arial', 20, bold=True)
        self.font_title = pygame.font.SysFont('Arial', 24, bold=True)
        
        # Maze data
        self.maze = None
        self.maze_surface = None
        self.generation_steps = []
        self.solver = None
        self.solution = []
        self.solving_steps = []
        
        # UI elements
        self.buttons = []
        self.sliders = []
        self.active_element = None
        
        # Theme
        self.current_theme = "Cosmic"
        self.theme = MazeTheme.get_theme(self.current_theme)
        
        # Initialize UI and generate initial maze
        self._init_ui()
        self.generate_maze()
    
    def _init_ui(self):
        """Initialize UI elements."""
        self.buttons = []
        self.sliders = []
        
        sidebar_x = self.screen_width - SIDEBAR_WIDTH
        y_offset = 20
        
        # Title
        self.title_rect = pygame.Rect(sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 40)
        y_offset += 50
        
        # Algorithm selection
        for algo, label in [
            (MazeAlgorithm.ELLERS, "Eller's Algorithm"),
            (MazeAlgorithm.BACKTRACKER, "Recursive Backtracker"),
            (MazeAlgorithm.HYBRID, "Hybrid Algorithm")
        ]:
            self.add_button(label, sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                          lambda a=algo: self.set_algorithm(a))
            y_offset += 40
        
        y_offset += 10
        
        # Sliders
        self.add_slider("Width", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30, 
                      10, 100, self.maze_width, lambda v: self.set_maze_width(int(v)))
        y_offset += 50
        
        self.add_slider("Height", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                      10, 100, self.maze_height, lambda v: self.set_maze_height(int(v)))
        y_offset += 50
        
        self.add_slider("Density", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                      0.1, 0.9, self.density, lambda v: self.set_density(v))
        y_offset += 50
        
        self.add_slider("Section Size", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                      3, 20, self.section_size, lambda v: self.set_section_size(int(v)))
        y_offset += 50
        
        self.add_slider("Animation Speed", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                      1, 50, self.animation_speed, lambda v: self.set_animation_speed(int(v)))
        y_offset += 50
        
        # Theme selection
        self.theme_rect = pygame.Rect(sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30)
        theme_names = MazeTheme.get_theme_names()
        theme_width = (SIDEBAR_WIDTH - 20) // len(theme_names)
        
        for i, theme_name in enumerate(theme_names):
            self.add_button(theme_name[:1], 
                          sidebar_x + 10 + i * theme_width, 
                          y_offset, 
                          theme_width, 30,
                          lambda tn=theme_name: self.set_theme(tn))
        y_offset += 50
        
        # Action buttons
        self.add_button("Generate", sidebar_x + 10, y_offset, (SIDEBAR_WIDTH - 30) // 2, 40,
                      self.generate_maze)
        self.add_button("Solve", sidebar_x + 20 + (SIDEBAR_WIDTH - 30) // 2, y_offset, 
                      (SIDEBAR_WIDTH - 30) // 2, 40, self.solve_maze)
        y_offset += 50
        
        # Draw Path button
        self.add_button("Draw Path", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 40,
                      self.enable_path_drawing_mode)
        y_offset += 50
        
        # Zoom controls
        self.add_button("Zoom In", sidebar_x + 10, y_offset, (SIDEBAR_WIDTH - 30) // 2, 30,
                      lambda: self.set_cell_size(self.cell_size + 2))
        self.add_button("Zoom Out", sidebar_x + 20 + (SIDEBAR_WIDTH - 30) // 2, y_offset,
                      (SIDEBAR_WIDTH - 30) // 2, 30, lambda: self.set_cell_size(self.cell_size - 2))
        y_offset += 50
        
        # Reset view
        self.add_button("Reset View", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                      self.reset_view)
        y_offset += 50
        
        # Save/Export
        self.add_button("Save PNG", sidebar_x + 10, y_offset, (SIDEBAR_WIDTH - 30) // 2, 30,
                      lambda: self.save_maze("png"))
        self.add_button("Save TXT", sidebar_x + 20 + (SIDEBAR_WIDTH - 30) // 2, y_offset,
                      (SIDEBAR_WIDTH - 30) // 2, 30, lambda: self.save_maze("txt"))
        y_offset += 60
        
        # Status area
        self.status_rect = pygame.Rect(sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 100)
        self.status_message = "Ready"
    
    def add_button(self, text, x, y, width, height, action):
        self.buttons.append({
            "rect": pygame.Rect(x, y, width, height),
            "text": text,
            "action": action,
            "hover": False
        })
    
    def add_slider(self, text, x, y, width, height, min_value, max_value, initial_value, action):
        rel_pos = (initial_value - min_value) / (max_value - min_value)
        handle_x = x + int(rel_pos * (width - 20))
        
        self.sliders.append({
            "rect": pygame.Rect(x, y, width, height),
            "text": text,
            "min": min_value,
            "max": max_value,
            "value": initial_value,
            "action": action,
            "active": False,
            "handle_rect": pygame.Rect(handle_x, y, 20, height)
        })
    
    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
        self.update_status(f"Algorithm set to {algorithm.value}")
    
    def set_maze_width(self, width):
        self.maze_width = max(10, min(100, width))
    
    def set_maze_height(self, height):
        self.maze_height = max(10, min(100, height))
    
    def set_density(self, density):
        self.density = max(0.1, min(0.9, density))
    
    def set_section_size(self, size):
        self.section_size = max(3, min(20, size))
    
    def set_animation_speed(self, speed):
        self.animation_speed = max(1, min(50, speed))
    
    def set_cell_size(self, size):
        old_size = self.cell_size
        self.cell_size = max(MIN_CELL_SIZE, min(MAX_CELL_SIZE, size))
        
        # Adjust offset to keep center point
        if self.maze and old_size != self.cell_size:
            center_x = self.offset_x + ((self.screen_width - SIDEBAR_WIDTH) / 2) / old_size
            center_y = self.offset_y + (self.screen_height / 2) / old_size
            
            self.offset_x = center_x - ((self.screen_width - SIDEBAR_WIDTH) / 2) / self.cell_size
            self.offset_y = center_y - (self.screen_height / 2) / self.cell_size
            
            # Ensure bounds
            self._constrain_offset()
            self._redraw_maze()
    
    def set_theme(self, theme_name):
        if theme_name in MazeTheme.get_theme_names():
            self.current_theme = theme_name
            self.theme = MazeTheme.get_theme(theme_name)
            self.update_status(f"Theme set to {theme_name}")
            self._redraw_maze()
    
    def reset_view(self):
        self.offset_x = 0
        self.offset_y = 0
        self.cell_size = DEFAULT_CELL_SIZE
        self._redraw_maze()
    
    def _constrain_offset(self):
        if not self.maze:
            return
        
        maze_pixel_width = self.maze_width * 2 + 1
        maze_pixel_height = self.maze_height * 2 + 1
        
        # Constrain X offset
        min_x = -5  # Allow padding
        max_x = max(maze_pixel_width - (self.screen_width - SIDEBAR_WIDTH) / self.cell_size + 5, min_x)
        self.offset_x = max(min_x, min(self.offset_x, max_x))
        
        # Constrain Y offset
        min_y = -5  # Allow padding
        max_y = max(maze_pixel_height - self.screen_height / self.cell_size + 5, min_y)
        self.offset_y = max(min_y, min(self.offset_y, max_y))
    
    def generate_maze(self):
        if self.generating or self.solving:
            return
        
        self.update_status("Generating maze...")
        self.generating = True
        self.show_solution = False
        self.current_step = 0
        
        # Create the maze
        self.maze = create_maze(
            self.maze_width, 
            self.maze_height, 
            self.algorithm,
            self.density,
            self.section_size
        )
        
        self.generation_steps = self.maze.generation_steps
        self.solver = MazeSolver(self.maze.maze)
        self.reset_view()
        
        self.update_status(f"Maze generated in {self.maze.generation_time:.3f} seconds!")
    
    def _finish_generation(self):
        self.generating = False
        self._redraw_maze()
        
        # Calculate complexity metrics
        path_length = len(self.solver.solve("astar"))
        complexity_score = path_length / (self.maze_width * self.maze_height)
        
        if complexity_score > 0.7:
            complexity_rating = "Extreme"
        elif complexity_score > 0.5:
            complexity_rating = "High"
        elif complexity_score > 0.3:
            complexity_rating = "Medium"
        else:
            complexity_rating = "Low"
            
        self.update_status(
            f"Maze generated with {self.algorithm.value}!\n"
            f"Size: {self.maze_width}x{self.maze_height}\n"
            f"Solution length: {path_length}\n"
            f"Complexity: {complexity_rating}"
        )
    
    def solve_maze(self):
        if not self.maze or self.generating or self.solving:
            return
        
        self.update_status("Solving maze...")
        self.solving = True
        self.show_solution = False
        self.current_step = 0
        
        # Solve with A*
        start_time = time.time()
        self.solution = self.solver.solve("astar")
        solve_time = time.time() - start_time
        
        self.solving_steps = self.solver.solving_steps
        
        if not self.solution:
            self.solving = False
            self.update_status("No solution found!")
            return
        
        self.update_status(f"Solution found! Length: {len(self.solution)}, Time: {solve_time:.3f}s")
    
    def _finish_solving(self):
        self.solving = False
        self.show_solution = True
        self._redraw_maze()
    
    def enable_path_drawing_mode(self):
        if self.generating or self.solving:
            self.update_status("Cannot start drawing while generating or solving")
            return
            
        self.drawing_mode = True
        self.user_path = []
        self.update_status("Draw your path! Click and drag from left edge to right edge.")
    
    def disable_path_drawing_mode(self):
        self.drawing_mode = False
        self.user_path = []
        self.update_status("Drawing mode canceled")
    
    def handle_path_drawing(self, mouse_pos):
        # Convert to grid coordinates
        canvas_rect = pygame.Rect(0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
        if not canvas_rect.collidepoint(mouse_pos):
            return
            
        cell_x = int(self.offset_x + mouse_pos[0] / self.cell_size)
        cell_y = int(self.offset_y + mouse_pos[1] / self.cell_size)
        
        # Ensure we're on a cell, not a wall
        if cell_x % 2 == 0 or cell_y % 2 == 0:
            return
        
        # Convert to maze cell coordinates
        grid_x = cell_x // 2
        grid_y = cell_y // 2
        
        # Add to path if valid
        if 0 <= grid_x < self.maze_width and 0 <= grid_y < self.maze_height:
            point = (grid_x, grid_y)
            if not self.user_path or point != self.user_path[-1]:
                self.user_path.append(point)
    
    def generate_maze_from_path(self):
        if not self.user_path or len(self.user_path) < 2:
            self.update_status("Path is too short. Please draw a longer path.")
            return
        
        # Verify path starts and ends correctly
        start_point = self.user_path[0]
        end_point = self.user_path[-1]
        
        valid_entry = start_point[0] == 0
        valid_exit = end_point[0] == self.maze_width - 1
        
        if not valid_entry or not valid_exit:
            self.update_status("Path must start at left edge and end at right edge.")
            return
        
        self.update_status("Generating maze from your path...")
        self.drawing_mode = False
        self.generating = True
        self.show_solution = False
        self.current_step = 0
        
        # Create custom path maze
        self.maze = CustomPathMaze(
            self.maze_width, 
            self.maze_height, 
            self.user_path,
            self.algorithm, 
            self.density, 
            self.section_size
        )
        
        self.maze.generate()
        self.generation_steps = self.maze.generation_steps
        self.solver = MazeSolver(self.maze.maze)
        self.reset_view()
        
        self.update_status(f"Maze generated with your custom path in {self.maze.generation_time:.3f} seconds!")
    
    def save_maze(self, format_type):
        if not self.maze:
            self.update_status("No maze to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "png":
            # Create surface for just the maze
            maze_width = (self.maze_width * 2 + 1) * self.cell_size
            maze_height = (self.maze_height * 2 + 1) * self.cell_size
            
            maze_surf = pygame.Surface((maze_width, maze_height))
            maze_surf.fill(self.theme["background"])
            
            # Save current settings
            temp_cell_size = self.cell_size
            temp_offset_x = self.offset_x
            temp_offset_y = self.offset_y
            
            # Use good export size
            self.cell_size = max(MIN_CELL_SIZE, min(MAX_CELL_SIZE, 20))
            self.offset_x = 0
            self.offset_y = 0
            
            # Draw maze and solution
            self._draw_maze(maze_surf, 0, 0, maze_width, maze_height)
            if self.show_solution and self.solution:
                self._draw_solution(maze_surf, 0, 0, maze_width, maze_height)
            
            # Restore settings
            self.cell_size = temp_cell_size
            self.offset_x = temp_offset_x
            self.offset_y = temp_offset_y
            
            # Save to file
            filename = f"maze_{self.algorithm.name}_{self.maze_width}x{self.maze_height}_{timestamp}.png"
            pygame.image.save(maze_surf, filename)
            self.update_status(f"Maze saved as {filename}")
            
        elif format_type == "txt":
            filename = f"maze_{self.algorithm.name}_{self.maze_width}x{self.maze_height}_{timestamp}.txt"
            self.maze.save_to_file(filename)
            self.update_status(f"Maze saved as {filename}")
    
    def update_status(self, message):
        self.status_message = message
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.VIDEORESIZE:
                self.screen_width = event.w
                self.screen_height = event.h
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self._init_ui()
                self._redraw_maze()
            
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
                
                # Update button hover state
                for button in self.buttons:
                    button["hover"] = button["rect"].collidepoint(event.pos)
                
                # Handle path drawing
                if self.drawing_mode and pygame.mouse.get_pressed()[0]:
                    self.handle_path_drawing(event.pos)
                
                # Handle pan dragging
                elif self.dragging and not self.drawing_mode:
                    dx = (event.pos[0] - self.drag_start[0]) / self.cell_size
                    dy = (event.pos[1] - self.drag_start[1]) / self.cell_size
                    self.offset_x -= dx
                    self.offset_y -= dy
                    self.drag_start = event.pos
                    self._constrain_offset()
                    self._redraw_maze()
                
                # Handle slider dragging
                elif self.active_element and "min" in self.active_element:
                    slider = self.active_element
                    slider_width = slider["rect"].width - 20
                    relative_x = max(0, min(slider_width, event.pos[0] - slider["rect"].x))
                    value = slider["min"] + (relative_x / slider_width) * (slider["max"] - slider["min"])
                    
                    slider["value"] = value
                    slider["handle_rect"].x = slider["rect"].x + relative_x
                    slider["action"](value)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # Check buttons
                    for button in self.buttons:
                        if button["rect"].collidepoint(event.pos):
                            button["action"]()
                            break
                    
                    # Check sliders
                    for slider in self.sliders:
                        if slider["handle_rect"].collidepoint(event.pos) or slider["rect"].collidepoint(event.pos):
                            self.active_element = slider
                            slider["active"] = True
                            
                            # Update slider position
                            slider_width = slider["rect"].width - 20
                            relative_x = max(0, min(slider_width, event.pos[0] - slider["rect"].x))
                            value = slider["min"] + (relative_x / slider_width) * (slider["max"] - slider["min"])
                            
                            slider["value"] = value
                            slider["handle_rect"].x = slider["rect"].x + relative_x
                            slider["action"](value)
                            break
                    
                    # Handle pan dragging
                    canvas_rect = pygame.Rect(0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
                    if canvas_rect.collidepoint(event.pos) and not self.active_element and not self.drawing_mode:
                        self.dragging = True
                        self.drag_start = event.pos
                
                elif event.button == 4:  # Mouse wheel up (zoom in)
                    self.set_cell_size(self.cell_size + 2)
                
                elif event.button == 5:  # Mouse wheel down (zoom out)
                    self.set_cell_size(self.cell_size - 2)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    self.dragging = False
                    
                    if self.active_element:
                        self.active_element["active"] = False
                        self.active_element = None
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.drawing_mode:
                        self.disable_path_drawing_mode()
                    else:
                        return False
                
                elif event.key == pygame.K_g and not self.drawing_mode:
                    self.generate_maze()
                
                elif event.key == pygame.K_s and not self.drawing_mode:
                    self.solve_maze()
                
                elif event.key == pygame.K_r and not self.drawing_mode:
                    self.reset_view()
                
                elif event.key == pygame.K_d:
                    # Toggle drawing mode
                    if self.drawing_mode:
                        self.disable_path_drawing_mode()
                    else:
                        self.enable_path_drawing_mode()
                
                elif event.key == pygame.K_RETURN and self.drawing_mode:
                    self.generate_maze_from_path()
                
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.set_cell_size(self.cell_size + 2)
                
                elif event.key == pygame.K_MINUS:
                    self.set_cell_size(self.cell_size - 2)
                
                elif event.key in [pygame.K_UP, pygame.K_w]:
                    self.offset_y -= 5
                    self._constrain_offset()
                    self._redraw_maze()
                
                elif event.key in [pygame.K_DOWN, pygame.K_s]:
                    self.offset_y += 5
                    self._constrain_offset()
                    self._redraw_maze()
                
                elif event.key in [pygame.K_LEFT, pygame.K_a]:
                    self.offset_x -= 5
                    self._constrain_offset()
                    self._redraw_maze()
                
                elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                    self.offset_x += 5
                    self._constrain_offset()
                    self._redraw_maze()
        
        return True
    
    def _redraw_maze(self):
        if not self.maze:
            return
        
        self.maze_surface = pygame.Surface((self.screen_width - SIDEBAR_WIDTH, self.screen_height))
        self.maze_surface.fill(self.theme["background"])
        
        self._draw_maze(self.maze_surface, 0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
        
        if self.show_solution and self.solution:
            self._draw_solution(self.maze_surface, 0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
    
    def _get_gradient_color(self, colors, position):
        if len(colors) == 1:
            return colors[0]
        
        if position <= 0:
            return colors[0]
        if position >= 1:
            return colors[-1]
        
        segment_count = len(colors) - 1
        segment_length = 1.0 / segment_count
        segment_index = int(position * segment_count)
        segment_position = (position - segment_index * segment_length) / segment_length
        
        color1 = colors[segment_index]
        color2 = colors[segment_index + 1]
        
        r = int(color1[0] + segment_position * (color2[0] - color1[0]))
        g = int(color1[1] + segment_position * (color2[1] - color1[1]))
        b = int(color1[2] + segment_position * (color2[2] - color1[2]))
        
        return (r, g, b)
    
    def _draw_maze(self, surface, x, y, width, height):
        if not self.maze:
            return
        
        # Choose maze data based on state
        if self.generating and self.current_step < len(self.generation_steps):
            maze_data = self.generation_steps[self.current_step]
        elif self.solving and self.current_step < len(self.solving_steps):
            maze_data = self.solving_steps[self.current_step][0]
        else:
            maze_data = self.maze.maze
        
        # Calculate visible range
        start_x = max(0, int(self.offset_x))
        start_y = max(0, int(self.offset_y))
        end_x = min(len(maze_data[0]), int(self.offset_x + width / self.cell_size + 1))
        end_y = min(len(maze_data), int(self.offset_y + height / self.cell_size + 1))
        
        # Draw cells
        for y_cell in range(start_y, end_y):
            for x_cell in range(start_x, end_x):
                if y_cell >= len(maze_data) or x_cell >= len(maze_data[0]):
                    continue
                
                # Gradient position
                gradient_pos = (x_cell / len(maze_data[0]) + y_cell / len(maze_data)) / 2
                
                cell_x = int((x_cell - self.offset_x) * self.cell_size)
                cell_y = int((y_cell - self.offset_y) * self.cell_size)
                cell_type = maze_data[y_cell][x_cell]
                
                if cell_type == '#':  # Wall
                    color = self._get_gradient_color(self.theme["wall"], gradient_pos)
                    pygame.draw.rect(surface, color, (cell_x, cell_y, self.cell_size, self.cell_size))
                    
                    # 3D effect
                    shadow_rect = pygame.Rect(cell_x, cell_y, self.cell_size, self.cell_size)
                    lighter = self._get_gradient_color(self.theme["wall"], min(1, gradient_pos + 0.1))
                    darker = self._get_gradient_color(self.theme["wall"], max(0, gradient_pos - 0.1))
                    
                    # Top and left lighter
                    pygame.draw.line(surface, lighter, 
                                   (shadow_rect.left, shadow_rect.top), 
                                   (shadow_rect.right, shadow_rect.top))
                    pygame.draw.line(surface, lighter, 
                                   (shadow_rect.left, shadow_rect.top), 
                                   (shadow_rect.left, shadow_rect.bottom))
                    
                    # Bottom and right darker
                    pygame.draw.line(surface, darker, 
                                   (shadow_rect.left, shadow_rect.bottom-1), 
                                   (shadow_rect.right, shadow_rect.bottom-1))
                    pygame.draw.line(surface, darker, 
                                   (shadow_rect.right-1, shadow_rect.top), 
                                   (shadow_rect.right-1, shadow_rect.bottom))
                
                elif cell_type == ' ':  # Path
                    # Special handling for entrance/exit
                    if x_cell == 0 and y_cell % 2 == 1:  # Entrance
                        color = self.theme["entrance"]
                    elif x_cell == len(maze_data[0])-1 and y_cell % 2 == 1:  # Exit
                        color = self.theme["exit"]
                    else:
                        color = self._get_gradient_color(self.theme["cell"], gradient_pos)
                    
                    pygame.draw.rect(surface, color, (cell_x, cell_y, self.cell_size, self.cell_size))
                
                elif cell_type in ['C', 'B', 'H', 'X']:  # Visualization markers
                    marker_colors = {
                        'C': self.theme["current"],      # Current cell
                        'B': (255, 165, 0),              # Backtracking
                        'H': (0, 191, 255),              # Highlight
                        'X': (255, 0, 0)                 # Solving path
                    }
                    
                    # Draw cell background
                    pygame.draw.rect(surface, self._get_gradient_color(self.theme["cell"], gradient_pos), 
                                   (cell_x, cell_y, self.cell_size, self.cell_size))
                    
                    # Draw marker
                    marker_color = marker_colors.get(cell_type, self.theme["highlight"])
                    center_x = cell_x + self.cell_size // 2
                    center_y = cell_y + self.cell_size // 2
                    
                    if cell_type == 'C':  # Circle for current
                        pygame.draw.circle(surface, marker_color, (center_x, center_y), self.cell_size // 3)
                    elif cell_type == 'B':  # Diamond for backtracking
                        points = [
                            (center_x, cell_y + self.cell_size // 4),
                            (cell_x + 3 * self.cell_size // 4, center_y),
                            (center_x, cell_y + 3 * self.cell_size // 4),
                            (cell_x + self.cell_size // 4, center_y)
                        ]
                        pygame.draw.polygon(surface, marker_color, points)
                    elif cell_type == 'H':  # Square for highlighted cell
                        pygame.draw.rect(surface, marker_color, 
                                       (cell_x + self.cell_size // 4, cell_y + self.cell_size // 4, 
                                        self.cell_size // 2, self.cell_size // 2))
                    else:  # X for solving path
                        pygame.draw.line(surface, marker_color, 
                                       (cell_x + self.cell_size // 4, cell_y + self.cell_size // 4),
                                       (cell_x + 3 * self.cell_size // 4, cell_y + 3 * self.cell_size // 4), 2)
                        pygame.draw.line(surface, marker_color, 
                                       (cell_x + 3 * self.cell_size // 4, cell_y + self.cell_size // 4),
                                       (cell_x + self.cell_size // 4, cell_y + 3 * self.cell_size // 4), 2)
    
    def _draw_solution(self, surface, x, y, width, height):
        if not self.solution:
            return
        
        # Draw solution path
        for i in range(1, len(self.solution)):
            x1, y1 = self.solution[i-1]
            x2, y2 = self.solution[i]
            
            # Calculate screen positions
            screen_x1 = int((x1 - self.offset_x) * self.cell_size + self.cell_size // 2)
            screen_y1 = int((y1 - self.offset_y) * self.cell_size + self.cell_size // 2)
            screen_x2 = int((x2 - self.offset_x) * self.cell_size + self.cell_size // 2)
            screen_y2 = int((y2 - self.offset_y) * self.cell_size + self.cell_size // 2)
            
            # Draw line with gradient
            pos = i / len(self.solution)
            color = self._get_gradient_color(self.theme["solution"], pos)
            line_width = max(2, self.cell_size // 4)
            
            pygame.draw.line(surface, color, (screen_x1, screen_y1), (screen_x2, screen_y2), line_width)
            
            # Add dots at vertices
            dot_radius = max(2, line_width // 2 + 1)
            pygame.draw.circle(surface, color, (screen_x1, screen_y1), dot_radius)
            pygame.draw.circle(surface, color, (screen_x2, screen_y2), dot_radius)
    
    def _draw_ui(self):
        # Sidebar background
        sidebar_rect = pygame.Rect(self.screen_width - SIDEBAR_WIDTH, 0, SIDEBAR_WIDTH, self.screen_height)
        
        # Draw gradient sidebar
        for y in range(0, self.screen_height, 2):
            progress = y / self.screen_height
            bg_color = self.theme["background"]
            color = self._get_gradient_color([
                (int(bg_color[0] * 1.5), int(bg_color[1] * 1.5), int(bg_color[2] * 1.5)),
                (int(bg_color[0] * 1.8), int(bg_color[1] * 1.8), int(bg_color[2] * 1.8))
            ], progress)
            pygame.draw.line(self.screen, color, (sidebar_rect.left, y), (sidebar_rect.right, y))
        
        # Separator line
        pygame.draw.line(self.screen, self.theme["wall"][0], 
                       (sidebar_rect.left, 0), (sidebar_rect.left, self.screen_height), 2)
        
        # Title
        title_text = "Magnificent Maze Generator"
        title_surf = self.font_title.render(title_text, True, self.theme["text"])
        title_rect = title_surf.get_rect(center=self.title_rect.center)
        self.screen.blit(title_surf, title_rect)
        
        # Draw buttons
        for button in self.buttons:
            color = self.theme["button_hover"] if button["hover"] else self.theme["button"]
            pygame.draw.rect(self.screen, color, button["rect"], border_radius=5)
            pygame.draw.rect(self.screen, self.theme["text"], button["rect"], width=1, border_radius=5)
            
            text_surf = self.font.render(button["text"], True, self.theme["text"])
            text_rect = text_surf.get_rect(center=button["rect"].center)
            self.screen.blit(text_surf, text_rect)
        
        # Draw sliders
        for slider in self.sliders:
            # Label
            label_surf = self.font.render(f"{slider['text']}: {slider['value']:.1f}", True, self.theme["text"])
            label_rect = pygame.Rect(slider["rect"].x, slider["rect"].y - 20, slider["rect"].width, 20)
            self.screen.blit(label_surf, label_rect)
            
            # Track
            track_rect = pygame.Rect(slider["rect"].x, slider["rect"].y + slider["rect"].height // 2 - 2,
                                   slider["rect"].width, 4)
            pygame.draw.rect(self.screen, self.theme["slider"], track_rect, border_radius=2)
            
            # Handle
            pygame.draw.rect(self.screen, self.theme["slider_handle"], slider["handle_rect"], border_radius=5)
            pygame.draw.rect(self.screen, self.theme["text"], slider["handle_rect"], width=1, border_radius=5)
        
        # Theme label
        theme_label = self.font.render("Theme:", True, self.theme["text"])
        theme_label_rect = pygame.Rect(self.theme_rect.x, self.theme_rect.y - 20, self.theme_rect.width, 20)
        self.screen.blit(theme_label, theme_label_rect)
        
        # Current algorithm
        algo_text = f"Algorithm: {self.algorithm.value}"
        algo_surf = self.font.render(algo_text, True, self.theme["text"])
        algo_rect = pygame.Rect(sidebar_rect.left + 10, 15, SIDEBAR_WIDTH - 20, 20)
        self.screen.blit(algo_surf, algo_rect)
        
        # Draw mode indicator
        if self.drawing_mode:
            mode_text = "MODE: DRAWING PATH"
            mode_color = (255, 100, 100)  # Red-ish color
        else:
            mode_text = "MODE: NORMAL"
            mode_color = self.theme["text"]
            
        mode_surf = self.font_large.render(mode_text, True, mode_color)
        mode_rect = mode_surf.get_rect(topright=(self.screen_width - 10, 10))
        self.screen.blit(mode_surf, mode_rect)
        
        # Draw path instructions if in drawing mode
        if self.drawing_mode:
            instructions = [
                "Click and drag to draw your path",
                "Path must connect left to right edge",
                "Press Enter to generate maze from path",
                "Press Escape to cancel"
            ]
            
            for i, line in enumerate(instructions):
                instr_surf = self.font.render(line, True, mode_color)
                instr_rect = instr_surf.get_rect(topright=(self.screen_width - 10, 40 + i * 25))
                
                # Background for readability
                bg_rect = instr_rect.inflate(10, 6)
                bg_surf = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                bg_surf.fill((0, 0, 0, 150))
                self.screen.blit(bg_surf, bg_rect)
                
                self.screen.blit(instr_surf, instr_rect)
        
        # Status message
        status_lines = self.status_message.split('\n')
        for i, line in enumerate(status_lines):
            status_surf = self.font.render(line, True, self.theme["text"])
            status_rect = status_surf.get_rect(topleft=(self.status_rect.x, self.status_rect.y + i * 20))
            self.screen.blit(status_surf, status_rect)
        
        # Coordinates on hover
        canvas_rect = pygame.Rect(0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
        if canvas_rect.collidepoint(self.mouse_pos):
            maze_x = int(self.offset_x + self.mouse_pos[0] / self.cell_size)
            maze_y = int(self.offset_y + self.mouse_pos[1] / self.cell_size)
            
            coord_text = f"Coordinates: ({maze_x}, {maze_y})"
            coord_surf = self.font.render(coord_text, True, self.theme["text"])
            coord_rect = coord_surf.get_rect(bottomright=(self.screen_width - SIDEBAR_WIDTH - 10, 
                                                        self.screen_height - 10))
            
            # Background for better visibility
            bg_rect = coord_rect.inflate(10, 6)
            bg_surf = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 150))
            self.screen.blit(bg_surf, bg_rect)
            self.screen.blit(coord_surf, coord_rect)
        
        # Help text
        help_text = "G=Generate | S=Solve | R=Reset | D=Draw | Arrows=Pan | +/-=Zoom"
        help_surf = self.font.render(help_text, True, self.theme["text"])
        help_rect = help_surf.get_rect(midbottom=(self.screen_width - SIDEBAR_WIDTH // 2, 
                                                self.screen_height - 10))
        self.screen.blit(help_surf, help_rect)
        
        # Draw user path if in drawing mode
        if self.drawing_mode and self.user_path:
            for i in range(1, len(self.user_path)):
                x1, y1 = self.user_path[i-1]
                x2, y2 = self.user_path[i]
                
                # Convert to screen coordinates
                screen_x1 = int((x1 * 2 + 1 - self.offset_x) * self.cell_size + self.cell_size // 2)
                screen_y1 = int((y1 * 2 + 1 - self.offset_y) * self.cell_size + self.cell_size // 2)
                screen_x2 = int((x2 * 2 + 1 - self.offset_x) * self.cell_size + self.cell_size // 2)
                screen_y2 = int((y2 * 2 + 1 - self.offset_y) * self.cell_size + self.cell_size // 2)
                
                # Draw line with gradient color
                pos = i / len(self.user_path)
                color = self._get_gradient_color([(255, 50, 50), (255, 200, 50)], pos)
                line_width = max(3, self.cell_size // 3)
                
                pygame.draw.line(self.screen, color, (screen_x1, screen_y1), (screen_x2, screen_y2), line_width)
                
                # Add dots
                dot_radius = max(3, line_width // 2 + 2)
                pygame.draw.circle(self.screen, color, (screen_x1, screen_y1), dot_radius)
                pygame.draw.circle(self.screen, color, (screen_x2, screen_y2), dot_radius)
    
    def update(self):
        # Handle maze generation animation
        if self.generating:
            if self.current_step < len(self.generation_steps) - 1:
                self.current_step = min(self.current_step + self.animation_speed, 
                                     len(self.generation_steps) - 1)
            else:
                self._finish_generation()
        
        # Handle maze solving animation
        elif self.solving:
            if self.current_step < len(self.solving_steps) - 1:
                self.current_step = min(self.current_step + self.animation_speed, 
                                     len(self.solving_steps) - 1)
            else:
                self._finish_solving()
    
    def draw(self):
        # Clear the screen
        self.screen.fill(self.theme["background"])
        
        # Draw the maze
        if self.maze_surface:
            self.screen.blit(self.maze_surface, (0, 0))
        elif self.maze:
            self._redraw_maze()
            self.screen.blit(self.maze_surface, (0, 0))
        
        # Draw UI elements
        self._draw_ui()
        
        # Update the display
        pygame.display.flip()
    
    def run(self):
        running = True
        
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Magnificent Maze Generator")
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH, 
                      help=f'Window width (default: {DEFAULT_WIDTH})')
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT, 
                      help=f'Window height (default: {DEFAULT_HEIGHT})')
    args = parser.parse_args()
    
    # Create and run the UI
    ui = MazeUI(args.width, args.height)
    ui.run()


if __name__ == "__main__":
    main()
