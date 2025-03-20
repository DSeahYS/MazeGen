#!/usr/bin/env python3
"""
Magnificent Maze Generator - A visually stunning maze generator and solver
with high-complexity algorithms, beautiful visualization, and custom path drawing.

This implementation includes:
- Multiple algorithms optimized for maximum complexity
- Beautiful animated maze generation and solving
- Various visual themes with gradient effects
- Interactive controls for all parameters
- User-drawn path feature for custom maze design
- Zoom and pan capabilities
- Export functionality
"""

import pygame
import sys
import os
import random
import time
import argparse
from typing import List, Tuple, Dict, Any, Optional, Set, Union
import math
import colorsys
from datetime import datetime
from enum import Enum

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
    """Enumeration of available maze generation algorithms."""
    ELLERS = "Eller's Algorithm"
    BACKTRACKER = "Recursive Backtracker"
    HYBRID = "Hybrid Algorithm"
    CUSTOM_PATH = "Custom Path Algorithm"


class BaseMaze:
    """Base class for maze generators."""
    
    def __init__(self, width: int, height: int):
        """Initialize the maze generator with the given dimensions."""
        self.width = width
        self.height = height
        self.maze_width = 2 * width + 1
        self.maze_height = 2 * height + 1
        self.maze = [['#' for _ in range(self.maze_width)] for _ in range(self.maze_height)]
        self.generation_time = 0
        self.generation_steps = []
        
    def generate(self) -> List[List[str]]:
        """Generate the maze (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def get_cell(self, x: int, y: int) -> str:
        """Get the value of a cell in the maze."""
        return self.maze[y][x]
    
    def set_cell(self, x: int, y: int, value: str) -> None:
        """Set the value of a cell in the maze."""
        self.maze[y][x] = value
        
    def create_entrance_exit(self) -> None:
        """Create entrance and exit for the maze."""
        self.maze[1][0] = ' '  # Entrance at top-left
        self.maze[self.maze_height-2][self.maze_width-1] = ' '  # Exit at bottom-right
    
    def save_to_file(self, filename: str) -> None:
        """Save the maze to a text file."""
        with open(filename, 'w') as f:
            for row in self.maze:
                f.write(''.join(row) + '\n')


class EllerMaze(BaseMaze):
    """Eller's Algorithm maze generator optimized for maximum complexity."""
    
    def __init__(self, width: int, height: int, density: float = 0.5):
        """Initialize the maze generator with the given dimensions."""
        super().__init__(width, height)
        self.density = max(0.1, min(0.9, density))
        
    def generate(self) -> List[List[str]]:
        """Generate maze using Eller's Algorithm."""
        start_time = time.time()
        
        # Initialize with a grid of all walls
        self.maze = [['#' for _ in range(self.maze_width)] for _ in range(self.maze_height)]
        self.generation_steps = []
        
        # Process each row
        row = [i for i in range(self.width)]  # Each cell starts in its own set
        next_set = self.width
        
        for y in range(self.height):
            # Set cells in the current row
            for x in range(self.width):
                self.maze[2*y+1][2*x+1] = ' '
            
            # Save current state for animation
            self.generation_steps.append([row[:] for row in self.maze])
            
            # Randomly connect adjacent cells in the same row
            for x in range(self.width - 1):
                # Only connect if cells are in different sets
                if row[x] != row[x+1] and random.random() < self.density:
                    # Remove the wall between cells
                    self.maze[2*y+1][2*x+2] = ' '
                    
                    # Merge sets (replace all occurrences of set[x+1] with set[x])
                    old_set = row[x+1]
                    new_set = row[x]
                    for i in range(self.width):
                        if row[i] == old_set:
                            row[i] = new_set
                    
                    # Save state after each horizontal connection
                    self.generation_steps.append([row[:] for row in self.maze])
            
            # Last row connects all different sets horizontally
            if y == self.height - 1:
                for x in range(self.width - 1):
                    if row[x] != row[x+1]:
                        # Remove the wall between cells
                        self.maze[2*y+1][2*x+2] = ' '
                        # Save state
                        self.generation_steps.append([row[:] for row in self.maze])
                continue
            
            # Group cells by set
            sets = {}
            for x, set_id in enumerate(row):
                if set_id not in sets:
                    sets[set_id] = []
                sets[set_id].append(x)
            
            # Initialize the next row with all cells in their own set
            next_row = [-1] * self.width
            
            # For each set, randomly connect some cells to the row below
            for set_id, cells in sets.items():
                # Maximize complexity by connecting just enough cells vertically
                vertical_density = max(1, int(len(cells) * (1 - self.density) + 0.5))
                connect_count = max(1, min(len(cells), vertical_density))
                cells_to_connect = random.sample(cells, connect_count)
                
                for x in cells_to_connect:
                    # Remove the wall below the cell
                    self.maze[2*y+2][2*x+1] = ' '
                    
                    # Pass the set ID to the cell below
                    next_row[x] = set_id
                    
                    # Save state after each vertical connection
                    self.generation_steps.append([row[:] for row in self.maze])
            
            # Assign new set IDs to unconnected cells in the next row
            for x in range(self.width):
                if next_row[x] == -1:
                    next_row[x] = next_set
                    next_set += 1
            
            # Update the current row for the next iteration
            row = next_row
        
        self.create_entrance_exit()
        
        # Save final state
        self.generation_steps.append([row[:] for row in self.maze])
        
        self.generation_time = time.time() - start_time
        return self.maze


class RecursiveBacktracker(BaseMaze):
    """
    Recursive Backtracker (Depth-First Search) maze generator.
    Creates mazes with long, winding corridors and high complexity.
    """
    
    def generate(self) -> List[List[str]]:
        """Generate maze using Recursive Backtracker Algorithm."""
        start_time = time.time()
        
        # Initialize with a grid of all walls
        self.maze = [['#' for _ in range(self.maze_width)] for _ in range(self.maze_height)]
        self.generation_steps = []
        
        # Make all cells in grid
        for y in range(self.height):
            for x in range(self.width):
                self.maze[2*y+1][2*x+1] = ' '
        
        # Save initial state
        self.generation_steps.append([row[:] for row in self.maze])
        
        # Execute recursive backtracker algorithm
        stack = []
        visited = set()
        
        # Start at a random cell
        start_x = random.randint(0, self.width - 1)
        start_y = random.randint(0, self.height - 1)
        stack.append((start_x, start_y))
        visited.add((start_x, start_y))
        
        while stack:
            # Get the current cell
            x, y = stack[-1]
            
            # Mark current cell for visualization
            current_maze = [row[:] for row in self.maze]
            current_maze[2*y+1][2*x+1] = 'C'  # Mark current cell
            self.generation_steps.append(current_maze)
            
            # Find unvisited neighbors
            neighbors = []
            # Check all four directions
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # North, East, South, West
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                # Choose a random unvisited neighbor
                nx, ny, dx, dy = random.choice(neighbors)
                
                # Remove the wall between current cell and chosen neighbor
                # Wall position is between cells, so it's at 2*y+1+dy, 2*x+1+dx
                self.maze[2*y+1+dy][2*x+1+dx] = ' '
                
                # Push the chosen cell to the stack and mark as visited
                stack.append((nx, ny))
                visited.add((nx, ny))
            else:
                # Backtrack if no unvisited neighbors
                stack.pop()
                
                # Save state after backtracking
                if stack:
                    x, y = stack[-1]
                    current_maze = [row[:] for row in self.maze]
                    current_maze[2*y+1][2*x+1] = 'B'  # Mark backtracking cell
                    self.generation_steps.append(current_maze)
        
        # Clean up temporary cell markers
        for y in range(self.height):
            for x in range(self.width):
                if self.maze[2*y+1][2*x+1] in ['C', 'B']:
                    self.maze[2*y+1][2*x+1] = ' '
        
        self.create_entrance_exit()
        
        # Save final state
        self.generation_steps.append([row[:] for row in self.maze])
        
        self.generation_time = time.time() - start_time
        return self.maze


class HybridMaze(BaseMaze):
    """
    Hybrid maze generator combining Eller's Algorithm and Recursive Backtracker 
    for maximum complexity. Eller's creates the overall structure, then sections
    are reworked using Recursive Backtracker for more winding passages.
    """
    
    def __init__(self, width: int, height: int, density: float = 0.6, section_size: int = 5):
        """Initialize the hybrid maze generator."""
        super().__init__(width, height)
        self.density = density
        self.section_size = section_size
        
    def generate(self) -> List[List[str]]:
        """Generate hybrid maze combining multiple algorithms."""
        start_time = time.time()
        self.generation_steps = []
        
        # Generate base maze using Eller's Algorithm
        eller = EllerMaze(self.width, self.height, self.density)
        self.maze = eller.generate()
        
        # Capture initial state from Eller's algorithm
        self.generation_steps = eller.generation_steps
        
        # Apply recursive backtracker to random sections for additional complexity
        sections_x = self.width // self.section_size
        sections_y = self.height // self.section_size
        
        for _ in range(max(1, min(10, sections_x * sections_y // 3))):
            # Choose a random section
            section_x = random.randint(0, max(0, sections_x - 1))
            section_y = random.randint(0, max(0, sections_y - 1))
            
            # Apply recursive backtracker to the section
            start_x = section_x * self.section_size
            start_y = section_y * self.section_size
            end_x = min(start_x + self.section_size, self.width)
            end_y = min(start_y + self.section_size, self.height)
            
            # Highlight the section being reworked
            highlight_maze = [row[:] for row in self.maze]
            for y in range(start_y, end_y):
                for x in range(start_x, end_x):
                    if highlight_maze[2*y+1][2*x+1] == ' ':
                        highlight_maze[2*y+1][2*x+1] = 'H'  # Highlight cell
            self.generation_steps.append(highlight_maze)
            
            self._rework_section(start_x, start_y, end_x, end_y)
        
        # Ensure there's a path from entrance to exit
        self.create_entrance_exit()
        
        # Save final state
        self.generation_steps.append([row[:] for row in self.maze])
        
        self.generation_time = time.time() - start_time
        return self.maze
    
    def _rework_section(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        """Rework a section of the maze using recursive backtracker."""
        # Reset the section to walls except for the cells
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                # Keep the cells, reset the walls
                self.maze[2*y+1][2*x+1] = ' '
                if x < end_x - 1:
                    self.maze[2*y+1][2*x+2] = '#'
                if y < end_y - 1:
                    self.maze[2*y+2][2*x+1] = '#'
        
        # Save state after section reset
        self.generation_steps.append([row[:] for row in self.maze])
        
        # Apply recursive backtracker to the section
        stack = []
        visited = set()
        
        # Start at a random cell within the section
        x = random.randint(start_x, end_x - 1)
        y = random.randint(start_y, end_y - 1)
        stack.append((x, y))
        visited.add((x, y))
        
        while stack:
            x, y = stack[-1]
            
            # Mark current cell for visualization
            current_maze = [row[:] for row in self.maze]
            current_maze[2*y+1][2*x+1] = 'C'  # Mark current cell
            self.generation_steps.append(current_maze)
            
            # Find unvisited neighbors within the section
            neighbors = []
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if start_x <= nx < end_x and start_y <= ny < end_y and (nx, ny) not in visited:
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                self.maze[2*y+1+dy][2*x+1+dx] = ' '
                stack.append((nx, ny))
                visited.add((nx, ny))
                
                # Save state after carving a passage
                self.generation_steps.append([row[:] for row in self.maze])
            else:
                stack.pop()
                
                # Save state after backtracking
                if stack:
                    x, y = stack[-1]
                    current_maze = [row[:] for row in self.maze]
                    current_maze[2*y+1][2*x+1] = 'B'  # Mark backtracking cell
                    self.generation_steps.append(current_maze)
        
        # Clean up temporary cell markers
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                if self.maze[2*y+1][2*x+1] in ['C', 'B', 'H']:
                    self.maze[2*y+1][2*x+1] = ' '


class CustomPathMaze(BaseMaze):
    """
    Custom Path Maze generator that builds a maze around a user-drawn path.
    The user path becomes the backbone of the maze, and the algorithm fills
    in the rest of the maze around it.
    """
    
    def __init__(self, width: int, height: int, user_path: List[Tuple[int, int]], density: float = 0.6):
        """Initialize the custom path maze generator."""
        super().__init__(width, height)
        self.user_path = user_path
        self.density = density
        
    def generate(self) -> List[List[str]]:
        """Generate maze around the user-drawn path."""
        start_time = time.time()
        
        # Initialize with a grid of all walls
        self.maze = [['#' for _ in range(self.maze_width)] for _ in range(self.maze_height)]
        self.generation_steps = []
        
        # Make all cells in grid
        for y in range(self.height):
            for x in range(self.width):
                self.maze[2*y+1][2*x+1] = ' '
        
        # Save initial state
        self.generation_steps.append([row[:] for row in self.maze])
        
        # First, create the user path
        if self.user_path and len(self.user_path) > 1:
            self._carve_user_path()
        
        # Next, use recursive backtracker to fill the rest of the maze
        self._fill_remaining_maze()
        
        # Ensure entrance and exit
        self.create_entrance_exit()
        
        # Save final state
        self.generation_steps.append([row[:] for row in self.maze])
        
        self.generation_time = time.time() - start_time
        return self.maze
    
    def _carve_user_path(self) -> None:
        """Carve the user-drawn path in the maze."""
        # Highlight the path cells
        path_maze = [row[:] for row in self.maze]
        for x, y in self.user_path:
            path_maze[2*y+1][2*x+1] = 'P'  # Mark path cell
        self.generation_steps.append(path_maze)
        
        # Carve passages between adjacent path cells
        for i in range(len(self.user_path) - 1):
            x1, y1 = self.user_path[i]
            x2, y2 = self.user_path[i+1]
            
            # Check if cells are adjacent
            if abs(x1 - x2) + abs(y1 - y2) == 1:
                # Cells are adjacent, remove the wall between them
                wall_x = x1 + (x2 - x1) // 2
                wall_y = y1 + (y2 - y1) // 2
                self.maze[2*wall_y+1][2*wall_x+1] = ' '
                
                # For diagonal cells, we need to choose which wall to remove
                # This is a simplified approach; we just make a direct connection
            else:
                # For non-adjacent cells, find a path between them
                self._connect_nonadjacent_cells(x1, y1, x2, y2)
            
            # Save state after carving each connection
            path_maze = [row[:] for row in self.maze]
            for x, y in self.user_path:
                path_maze[2*y+1][2*x+1] = 'P'  # Mark path cell
            self.generation_steps.append(path_maze)
    
    def _connect_nonadjacent_cells(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Connect two non-adjacent cells with a path."""
        # Use a simple approach: move horizontally, then vertically
        x, y = x1, y1
        
        # Move horizontally
        while x != x2:
            nx = x + (1 if x < x2 else -1)
            # Remove the wall between cells
            wall_x = x + (nx - x) // 2
            self.maze[2*y+1][2*wall_x+2] = ' '
            x = nx
        
        # Move vertically
        while y != y2:
            ny = y + (1 if y < y2 else -1)
            # Remove the wall between cells
            wall_y = y + (ny - y) // 2
            self.maze[2*wall_y+2][2*x+1] = ' '
            y = ny
    
    def _fill_remaining_maze(self) -> None:
        """Fill the rest of the maze using Recursive Backtracker algorithm."""
        # Mark all cells that are part of the user path as visited
        visited = set(self.user_path)
        
        # Keep track of the frontier - cells adjacent to the path
        frontier = set()
        for x, y in visited:
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    (nx, ny) not in visited):
                    frontier.add((nx, ny))
        
        # Process frontier cells
        while frontier:
            # Choose a random frontier cell
            x, y = random.choice(list(frontier))
            frontier.remove((x, y))
            visited.add((x, y))
            
            # Find neighbors that are already in the maze
            neighbors = []
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    (nx, ny) in visited):
                    neighbors.append((nx, ny, dx, dy))
            
            # Connect to a random neighbor that's already in the maze
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                # Remove the wall between cells
                self.maze[2*y+1+dy//2][2*x+1+dx//2] = ' '
                
                # Mark for visualization
                current_maze = [row[:] for row in self.maze]
                current_maze[2*y+1][2*x+1] = 'C'  # Mark current cell
                self.generation_steps.append(current_maze)
            
            # Add new frontier cells
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    (nx, ny) not in visited and (nx, ny) not in frontier):
                    frontier.add((nx, ny))
        
        # Fill any remaining cells using recursive backtracker
        self._fill_isolated_cells(visited)
    
    def _fill_isolated_cells(self, visited: Set[Tuple[int, int]]) -> None:
        """Fill any isolated cells that weren't reached yet."""
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in visited:
                    # Use a mini recursive backtracker to fill this region
                    stack = [(x, y)]
                    region_visited = {(x, y)}
                    
                    while stack:
                        cx, cy = stack[-1]
                        
                        # Find unvisited neighbors within this isolated region
                        neighbors = []
                        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                            nx, ny = cx + dx, cy + dy
                            if (0 <= nx < self.width and 0 <= ny < self.height and 
                                (nx, ny) not in visited and (nx, ny) not in region_visited):
                                neighbors.append((nx, ny, dx, dy))
                        
                        if neighbors:
                            nx, ny, dx, dy = random.choice(neighbors)
                            # Remove wall between cells
                            self.maze[2*cy+1+dy//2][2*cx+1+dx//2] = ' '
                            stack.append((nx, ny))
                            region_visited.add((nx, ny))
                        else:
                            stack.pop()
                    
                    # Connect this region to the main maze
                    edge_cells = []
                    for rx, ry in region_visited:
                        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                            nx, ny = rx + dx, ry + dy
                            if (0 <= nx < self.width and 0 <= ny < self.height and 
                                (nx, ny) in visited):
                                edge_cells.append((rx, ry, nx, ny))
                    
                    if edge_cells:
                        rx, ry, nx, ny = random.choice(edge_cells)
                        # Remove wall between the region and the main maze
                        wall_x = rx + (nx - rx) // 2
                        wall_y = ry + (ny - ry) // 2
                        self.maze[2*wall_y+1][2*wall_x+1] = ' '
                    
                    # Update visited cells
                    visited.update(region_visited)


def create_maze(width: int, height: int, algorithm: MazeAlgorithm, 
                density: float = 0.5, section_size: int = 5,
                user_path: List[Tuple[int, int]] = None) -> BaseMaze:
    """Create a maze with the specified algorithm and parameters."""
    if algorithm == MazeAlgorithm.ELLERS:
        generator = EllerMaze(width, height, density)
    elif algorithm == MazeAlgorithm.BACKTRACKER:
        generator = RecursiveBacktracker(width, height)
    elif algorithm == MazeAlgorithm.CUSTOM_PATH and user_path:
        generator = CustomPathMaze(width, height, user_path, density)
    else:  # Hybrid (default) or any other case
        generator = HybridMaze(width, height, density, section_size)
    
    generator.generate()
    return generator


# --- Maze Solving ---

class MazeSolver:
    """Class to solve mazes using various algorithms."""
    
    def __init__(self, maze: List[List[str]]):
        """Initialize the solver with a maze."""
        self.maze = maze
        self.height = len(maze)
        self.width = len(maze[0]) if self.height > 0 else 0
        self.solution = []
        self.visited = set()
        self.solving_steps = []
    
    def solve(self, algorithm: str = "astar") -> List[Tuple[int, int]]:
        """Solve the maze using the specified algorithm."""
        # Find entrance and exit
        entrance = None
        exit = None
        
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
        
        # Reset solver state
        self.solution = []
        self.visited = set()
        self.solving_steps = []
        
        # Choose algorithm
        if algorithm == "astar":
            return self._solve_astar(entrance, exit)
        elif algorithm == "dfs":
            return self._solve_dfs(entrance, exit)
        elif algorithm == "bfs":
            return self._solve_bfs(entrance, exit)
        else:
            return self._solve_astar(entrance, exit)  # Default to A*
    
    def _solve_dfs(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Solve the maze using depth-first search."""
        stack = [(start, [start])]
        self.visited = set([start])
        
        while stack:
            (x, y), path = stack.pop()
            
            # Record step for visualization
            current_maze = [row[:] for row in self.maze]
            current_maze[y][x] = 'X'  # Mark current position
            self.solving_steps.append((current_maze, path))
            
            if (x, y) == end:
                self.solution = path
                return path
            
            # Check all four directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Down, Right, Up, Left
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    self.maze[ny][nx] == ' ' and (nx, ny) not in self.visited):
                    stack.append(((nx, ny), path + [(nx, ny)]))
                    self.visited.add((nx, ny))
        
        return []
    
    def _solve_bfs(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Solve the maze using breadth-first search."""
        queue = [(start, [start])]
        self.visited = set([start])
        
        while queue:
            (x, y), path = queue.pop(0)
            
            # Record step for visualization
            current_maze = [row[:] for row in self.maze]
            current_maze[y][x] = 'X'  # Mark current position
            self.solving_steps.append((current_maze, path))
            
            if (x, y) == end:
                self.solution = path
                return path
            
            # Check all four directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Down, Right, Up, Left
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    self.maze[ny][nx] == ' ' and (nx, ny) not in self.visited):
                    queue.append(((nx, ny), path + [(nx, ny)]))
                    self.visited.add((nx, ny))
        
        return []
    
    def _solve_astar(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Solve the maze using A* algorithm."""
        import heapq
        
        # Heuristic function (Manhattan distance)
        def h(pos: Tuple[int, int]) -> int:
            return abs(pos[0] - end[0]) + abs(pos[1] - end[1])
        
        # Priority queue
        open_set = [(h(start), 0, start, [start])]  # (f, g, position, path)
        closed_set = set()
        g_score = {start: 0}  # Cost from start to current position
        
        while open_set:
            f, g, (x, y), path = heapq.heappop(open_set)
            
            # Record step for visualization
            current_maze = [row[:] for row in self.maze]
            current_maze[y][x] = 'X'  # Mark current position
            self.solving_steps.append((current_maze, path))
            
            if (x, y) == end:
                self.solution = path
                return path
            
            if (x, y) in closed_set:
                continue
            
            closed_set.add((x, y))
            
            # Check all four directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Down, Right, Up, Left
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
    """Color themes for the maze display."""
    
    THEMES = {
        "Cosmic": {
            "background": (5, 5, 15),
            "wall": [(25, 25, 112), (138, 43, 226)],  # Navy blue to purple gradient
            "path": [(30, 30, 70), (70, 30, 70)],
            "cell": [(0, 0, 0), (10, 10, 30)],
            "current": (255, 215, 0),  # Gold
            "entrance": (0, 255, 127),  # Spring green
            "exit": (220, 20, 60),  # Crimson
            "solution": [(255, 105, 180), (255, 20, 147)],  # Hot pink gradient
            "highlight": (255, 255, 255),
            "text": (240, 240, 255),
            "button": (70, 70, 120),
            "button_hover": (90, 90, 150),
            "slider": (70, 70, 120),
            "slider_handle": (150, 150, 200),
            "user_path": (255, 140, 0),  # Bright orange for user-drawn path
            "user_path_highlight": (255, 215, 0),  # Gold highlight for user path nodes
        },
        "Forest": {
            "background": (10, 30, 15),
            "wall": [(34, 139, 34), (0, 100, 0)],  # Forest green gradient
            "path": [(20, 40, 20), (30, 50, 30)],
            "cell": [(0, 0, 0), (10, 20, 10)],
            "current": (255, 215, 0),  # Gold
            "entrance": (152, 251, 152),  # Pale green
            "exit": (220, 20, 60),  # Crimson
            "solution": [(255, 165, 0), (255, 140, 0)],  # Orange gradient
            "highlight": (255, 255, 255),
            "text": (220, 255, 220),
            "button": (40, 80, 40),
            "button_hover": (60, 100, 60),
            "slider": (40, 80, 40),
            "slider_handle": (100, 170, 100),
            "user_path": (255, 140, 0),  # Bright orange for user-drawn path
            "user_path_highlight": (255, 215, 0),  # Gold highlight for user path nodes
        },
        "Volcano": {
            "background": (20, 10, 5),
            "wall": [(139, 0, 0), (80, 0, 0)],  # Dark red gradient
            "path": [(40, 20, 10), (50, 25, 15)],
            "cell": [(0, 0, 0), (20, 10, 5)],
            "current": (255, 215, 0),  # Gold
            "entrance": (152, 251, 152),  # Pale green
            "exit": (30, 144, 255),  # Dodger blue
            "solution": [(255, 165, 0), (255, 69, 0)],  # Orange to red-orange gradient
            "highlight": (255, 255, 255),
            "text": (255, 230, 220),
            "button": (100, 40, 40),
            "button_hover": (130, 60, 60),
            "slider": (100, 40, 40),
            "slider_handle": (180, 100, 100),
            "user_path": (0, 191, 255),  # Bright blue for user-drawn path in volcano theme
            "user_path_highlight": (30, 144, 255),  # Blue highlight for user path nodes
        },
        "Neon": {
            "background": (5, 5, 5),
            "wall": [(0, 255, 255), (0, 150, 255)],  # Cyan to blue gradient
            "path": [(15, 15, 15), (20, 20, 20)],
            "cell": [(0, 0, 0), (5, 5, 5)],
            "current": (255, 215, 0),  # Gold
            "entrance": (0, 255, 127),  # Spring green
            "exit": (255, 20, 147),  # Deep pink
            "solution": [(255, 0, 255), (200, 0, 255)],  # Magenta gradient
            "highlight": (255, 255, 255),
            "text": (180, 255, 255),
            "button": (0, 100, 100),
            "button_hover": (0, 130, 130),
            "slider": (0, 100, 100),
            "slider_handle": (0, 200, 200),
            "user_path": (255, 215, 0),  # Gold for user-drawn path in neon theme
            "user_path_highlight": (255, 255, 0),  # Yellow highlight for user path nodes
        },
        "Monochrome": {
            "background": (5, 5, 5),
            "wall": [(200, 200, 200), (150, 150, 150)],  # Light gray gradient
            "path": [(20, 20, 20), (30, 30, 30)],
            "cell": [(0, 0, 0), (10, 10, 10)],
            "current": (255, 255, 255),  # White
            "entrance": (150, 150, 150),  # Medium gray
            "exit": (200, 200, 200),  # Light gray
            "solution": [(255, 255, 255), (220, 220, 220)],  # White gradient
            "highlight": (255, 255, 255),
            "text": (220, 220, 220),
            "button": (70, 70, 70),
            "button_hover": (90, 90, 90),
            "slider": (70, 70, 70),
            "slider_handle": (150, 150, 150),
            "user_path": (180, 180, 180),  # Light gray for user-drawn path
            "user_path_highlight": (255, 255, 255),  # White highlight for user path nodes
        }
    }
    
    @classmethod
    def get_theme(cls, name: str) -> Dict:
        """Get a theme by name."""
        return cls.THEMES.get(name, cls.THEMES["Cosmic"])
    
    @classmethod
    def get_theme_names(cls) -> List[str]:
        """Get a list of all theme names."""
        return list(cls.THEMES.keys())


class MazeUI:
    """Magnificent Maze UI class."""
    
    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        """Initialize the UI."""
        self.screen_width = width
        self.screen_height = height
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Magnificent Maze Generator")
        
        # Try to set a window icon
        try:
            icon = pygame.Surface((32, 32))
            icon.fill((50, 50, 100))
            for i in range(4):
                pygame.draw.line(icon, (255, 255, 255), (8, 8+i*5), (24, 8+i*5), 2)
                pygame.draw.line(icon, (255, 255, 255), (8+i*5, 8), (8+i*5, 24), 2)
            pygame.display.set_icon(icon)
        except:
            pass  # Ignore if icon can't be set
        
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
        self.animation_speed = 10  # Steps per frame
        self.current_step = 0
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
        
        # Path drawing mode
        self.drawing_mode = False
        self.user_path = []
        self.last_drawn_point = None
        
        # UI elements
        self.buttons = []
        self.sliders = []
        self.active_element = None
        
        # Theme
        self.current_theme = "Cosmic"
        self.theme = MazeTheme.get_theme(self.current_theme)
        
        # Initialize UI
        self._init_ui()
        
        # Generate initial maze
        self.generate_maze()
    
    def _init_ui(self):
        """Initialize UI elements."""
        # Clear existing elements
        self.buttons = []
        self.sliders = []
        
        # Calculate positions
        canvas_width = self.screen_width - SIDEBAR_WIDTH
        sidebar_x = canvas_width
        y_offset = 20
        
        # Title
        self.title_rect = pygame.Rect(sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 40)
        y_offset += 50
        
        # Algorithm selection
        self.add_button("Eller's Algorithm", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        lambda: self.set_algorithm(MazeAlgorithm.ELLERS))
        y_offset += 40
        self.add_button("Recursive Backtracker", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        lambda: self.set_algorithm(MazeAlgorithm.BACKTRACKER))
        y_offset += 40
        self.add_button("Hybrid Algorithm", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        lambda: self.set_algorithm(MazeAlgorithm.HYBRID))
        y_offset += 40
        self.add_button("Custom Path Algorithm", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        lambda: self.set_algorithm(MazeAlgorithm.CUSTOM_PATH))
        y_offset += 50
        
        # Draw Path Mode button
        self.add_button("Enter Draw Path Mode", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 40,
                        self.toggle_drawing_mode)
        y_offset += 50
        
        # Maze dimensions
        self.add_slider("Width", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30, 
                        10, 100, self.maze_width, lambda v: self.set_maze_width(int(v)))
        y_offset += 50
        self.add_slider("Height", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        10, 100, self.maze_height, lambda v: self.set_maze_height(int(v)))
        y_offset += 50
        
        # Density (for Eller's and Hybrid)
        self.add_slider("Density", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        0.1, 0.9, self.density, lambda v: self.set_density(v))
        y_offset += 50
        
        # Section size (for Hybrid)
        self.add_slider("Section Size", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        3, 20, self.section_size, lambda v: self.set_section_size(int(v)))
        y_offset += 50
        
        # Animation speed
        self.add_slider("Animation Speed", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        1, 50, self.animation_speed, lambda v: self.set_animation_speed(int(v)))
        y_offset += 50
        
        # Theme selection
        self.theme_rect = pygame.Rect(sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30)
        theme_names = MazeTheme.get_theme_names()
        theme_buttons_width = (SIDEBAR_WIDTH - 20) // len(theme_names)
        
        for i, theme_name in enumerate(theme_names):
            self.add_button(theme_name[:1], 
                           sidebar_x + 10 + i * theme_buttons_width, 
                           y_offset, 
                           theme_buttons_width, 30,
                           lambda tn=theme_name: self.set_theme(tn))
        y_offset += 50
        
        # Action buttons
        self.add_button("Generate", sidebar_x + 10, y_offset, (SIDEBAR_WIDTH - 30) // 2, 40,
                        self.generate_maze)
        self.add_button("Solve", sidebar_x + 20 + (SIDEBAR_WIDTH - 30) // 2, y_offset, 
                       (SIDEBAR_WIDTH - 30) // 2, 40, self.solve_maze)
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
        self.add_button("Save as PNG", sidebar_x + 10, y_offset, (SIDEBAR_WIDTH - 30) // 2, 30,
                        lambda: self.save_maze("png"))
        self.add_button("Save as TXT", sidebar_x + 20 + (SIDEBAR_WIDTH - 30) // 2, y_offset,
                       (SIDEBAR_WIDTH - 30) // 2, 30, lambda: self.save_maze("txt"))
        y_offset += 60
        
        # Status area
        self.status_rect = pygame.Rect(sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 100)
    
    def add_button(self, text: str, x: int, y: int, width: int, height: int, 
                  action: callable) -> None:
        """Add a button to the UI."""
        self.buttons.append({
            "rect": pygame.Rect(x, y, width, height),
            "text": text,
            "action": action,
            "hover": False
        })
    
    def add_slider(self, text: str, x: int, y: int, width: int, height: int,
                  min_value: float, max_value: float, initial_value: float,
                  action: callable) -> None:
        """Add a slider to the UI."""
        self.sliders.append({
            "rect": pygame.Rect(x, y, width, height),
            "text": text,
            "min": min_value,
            "max": max_value,
            "value": initial_value,
            "action": action,
            "active": False,
            "handle_rect": pygame.Rect(
                x + int((initial_value - min_value) / (max_value - min_value) * (width - 20)),
                y, 20, height
            )
        })
    
    def set_algorithm(self, algorithm: MazeAlgorithm) -> None:
        """Set the maze generation algorithm."""
        self.algorithm = algorithm
        self.update_status(f"Algorithm set to {algorithm.value}")
        
        # If setting to custom path but no path exists, enter drawing mode
        if algorithm == MazeAlgorithm.CUSTOM_PATH and not self.user_path:
            self.toggle_drawing_mode()
    
    def set_maze_width(self, width: int) -> None:
        """Set the maze width."""
        self.maze_width = max(10, min(100, width))
        
        # Clear user path when dimensions change
        if self.user_path:
            self.user_path = []
            self._redraw_maze()
    
    def set_maze_height(self, height: int) -> None:
        """Set the maze height."""
        self.maze_height = max(10, min(100, height))
        
        # Clear user path when dimensions change
        if self.user_path:
            self.user_path = []
            self._redraw_maze()
    
    def set_density(self, density: float) -> None:
        """Set the connection density."""
        self.density = max(0.1, min(0.9, density))
    
    def set_section_size(self, size: int) -> None:
        """Set the section size for hybrid algorithm."""
        self.section_size = max(3, min(20, size))
    
    def set_animation_speed(self, speed: int) -> None:
        """Set the animation speed."""
        self.animation_speed = max(1, min(50, speed))
    
    def set_cell_size(self, size: int) -> None:
        """Set the cell display size (zoom)."""
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
            
            # Redraw maze
            self._redraw_maze()
    
    def set_theme(self, theme_name: str) -> None:
        """Set the display theme."""
        if theme_name in MazeTheme.get_theme_names():
            self.current_theme = theme_name
            self.theme = MazeTheme.get_theme(theme_name)
            self.update_status(f"Theme set to {theme_name}")
            
            # Redraw maze with new theme
            self._redraw_maze()
    
    def reset_view(self) -> None:
        """Reset the view to default."""
        self.offset_x = 0
        self.offset_y = 0
        self.cell_size = DEFAULT_CELL_SIZE
        self._redraw_maze()
    
    def toggle_drawing_mode(self) -> None:
        """Toggle the path drawing mode."""
        if self.generating or self.solving:
            return
        
        self.drawing_mode = not self.drawing_mode
        
        if self.drawing_mode:
            # Enter drawing mode
            self.user_path = []
            self.last_drawn_point = None
            self.update_status("Draw Path Mode: Click and drag to create a path. Press Enter when done, Escape to cancel.")
            
            # Auto-select the custom path algorithm
            self.algorithm = MazeAlgorithm.CUSTOM_PATH
            
            # Create grid for drawing
            self._prepare_drawing_grid()
        else:
            # Exit drawing mode
            if not self.user_path:
                self.update_status("Drawing canceled. No path was created.")
            else:
                self.update_status(f"Path created with {len(self.user_path)} points. Generate maze to apply.")
    
    def _prepare_drawing_grid(self) -> None:
        """Prepare a grid for path drawing."""
        # Create a blank grid for drawing
        self.maze = None
        self.maze_surface = None
        self._redraw_maze(draw_grid=True)
    
    def _constrain_offset(self) -> None:
        """Constrain the offset to keep the maze in view."""
        if not self.maze and not self.drawing_mode:
            return
        
        # Calculate bounds
        maze_pixel_width = self.maze_width * 2 + 1
        maze_pixel_height = self.maze_height * 2 + 1
        
        # Constrain X offset
        min_x = -5  # Allow a bit of padding
        max_x = max(maze_pixel_width - (self.screen_width - SIDEBAR_WIDTH) / self.cell_size + 5, min_x)
        self.offset_x = max(min_x, min(self.offset_x, max_x))
        
        # Constrain Y offset
        min_y = -5  # Allow a bit of padding
        max_y = max(maze_pixel_height - self.screen_height / self.cell_size + 5, min_y)
        self.offset_y = max(min_y, min(self.offset_y, max_y))
    
    def handle_path_drawing(self, mouse_pos: Tuple[int, int]) -> None:
        """Handle drawing a path with the mouse."""
        if not self.drawing_mode:
            return
        
        # Convert mouse position to grid coordinates
        grid_x = int((mouse_pos[0] / self.cell_size + self.offset_x) / 2)
        grid_y = int((mouse_pos[1] / self.cell_size + self.offset_y) / 2)
        
        # Ensure coordinates are within bounds
        if not (0 <= grid_x < self.maze_width and 0 <= grid_y < self.maze_height):
            return
        
        current_point = (grid_x, grid_y)
        
        # If this is a new point, add it to the path
        if current_point != self.last_drawn_point:
            # If there's a previous point, make sure they're adjacent
            if self.last_drawn_point:
                prev_x, prev_y = self.last_drawn_point
                
                # If points are not adjacent, create intermediate points to make a continuous path
                dx = grid_x - prev_x
                dy = grid_y - prev_y
                
                # Manhattan distance
                manhattan_dist = abs(dx) + abs(dy)
                
                if manhattan_dist > 1:
                    # Create a path of intermediate points
                    steps_x = abs(dx)
                    steps_y = abs(dy)
                    
                    # First move horizontally, then vertically
                    x, y = prev_x, prev_y
                    
                    # Move along X
                    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
                    for _ in range(steps_x):
                        x += step_x
                        intermediate_point = (x, y)
                        if intermediate_point not in self.user_path:
                            self.user_path.append(intermediate_point)
                    
                    # Move along Y
                    step_y = 1 if dy > 0 else -1 if dy < 0 else 0
                    for _ in range(steps_y):
                        y += step_y
                        intermediate_point = (x, y)
                        if intermediate_point not in self.user_path:
                            self.user_path.append(intermediate_point)
                else:
                    # Points are adjacent, just add the new point
                    if current_point not in self.user_path:
                        self.user_path.append(current_point)
            else:
                # First point in the path
                self.user_path.append(current_point)
            
            self.last_drawn_point = current_point
            self._redraw_maze(draw_grid=True)
    
    def generate_maze(self) -> None:
        """Generate a new maze."""
        if self.generating or self.solving:
            return
        
        if self.drawing_mode:
            self.toggle_drawing_mode()  # Exit drawing mode
        
        self.update_status("Generating maze...")
        self.generating = True
        self.show_solution = False
        self.current_step = 0
        
        # Create the maze based on the selected algorithm
        if self.algorithm == MazeAlgorithm.CUSTOM_PATH and not self.user_path:
            self.update_status("No custom path drawn. Please enter drawing mode first.")
            self.generating = False
            return
        
        # Create the maze
        self.maze = create_maze(
            self.maze_width, 
            self.maze_height, 
            self.algorithm,
            self.density,
            self.section_size,
            self.user_path if self.algorithm == MazeAlgorithm.CUSTOM_PATH else None
        )
        
        # Store generation steps for animation
        self.generation_steps = self.maze.generation_steps
        
        # Initialize solver
        self.solver = MazeSolver(self.maze.maze)
        
        # Reset view for new maze
        self.reset_view()
        
        self.update_status(f"Maze generated in {self.maze.generation_time:.3f} seconds!")
    
    def _finish_generation(self) -> None:
        """Finish the maze generation animation."""
        self.generating = False
        self._redraw_maze()
        
        # Calculate complexity metrics
        path_length = len(self.solver.solve("astar"))
        complexity_score = path_length / (self.maze_width * self.maze_height)
        complexity_rating = "Low"
        if complexity_score > 0.7:
            complexity_rating = "Extreme"
        elif complexity_score > 0.5:
            complexity_rating = "High"
        elif complexity_score > 0.3:
            complexity_rating = "Medium"
            
        self.update_status(
            f"Maze generated with {self.algorithm.value}!\n"
            f"Size: {self.maze_width}x{self.maze_height}\n"
            f"Solution length: {path_length}\n"
            f"Complexity rating: {complexity_rating}"
        )
    
    def solve_maze(self) -> None:
        """Solve the current maze."""
        if not self.maze or self.generating or self.solving:
            return
        
        self.update_status("Solving maze...")
        self.solving = True
        self.show_solution = False
        self.current_step = 0
        
        # Solve with A* algorithm
        start_time = time.time()
        self.solution = self.solver.solve("astar")
        solve_time = time.time() - start_time
        
        # Store solving steps for animation
        self.solving_steps = self.solver.solving_steps
        
        if not self.solution:
            self.solving = False
            self.update_status("No solution found!")
            return
        
        self.update_status(f"Solution found! Length: {len(self.solution)}, Time: {solve_time:.3f}s")
    
    def _finish_solving(self) -> None:
        """Finish the maze solving animation."""
        self.solving = False
        self.show_solution = True
        self._redraw_maze()
    
    def save_maze(self, format_type: str) -> None:
        """Save the current maze to a file."""
        if not self.maze:
            self.update_status("No maze to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "png":
            # Create a surface containing just the maze
            maze_width = (self.maze_width * 2 + 1) * self.cell_size
            maze_height = (self.maze_height * 2 + 1) * self.cell_size
            
            maze_surf = pygame.Surface((maze_width, maze_height))
            maze_surf.fill(self.theme["background"])
            
            # Draw the maze on the surface with a high cell size for quality
            temp_cell_size = self.cell_size
            temp_offset_x = self.offset_x
            temp_offset_y = self.offset_y
            
            self.cell_size = max(MIN_CELL_SIZE, min(MAX_CELL_SIZE, 20))  # Reasonable size for export
            self.offset_x = 0
            self.offset_y = 0
            
            self._draw_maze(maze_surf, 0, 0, maze_width, maze_height)
            
            # Draw solution if shown
            if self.show_solution and self.solution:
                self._draw_solution(maze_surf, 0, 0, maze_width, maze_height)
            
            # Restore original values
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
    
    def update_status(self, message: str) -> None:
        """Update the status message."""
        self.status_message = message
    
    def handle_events(self) -> bool:
        """Handle pygame events."""
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
                
                # Handle UI element hover
                for button in self.buttons:
                    button["hover"] = button["rect"].collidepoint(event.pos)
                
                # Handle path drawing
                if self.drawing_mode and pygame.mouse.get_pressed()[0]:
                    canvas_rect = pygame.Rect(0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
                    if canvas_rect.collidepoint(event.pos):
                        self.handle_path_drawing(event.pos)
                
                # Handle dragging for pan
                elif self.dragging:
                    dx = (event.pos[0] - self.drag_start[0]) / self.cell_size
                    dy = (event.pos[1] - self.drag_start[1]) / self.cell_size
                    self.offset_x -= dx
                    self.offset_y -= dy
                    self.drag_start = event.pos
                    self._constrain_offset()
                    self._redraw_maze(draw_grid=self.drawing_mode)
                
                # Handle slider dragging
                elif self.active_element and "min" in self.active_element:
                    slider = self.active_element
                    # Calculate value based on mouse position
                    slider_width = slider["rect"].width - 20  # Subtract handle width
                    relative_x = max(0, min(slider_width, event.pos[0] - slider["rect"].x))
                    value = slider["min"] + (relative_x / slider_width) * (slider["max"] - slider["min"])
                    
                    # Update slider value and handle position
                    slider["value"] = value
                    slider["handle_rect"].x = slider["rect"].x + relative_x
                    
                    # Call action
                    slider["action"](value)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # Check for UI interactions first
                    ui_handled = False
                    
                    # Check for button clicks
                    for button in self.buttons:
                        if button["rect"].collidepoint(event.pos):
                            button["action"]()
                            ui_handled = True
                            break
                    
                    # Check for slider interaction
                    if not ui_handled:
                        for slider in self.sliders:
                            if slider["handle_rect"].collidepoint(event.pos) or slider["rect"].collidepoint(event.pos):
                                self.active_element = slider
                                slider["active"] = True
                                
                                # Update slider position immediately
                                slider_width = slider["rect"].width - 20
                                relative_x = max(0, min(slider_width, event.pos[0] - slider["rect"].x))
                                value = slider["min"] + (relative_x / slider_width) * (slider["max"] - slider["min"])
                                
                                slider["value"] = value
                                slider["handle_rect"].x = slider["rect"].x + relative_x
                                
                                slider["action"](value)
                                ui_handled = True
                                break
                    
                    # Handle path drawing
                    if not ui_handled and self.drawing_mode:
                        canvas_rect = pygame.Rect(0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
                        if canvas_rect.collidepoint(event.pos):
                            self.handle_path_drawing(event.pos)
                            ui_handled = True
                    
                    # Handle dragging for pan (if not on UI element and not drawing)
                    if not ui_handled and not self.drawing_mode:
                        canvas_rect = pygame.Rect(0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
                        if canvas_rect.collidepoint(event.pos):
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
                        # Cancel drawing mode
                        self.drawing_mode = False
                        self.user_path = []
                        self.update_status("Path drawing canceled.")
                        self._redraw_maze()
                    else:
                        return False
                
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    if self.drawing_mode:
                        # Finish drawing mode
                        self.toggle_drawing_mode()
                
                elif event.key == pygame.K_g:
                    self.generate_maze()
                
                elif event.key == pygame.K_s:
                    self.solve_maze()
                
                elif event.key == pygame.K_r:
                    self.reset_view()
                
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.set_cell_size(self.cell_size + 2)
                
                elif event.key == pygame.K_MINUS:
                    self.set_cell_size(self.cell_size - 2)
                
                elif event.key == pygame.K_c:
                    if not self.generating and not self.solving:
                        # Clear the user path
                        self.user_path = []
                        self._redraw_maze(draw_grid=self.drawing_mode)
                        self.update_status("User path cleared.")
                
                elif event.key in [pygame.K_UP, pygame.K_w]:
                    self.offset_y -= 5
                    self._constrain_offset()
                    self._redraw_maze(draw_grid=self.drawing_mode)
                
                elif event.key in [pygame.K_DOWN, pygame.K_s]:
                    self.offset_y += 5
                    self._constrain_offset()
                    self._redraw_maze(draw_grid=self.drawing_mode)
                
                elif event.key in [pygame.K_LEFT, pygame.K_a]:
                    self.offset_x -= 5
                    self._constrain_offset()
                    self._redraw_maze(draw_grid=self.drawing_mode)
                
                elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                    self.offset_x += 5
                    self._constrain_offset()
                    self._redraw_maze(draw_grid=self.drawing_mode)
        
        return True
    
    def _redraw_maze(self, draw_grid: bool = False) -> None:
        """Redraw the maze surface."""
        # Create a new surface
        self.maze_surface = pygame.Surface((self.screen_width - SIDEBAR_WIDTH, self.screen_height))
        self.maze_surface.fill(self.theme["background"])
        
        if draw_grid:
            # Draw the grid for path drawing
            self._draw_grid(self.maze_surface, 0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
            
            # Draw the user path
            if self.user_path:
                self._draw_user_path(self.maze_surface, 0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
        elif self.maze:
            # Draw the maze
            self._draw_maze(self.maze_surface, 0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
            
            # Draw solution if shown
            if self.show_solution and self.solution:
                self._draw_solution(self.maze_surface, 0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
    
    def _get_gradient_color(self, colors: List[Tuple[int, int, int]], position: float) -> Tuple[int, int, int]:
        """Get a color from a gradient based on position (0-1)."""
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
    
    def _draw_grid(self, surface: pygame.Surface, x: int, y: int, width: int, height: int) -> None:
        """Draw a grid for path drawing."""
        # Calculate grid boundaries
        start_x = max(0, int(self.offset_x))
        start_y = max(0, int(self.offset_y))
        
        end_x = min(self.maze_width, int(self.offset_x + width / self.cell_size / 2 + 1))
        end_y = min(self.maze_height, int(self.offset_y + height / self.cell_size / 2 + 1))
        
        # Draw grid cells
        for grid_y in range(start_y, end_y):
            for grid_x in range(start_x, end_x):
                cell_x = int((grid_x * 2 - self.offset_x) * self.cell_size)
                cell_y = int((grid_y * 2 - self.offset_y) * self.cell_size)
                
                # Draw cell background
                cell_color = self.theme["cell"][0]
                pygame.draw.rect(surface, cell_color, 
                               (cell_x, cell_y, self.cell_size * 2, self.cell_size * 2))
                
                # Draw cell border
                border_color = (50, 50, 50)  # Subtle grid lines
                pygame.draw.rect(surface, border_color, 
                               (cell_x, cell_y, self.cell_size * 2, self.cell_size * 2), 1)
    
    def _draw_user_path(self, surface: pygame.Surface, x: int, y: int, width: int, height: int) -> None:
        """Draw the user-drawn path on the surface."""
        if not self.user_path:
            return
        
        # Draw lines connecting the path points
        for i in range(len(self.user_path) - 1):
            x1, y1 = self.user_path[i]
            x2, y2 = self.user_path[i+1]
            
            # Calculate screen positions
            screen_x1 = int((x1 * 2 - self.offset_x) * self.cell_size + self.cell_size)
            screen_y1 = int((y1 * 2 - self.offset_y) * self.cell_size + self.cell_size)
            screen_x2 = int((x2 * 2 - self.offset_x) * self.cell_size + self.cell_size)
            screen_y2 = int((y2 * 2 - self.offset_y) * self.cell_size + self.cell_size)
            
            # Draw line with path color
            line_width = max(2, self.cell_size // 3)
            pygame.draw.line(surface, self.theme["user_path"], 
                           (screen_x1, screen_y1), (screen_x2, screen_y2), line_width)
        
        # Draw nodes for each point in the path
        for i, (px, py) in enumerate(self.user_path):
            screen_x = int((px * 2 - self.offset_x) * self.cell_size + self.cell_size)
            screen_y = int((py * 2 - self.offset_y) * self.cell_size + self.cell_size)
            
            # Different color for start and end points
            if i == 0 or i == len(self.user_path) - 1:
                node_color = self.theme["user_path_highlight"]
                node_size = max(3, self.cell_size // 2)
            else:
                node_color = self.theme["user_path"]
                node_size = max(2, self.cell_size // 3)
            
            pygame.draw.circle(surface, node_color, (screen_x, screen_y), node_size)
    
    def _draw_maze(self, surface: pygame.Surface, x: int, y: int, width: int, height: int) -> None:
        """Draw the maze on the given surface."""
        if not self.maze:
            return
        
        maze_data = None
        
        # Use the appropriate maze data based on state
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
        
        # Create a sense of depth with subtle gradient
        for y_cell in range(start_y, end_y):
            for x_cell in range(start_x, end_x):
                # Skip if outside the maze
                if y_cell >= len(maze_data) or x_cell >= len(maze_data[0]):
                    continue
                
                # Calculate position for gradient effect
                gradient_pos_x = x_cell / len(maze_data[0])
                gradient_pos_y = y_cell / len(maze_data)
                gradient_pos = (gradient_pos_x + gradient_pos_y) / 2
                
                cell_x = int((x_cell - self.offset_x) * self.cell_size)
                cell_y = int((y_cell - self.offset_y) * self.cell_size)
                
                cell_type = maze_data[y_cell][x_cell]
                
                # Cell is a wall
                if cell_type == '#':
                    color = self._get_gradient_color(self.theme["wall"], gradient_pos)
                    pygame.draw.rect(surface, color, 
                                    (cell_x, cell_y, self.cell_size, self.cell_size))
                    
                    # Add subtle 3D effect to walls
                    shadow_rect = pygame.Rect(cell_x, cell_y, self.cell_size, self.cell_size)
                    lighter = self._get_gradient_color(self.theme["wall"], min(1, gradient_pos + 0.1))
                    darker = self._get_gradient_color(self.theme["wall"], max(0, gradient_pos - 0.1))
                    
                    # Top and left edges lighter
                    pygame.draw.line(surface, lighter, 
                                    (shadow_rect.left, shadow_rect.top), 
                                    (shadow_rect.right, shadow_rect.top))
                    pygame.draw.line(surface, lighter, 
                                    (shadow_rect.left, shadow_rect.top), 
                                    (shadow_rect.left, shadow_rect.bottom))
                    
                    # Bottom and right edges darker
                    pygame.draw.line(surface, darker, 
                                    (shadow_rect.left, shadow_rect.bottom-1), 
                                    (shadow_rect.right, shadow_rect.bottom-1))
                    pygame.draw.line(surface, darker, 
                                    (shadow_rect.right-1, shadow_rect.top), 
                                    (shadow_rect.right-1, shadow_rect.bottom))
                
                # Cell is a path
                elif cell_type == ' ':
                    # Special handling for entrance and exit
                    if (x_cell == 0 and y_cell % 2 == 1):  # Entrance
                        color = self.theme["entrance"]
                    elif (x_cell == len(maze_data[0])-1 and y_cell % 2 == 1):  # Exit
                        color = self.theme["exit"]
                    else:
                        color = self._get_gradient_color(self.theme["cell"], gradient_pos)
                    
                    pygame.draw.rect(surface, color, 
                                   (cell_x, cell_y, self.cell_size, self.cell_size))
                
                # Cell is part of user path visualization
                elif cell_type == 'P':
                    pygame.draw.rect(surface, self.theme["user_path"], 
                                   (cell_x, cell_y, self.cell_size, self.cell_size))
                
                # Cell is marked for visualization (current cell in generation)
                elif cell_type in ['C', 'B', 'H', 'X']:
                    marker_colors = {
                        'C': self.theme["current"],  # Current cell
                        'B': (255, 165, 0),         # Backtracking
                        'H': (0, 191, 255),         # Highlight
                        'X': (255, 0, 0)            # Solving path
                    }
                    
                    # Draw cell background
                    pygame.draw.rect(surface, self._get_gradient_color(self.theme["cell"], gradient_pos), 
                                   (cell_x, cell_y, self.cell_size, self.cell_size))
                    
                    # Draw marker
                    marker_color = marker_colors.get(cell_type, self.theme["highlight"])
                    
                    # Different marker shapes for different types
                    if cell_type == 'C':  # Circle for current
                        pygame.draw.circle(surface, marker_color, 
                                         (cell_x + self.cell_size // 2, cell_y + self.cell_size // 2), 
                                         self.cell_size // 3)
                    elif cell_type == 'B':  # Diamond for backtracking
                        points = [
                            (cell_x + self.cell_size // 2, cell_y + self.cell_size // 4),
                            (cell_x + 3 * self.cell_size // 4, cell_y + self.cell_size // 2),
                            (cell_x + self.cell_size // 2, cell_y + 3 * self.cell_size // 4),
                            (cell_x + self.cell_size // 4, cell_y + self.cell_size // 2)
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
    
    def _draw_solution(self, surface: pygame.Surface, x: int, y: int, width: int, height: int) -> None:
        """Draw the solution path on the given surface."""
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
            
            # Draw line with gradient color based on position in path
            pos = i / len(self.solution)
            color = self._get_gradient_color(self.theme["solution"], pos)
            
            # Thicker line for visibility
            line_width = max(2, self.cell_size // 4)
            
            # Draw line
            pygame.draw.line(surface, color, (screen_x1, screen_y1), (screen_x2, screen_y2), line_width)
            
            # Add dot at each vertex
            dot_radius = max(2, line_width // 2 + 1)
            pygame.draw.circle(surface, color, (screen_x1, screen_y1), dot_radius)
            pygame.draw.circle(surface, color, (screen_x2, screen_y2), dot_radius)
    
    def _draw_ui(self) -> None:
        """Draw the UI elements."""
        # Draw sidebar background
        sidebar_rect = pygame.Rect(self.screen_width - SIDEBAR_WIDTH, 0, SIDEBAR_WIDTH, self.screen_height)
        
        # Draw sidebar with gradient
        for y in range(0, self.screen_height, 2):
            progress = y / self.screen_height
            color = self._get_gradient_color([
                (int(self.theme["background"][0] * 1.5), int(self.theme["background"][1] * 1.5), int(self.theme["background"][2] * 1.5)),
                (int(self.theme["background"][0] * 1.8), int(self.theme["background"][1] * 1.8), int(self.theme["background"][2] * 1.8))
            ], progress)
            pygame.draw.line(self.screen, color, 
                           (sidebar_rect.left, y), (sidebar_rect.right, y))
        
        # Draw separator line
        pygame.draw.line(self.screen, self.theme["wall"][0], 
                       (sidebar_rect.left, 0), (sidebar_rect.left, self.screen_height), 2)
        
        # Draw title
        title_text = "Magnificent Maze Generator"
        title_surf = self.font_title.render(title_text, True, self.theme["text"])
        title_rect = title_surf.get_rect(center=self.title_rect.center)
        self.screen.blit(title_surf, title_rect)
        
        # Draw buttons
        for button in self.buttons:
            # Special handling for Draw Path Mode button
            if button["text"] == "Enter Draw Path Mode" and self.drawing_mode:
                text = "Exit Draw Path Mode"
                color = (150, 50, 50) if button["hover"] else (120, 30, 30)
            else:
                text = button["text"]
                color = self.theme["button_hover"] if button["hover"] else self.theme["button"]
            
            pygame.draw.rect(self.screen, color, button["rect"], border_radius=5)
            pygame.draw.rect(self.screen, self.theme["text"], button["rect"], width=1, border_radius=5)
            
            # Draw button text
            text_surf = self.font.render(text, True, self.theme["text"])
            text_rect = text_surf.get_rect(center=button["rect"].center)
            self.screen.blit(text_surf, text_rect)
        
        # Draw sliders
        for slider in self.sliders:
            # Draw slider label
            label_surf = self.font.render(f"{slider['text']}: {slider['value']:.1f}", True, self.theme["text"])
            label_rect = pygame.Rect(slider["rect"].x, slider["rect"].y - 20, slider["rect"].width, 20)
            self.screen.blit(label_surf, label_rect)
            
            # Draw slider track
            track_rect = pygame.Rect(slider["rect"].x, slider["rect"].y + slider["rect"].height // 2 - 2,
                                  slider["rect"].width, 4)
            pygame.draw.rect(self.screen, self.theme["slider"], track_rect, border_radius=2)
            
            # Draw slider handle
            handle_color = self.theme["slider_handle"]
            pygame.draw.rect(self.screen, handle_color, slider["handle_rect"], border_radius=5)
            pygame.draw.rect(self.screen, self.theme["text"], slider["handle_rect"], width=1, border_radius=5)
        
        # Draw theme label
        theme_label = self.font.render("Theme:", True, self.theme["text"])
        theme_label_rect = pygame.Rect(self.theme_rect.x, self.theme_rect.y - 20, self.theme_rect.width, 20)
        self.screen.blit(theme_label, theme_label_rect)
        
        # Draw current algorithm indicator
        algo_text = f"Current Algorithm: {self.algorithm.value}"
        algo_surf = self.font.render(algo_text, True, self.theme["text"])
        algo_rect = pygame.Rect(sidebar_rect.left + 10, 15, SIDEBAR_WIDTH - 20, 20)
        self.screen.blit(algo_surf, algo_rect)
        
        # Draw current mode (normal or path drawing)
        if hasattr(self, 'drawing_mode') and self.drawing_mode:
            mode_text = "MODE: DRAWING PATH"
            mode_color = (255, 100, 100)  # Red-ish color to indicate drawing mode
        else:
            mode_text = "MODE: NORMAL"
            mode_color = self.theme["text"]
            
        mode_surf = self.font_large.render(mode_text, True, mode_color)
        mode_rect = mode_surf.get_rect(topright=(self.screen_width - 10, 10))
        self.screen.blit(mode_surf, mode_rect)
        
        # Draw path instructions if in drawing mode
        if hasattr(self, 'drawing_mode') and self.drawing_mode:
            instructions = [
                "Click and drag to draw your path",
                "Path must connect entrance to exit",
                "Press Enter to generate maze from path",
                "Press Escape to cancel"
            ]
            
            for i, line in enumerate(instructions):
                instr_surf = self.font.render(line, True, mode_color)
                instr_rect = instr_surf.get_rect(topright=(self.screen_width - 10, 40 + i * 25))
                
                # Add a semi-transparent background for better readability
                bg_rect = instr_rect.inflate(10, 6)
                bg_surf = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                bg_surf.fill((0, 0, 0, 150))
                self.screen.blit(bg_surf, bg_rect)
                
                self.screen.blit(instr_surf, instr_rect)
        
        # Draw status
        if hasattr(self, 'status_message'):
            # Split message into lines
            status_lines = self.status_message.split('\n')
            for i, line in enumerate(status_lines):
                status_surf = self.font.render(line, True, self.theme["text"])
                status_rect = status_surf.get_rect(topleft=(self.status_rect.x, self.status_rect.y + i * 20))
                self.screen.blit(status_surf, status_rect)
        
        # Draw current coordinates if mouse is over the maze area
        canvas_rect = pygame.Rect(0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
        if canvas_rect.collidepoint(self.mouse_pos):
            maze_x = int(self.offset_x + self.mouse_pos[0] / self.cell_size)
            maze_y = int(self.offset_y + self.mouse_pos[1] / self.cell_size)
            
            coord_text = f"Coordinates: ({maze_x}, {maze_y})"
            coord_surf = self.font.render(coord_text, True, self.theme["text"])
            coord_rect = coord_surf.get_rect(bottomright=(self.screen_width - SIDEBAR_WIDTH - 10, self.screen_height - 10))
            
            # Draw background for better visibility
            bg_rect = coord_rect.inflate(10, 6)
            pygame.draw.rect(self.screen, (0, 0, 0, 150), bg_rect)
            self.screen.blit(coord_surf, coord_rect)
        
        # Draw key commands help
        help_text = "Keys: G=Generate | S=Solve | R=Reset View | Arrow Keys=Pan | +/- =Zoom | D=Draw Path"
        help_surf = self.font.render(help_text, True, self.theme["text"])
        help_rect = help_surf.get_rect(midbottom=(self.screen_width - SIDEBAR_WIDTH // 2, self.screen_height - 10))
        self.screen.blit(help_surf, help_rect)
        
        # Draw user path if in drawing mode
        if hasattr(self, 'drawing_mode') and self.drawing_mode and hasattr(self, 'user_path') and self.user_path:
            # Draw the user's path
            for i in range(1, len(self.user_path)):
                x1, y1 = self.user_path[i-1]
                x2, y2 = self.user_path[i]
                
                # Convert to screen coordinates
                screen_x1 = int((x1 * 2 + 1 - self.offset_x) * self.cell_size + self.cell_size // 2)
                screen_y1 = int((y1 * 2 + 1 - self.offset_y) * self.cell_size + self.cell_size // 2)
                screen_x2 = int((x2 * 2 + 1 - self.offset_x) * self.cell_size + self.cell_size // 2)
                screen_y2 = int((y2 * 2 + 1 - self.offset_y) * self.cell_size + self.cell_size // 2)
                
                # Draw line with gradient color based on position in path
                pos = i / len(self.user_path)
                color = self._get_gradient_color([(255, 50, 50), (255, 200, 50)], pos)
                
                # Thicker line for visibility
                line_width = max(3, self.cell_size // 3)
                
                # Draw line on main screen (not the maze surface)
                pygame.draw.line(self.screen, color, (screen_x1, screen_y1), (screen_x2, screen_y2), line_width)
                
                # Add dot at each vertex
                dot_radius = max(3, line_width // 2 + 2)
                pygame.draw.circle(self.screen, color, (screen_x1, screen_y1), dot_radius)
                pygame.draw.circle(self.screen, color, (screen_x2, screen_y2), dot_radius)
    
    def update(self) -> None:
        """Update the game state."""
        # Handle maze generation animation
        if self.generating:
            if self.current_step < len(self.generation_steps) - 1:
                # Advance multiple steps per frame based on animation speed
                self.current_step = min(self.current_step + self.animation_speed, len(self.generation_steps) - 1)
            else:
                self._finish_generation()
        
        # Handle maze solving animation
        elif self.solving:
            if self.current_step < len(self.solving_steps) - 1:
                # Advance multiple steps per frame based on animation speed
                self.current_step = min(self.current_step + self.animation_speed, len(self.solving_steps) - 1)
            else:
                self._finish_solving()
    
    def draw(self) -> None:
        """Draw everything to the screen."""
        # Clear the screen
        self.screen.fill(self.theme["background"])
        
        # Draw the maze
        if self.maze_surface:
            self.screen.blit(self.maze_surface, (0, 0))
        elif self.maze:
            # Create the maze surface if it doesn't exist
            self._redraw_maze()
            self.screen.blit(self.maze_surface, (0, 0))
        
        # Draw UI elements
        self._draw_ui()
        
        # Update the display
        pygame.display.flip()
    
    def enable_path_drawing_mode(self) -> None:
        """Enable path drawing mode."""
        if self.generating or self.solving:
            self.update_status("Cannot start drawing while generating or solving")
            return
            
        self.drawing_mode = True
        self.user_path = []
        self.update_status("Drawing Mode: Click and drag to draw your path through the maze")
    
    def disable_path_drawing_mode(self) -> None:
        """Disable path drawing mode."""
        self.drawing_mode = False
        self.user_path = []
        self.update_status("Drawing mode canceled")
    
    def handle_path_drawing(self, mouse_pos) -> None:
        """Handle drawing a path point at the given mouse position."""
        # Convert mouse position to grid coordinates
        canvas_rect = pygame.Rect(0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
        if not canvas_rect.collidepoint(mouse_pos):
            return  # Ignore if mouse is in sidebar
            
        cell_x = int(self.offset_x + mouse_pos[0] / self.cell_size)
        cell_y = int(self.offset_y + mouse_pos[1] / self.cell_size)
        
        # Ensure we're on a cell, not a wall
        if cell_x % 2 == 0 or cell_y % 2 == 0:
            return
        
        # Convert to maze cell coordinates (not pixel/wall coordinates)
        grid_x = cell_x // 2
        grid_y = cell_y // 2
        
        # Add point to path if it's valid and not already in path
        if 0 <= grid_x < self.maze_width and 0 <= grid_y < self.maze_height:
            point = (grid_x, grid_y)
            if not self.user_path or point != self.user_path[-1]:
                self.user_path.append(point)
                # No need to redraw the maze here, as we're drawing the path on top in _draw_ui
    
    def generate_maze_from_path(self) -> None:
        """Generate a maze that incorporates the user-drawn path."""
        if not self.user_path or len(self.user_path) < 2:
            self.update_status("Path is too short. Please draw a longer path.")
            return
        
        # First verify that the path starts at an entry point and ends at an exit point
        # For simplicity, we'll assume entry is left edge and exit is right edge
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
        
        # Create a custom maze with the user path
        self.maze = CustomPathMaze(self.maze_width, self.maze_height, self.user_path, 
                                 self.algorithm, self.density, self.section_size)
        
        # Generate the maze
        self.maze.generate()
        
        # Store generation steps for animation
        self.generation_steps = self.maze.generation_steps
        
        # Initialize solver
        self.solver = MazeSolver(self.maze.maze)
        
        # Reset view for new maze
        self.reset_view()
        
        self.update_status(f"Maze generated with your custom path in {self.maze.generation_time:.3f} seconds!")
    
    def handle_events(self) -> bool:
        """Handle pygame events."""
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
                
                # Handle UI element hover
                for button in self.buttons:
                    button["hover"] = button["rect"].collidepoint(event.pos)
                
                # Handle path drawing if in drawing mode
                if hasattr(self, 'drawing_mode') and self.drawing_mode and pygame.mouse.get_pressed()[0]:
                    self.handle_path_drawing(event.pos)
                
                # Handle dragging for pan (if not in drawing mode)
                elif self.dragging and not (hasattr(self, 'drawing_mode') and self.drawing_mode):
                    dx = (event.pos[0] - self.drag_start[0]) / self.cell_size
                    dy = (event.pos[1] - self.drag_start[1]) / self.cell_size
                    self.offset_x -= dx
                    self.offset_y -= dy
                    self.drag_start = event.pos
                    self._constrain_offset()
                    self._redraw_maze()
                
                # Handle slider dragging
                if self.active_element and "min" in self.active_element:
                    slider = self.active_element
                    # Calculate value based on mouse position
                    slider_width = slider["rect"].width - 20  # Subtract handle width
                    relative_x = max(0, min(slider_width, event.pos[0] - slider["rect"].x))
                    value = slider["min"] + (relative_x / slider_width) * (slider["max"] - slider["min"])
                    
                    # Update slider value and handle position
                    slider["value"] = value
                    slider["handle_rect"].x = slider["rect"].x + relative_x
                    
                    # Call action
                    slider["action"](value)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # Check for button clicks
                    for button in self.buttons:
                        if button["rect"].collidepoint(event.pos):
                            button["action"]()
                            break
                    
                    # Check for slider interaction
                    for slider in self.sliders:
                        if slider["handle_rect"].collidepoint(event.pos) or slider["rect"].collidepoint(event.pos):
                            self.active_element = slider
                            slider["active"] = True
                            
                            # Update slider position immediately
                            slider_width = slider["rect"].width - 20
                            relative_x = max(0, min(slider_width, event.pos[0] - slider["rect"].x))
                            value = slider["min"] + (relative_x / slider_width) * (slider["max"] - slider["min"])
                            
                            slider["value"] = value
                            slider["handle_rect"].x = slider["rect"].x + relative_x
                            
                            slider["action"](value)
                            break
                    
                    # Handle dragging for pan (if not on UI element and not in drawing mode)
                    canvas_rect = pygame.Rect(0, 0, self.screen_width - SIDEBAR_WIDTH, self.screen_height)
                    if (canvas_rect.collidepoint(event.pos) and not self.active_element and 
                        not (hasattr(self, 'drawing_mode') and self.drawing_mode)):
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
                    # If in drawing mode, exit it
                    if hasattr(self, 'drawing_mode') and self.drawing_mode:
                        self.disable_path_drawing_mode()
                    else:
                        return False
                
                elif event.key == pygame.K_g:
                    if not hasattr(self, 'drawing_mode') or not self.drawing_mode:
                        self.generate_maze()
                
                elif event.key == pygame.K_s:
                    if not hasattr(self, 'drawing_mode') or not self.drawing_mode:
                        self.solve_maze()
                
                elif event.key == pygame.K_r:
                    if not hasattr(self, 'drawing_mode') or not self.drawing_mode:
                        self.reset_view()
                
                elif event.key == pygame.K_d:
                    # Toggle drawing mode
                    if hasattr(self, 'drawing_mode') and self.drawing_mode:
                        self.disable_path_drawing_mode()
                    else:
                        self.enable_path_drawing_mode()
                
                elif event.key == pygame.K_RETURN:
                    # Generate maze from path if in drawing mode
                    if hasattr(self, 'drawing_mode') and self.drawing_mode:
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
    
    def _init_ui(self):
        """Initialize UI elements."""
        # Clear existing elements
        self.buttons = []
        self.sliders = []
        
        # Calculate positions
        canvas_width = self.screen_width - SIDEBAR_WIDTH
        sidebar_x = canvas_width
        y_offset = 20
        
        # Title
        self.title_rect = pygame.Rect(sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 40)
        y_offset += 50
        
        # Algorithm selection
        self.add_button("Eller's Algorithm", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        lambda: self.set_algorithm(MazeAlgorithm.ELLERS))
        y_offset += 40
        self.add_button("Recursive Backtracker", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        lambda: self.set_algorithm(MazeAlgorithm.BACKTRACKER))
        y_offset += 40
        self.add_button("Hybrid Algorithm", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        lambda: self.set_algorithm(MazeAlgorithm.HYBRID))
        y_offset += 50
        
        # Maze dimensions
        self.add_slider("Width", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30, 
                        10, 100, self.maze_width, lambda v: self.set_maze_width(int(v)))
        y_offset += 50
        self.add_slider("Height", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        10, 100, self.maze_height, lambda v: self.set_maze_height(int(v)))
        y_offset += 50
        
        # Density (for Eller's and Hybrid)
        self.add_slider("Density", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        0.1, 0.9, self.density, lambda v: self.set_density(v))
        y_offset += 50
        
        # Section size (for Hybrid)
        self.add_slider("Section Size", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        3, 20, self.section_size, lambda v: self.set_section_size(int(v)))
        y_offset += 50
        
        # Animation speed
        self.add_slider("Animation Speed", sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30,
                        1, 50, self.animation_speed, lambda v: self.set_animation_speed(int(v)))
        y_offset += 50
        
        # Theme selection
        self.theme_rect = pygame.Rect(sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 30)
        theme_names = MazeTheme.get_theme_names()
        theme_buttons_width = (SIDEBAR_WIDTH - 20) // len(theme_names)
        
        for i, theme_name in enumerate(theme_names):
            self.add_button(theme_name[:1], 
                           sidebar_x + 10 + i * theme_buttons_width, 
                           y_offset, 
                           theme_buttons_width, 30,
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
        self.add_button("Save as PNG", sidebar_x + 10, y_offset, (SIDEBAR_WIDTH - 30) // 2, 30,
                        lambda: self.save_maze("png"))
        self.add_button("Save as TXT", sidebar_x + 20 + (SIDEBAR_WIDTH - 30) // 2, y_offset,
                       (SIDEBAR_WIDTH - 30) // 2, 30, lambda: self.save_maze("txt"))
        y_offset += 60
        
        # Status area
        self.status_rect = pygame.Rect(sidebar_x + 10, y_offset, SIDEBAR_WIDTH - 20, 100)
        
        # Initialize drawing mode state if it doesn't exist
        if not hasattr(self, 'drawing_mode'):
            self.drawing_mode = False
            self.user_path = []


# Add the CustomPathMaze class for generating mazes with user-defined paths
class CustomPathMaze(BaseMaze):
    """Maze generator that incorporates a user-defined path."""
    
    def __init__(self, width: int, height: int, path: List[Tuple[int, int]], 
                algorithm: MazeAlgorithm, density: float = 0.5, section_size: int = 5):
        """Initialize the custom path maze generator."""
        super().__init__(width, height)
        self.path = path
        self.algorithm = algorithm
        self.density = density
        self.section_size = section_size
    
    def generate(self) -> List[List[str]]:
        """Generate a maze that incorporates the user-defined path."""
        start_time = time.time()
        
        # Initialize with a grid of all walls
        self.maze = [['#' for _ in range(self.maze_width)] for _ in range(self.maze_height)]
        self.generation_steps = []
        
        # First, create the user-defined path
        self._create_user_path()
        
        # Save initial state with user path
        self.generation_steps.append([row[:] for row in self.maze])
        
        # Now generate the rest of the maze around the path
        if self.algorithm == MazeAlgorithm.ELLERS:
            self._generate_around_path_eller()
        elif self.algorithm == MazeAlgorithm.BACKTRACKER:
            self._generate_around_path_backtracker()
        else:  # Hybrid
            self._generate_around_path_hybrid()
        
        self.create_entrance_exit()
        
        # Save final state
        self.generation_steps.append([row[:] for row in self.maze])
        
        self.generation_time = time.time() - start_time
        return self.maze
    
    def _create_user_path(self) -> None:
        """Create the initial path based on user input."""
        # First create cells at all path points
        for x, y in self.path:
            self.maze[2*y+1][2*x+1] = ' '
        
        # Then connect adjacent points
        for i in range(len(self.path) - 1):
            x1, y1 = self.path[i]
            x2, y2 = self.path[i+1]
            
            # Only allow orthogonally adjacent points
            dx = x2 - x1
            dy = y2 - y1
            
            # For simplicity, we'll only handle immediate neighbors
            if abs(dx) + abs(dy) == 1:
                # Remove the wall between cells
                wall_x = 2*x1 + 1 + dx
                wall_y = 2*y1 + 1 + dy
                self.maze[wall_y][wall_x] = ' '
            else:
                # For non-adjacent points, create a straight path between them
                steps = max(abs(dx), abs(dy))
                for step in range(1, steps + 1):
                    # Calculate intermediate position
                    t = step / steps
                    ix = int(x1 + dx * t)
                    iy = int(y1 + dy * t)
                    
                    # Create cell at this position
                    self.maze[2*iy+1][2*ix+1] = ' '
                    
                    # Connect to previous cell
                    if step > 1:
                        prev_x = int(x1 + dx * (step - 1) / steps)
                        prev_y = int(y1 + dy * (step - 1) / steps)
                        
                        wall_x = 2*prev_x + 1 + (1 if ix > prev_x else (-1 if ix < prev_x else 0))
                        wall_y = 2*prev_y + 1 + (1 if iy > prev_y else (-1 if iy < prev_y else 0))
                        
                        self.maze[wall_y][wall_x] = ' '
    
    def _generate_around_path_eller(self) -> None:
        """Generate the rest of the maze using Eller's algorithm while preserving the user path."""
        # Create a grid that marks which cells are already part of the path
        path_cells = set()
        for x, y in self.path:
            path_cells.add((x, y))
        
        # Process each row
        row = [i for i in range(self.width)]  # Each cell starts in its own set
        next_set = self.width
        
        # Initialize sets based on user path connections
        for y in range(self.height):
            # Check horizontal connections in this row
            for x in range(self.width - 1):
                # If cells are already connected in the user path, merge their sets
                if (self.maze[2*y+1][2*x+2] == ' '):
                    # Cells are connected, merge their sets
                    old_set = row[x+1]
                    new_set = row[x]
                    for i in range(self.width):
                        if row[i] == old_set:
                            row[i] = new_set
            
            # Save current state for animation
            self.generation_steps.append([row[:] for row in self.maze])
            
            # Randomly connect remaining cells in the row
            for x in range(self.width - 1):
                # Only connect if cells are in different sets and not part of the user path
                if (row[x] != row[x+1] and random.random() < self.density and
                    self.maze[2*y+1][2*x+2] == '#'):  # Wall exists (not part of user path)
                    
                    # Remove the wall between cells
                    self.maze[2*y+1][2*x+2] = ' '
                    
                    # Merge sets
                    old_set = row[x+1]
                    new_set = row[x]
                    for i in range(self.width):
                        if row[i] == old_set:
                            row[i] = new_set
                    
                    # Save state after each horizontal connection
                    self.generation_steps.append([row[:] for row in self.maze])
            
            # Last row connects all different sets horizontally
            if y == self.height - 1:
                for x in range(self.width - 1):
                    if row[x] != row[x+1]:
                        # Remove the wall between cells
                        self.maze[2*y+1][2*x+2] = ' '
                        # Save state
                        self.generation_steps.append([row[:] for row in self.maze])
                continue
            
            # Group cells by set
            sets = {}
            for x, set_id in enumerate(row):
                if set_id not in sets:
                    sets[set_id] = []
                sets[set_id].append(x)
            
            # Initialize the next row with all cells in their own set
            next_row = [-1] * self.width
            
            # Check for existing vertical connections from the user path
            for x in range(self.width):
                if self.maze[2*y+2][2*x+1] == ' ':  # Vertical connection exists
                    next_row[x] = row[x]  # Pass the set ID to the cell below
            
            # For each set, randomly connect some cells to the row below
            for set_id, cells in sets.items():
                # Filter out cells that are already connected from user path
                unconnected_cells = [x for x in cells if next_row[x] == -1]
                
                if unconnected_cells:
                    # Connect at least one cell from each set
                    vertical_density = max(1, int(len(unconnected_cells) * (1 - self.density) + 0.5))
                    connect_count = max(1, min(len(unconnected_cells), vertical_density))
                    cells_to_connect = random.sample(unconnected_cells, connect_count)
                    
                    for x in cells_to_connect:
                        # Remove the wall below the cell
                        self.maze[2*y+2][2*x+1] = ' '
                        
                        # Pass the set ID to the cell below
                        next_row[x] = set_id
                        
                        # Save state after each vertical connection
                        self.generation_steps.append([row[:] for row in self.maze])
            
            # Assign new set IDs to unconnected cells in the next row
            for x in range(self.width):
                if next_row[x] == -1:
                    next_row[x] = next_set
                    next_set += 1
            
            # Update the current row for the next iteration
            row = next_row
    
    def _generate_around_path_backtracker(self) -> None:
        """Generate the rest of the maze using Recursive Backtracker while preserving the user path."""
        # Create a set of all cells in the path
        path_cells = set()
        for x, y in self.path:
            path_cells.add((x, y))
        
        # Create a set of all cells that have been visited (including path cells)
        visited = set(path_cells)
        
        # Find cells that are part of the path and have open passages
        frontier = []
        for x, y in path_cells:
            # Check if this cell has any walls that can be carved
            has_unvisited_neighbor = False
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # North, East, South, West
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    has_unvisited_neighbor = True
                    break
            
            if has_unvisited_neighbor:
                frontier.append((x, y))
        
        # If no frontier cells, pick any unvisited cell
        if not frontier and len(visited) < self.width * self.height:
            for y in range(self.height):
                for x in range(self.width):
                    if (x, y) not in visited:
                        # Create cell and add to frontier
                        self.maze[2*y+1][2*x+1] = ' '
                        frontier.append((x, y))
                        visited.add((x, y))
                        break
                if frontier:
                    break
        
        # Now perform recursive backtracking from the frontier cells
        while frontier:
            # Get the current cell
            x, y = frontier[-1]
            
            # Mark current cell for visualization
            current_maze = [row[:] for row in self.maze]
            current_maze[2*y+1][2*x+1] = 'C'  # Mark current cell
            self.generation_steps.append(current_maze)
            
            # Find unvisited neighbors
            neighbors = []
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # North, East, South, West
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                # Choose a random unvisited neighbor
                nx, ny, dx, dy = random.choice(neighbors)
                
                # Create the cell
                self.maze[2*ny+1][2*nx+1] = ' '
                
                # Remove the wall between current cell and chosen neighbor
                self.maze[2*y+1+dy][2*x+1+dx] = ' '
                
                # Add to frontier and mark as visited
                frontier.append((nx, ny))
                visited.add((nx, ny))
            else:
                # Backtrack if no unvisited neighbors
                frontier.pop()
                
                # Save state after backtracking
                if frontier:
                    x, y = frontier[-1]
                    current_maze = [row[:] for row in self.maze]
                    current_maze[2*y+1][2*x+1] = 'B'  # Mark backtracking cell
                    self.generation_steps.append(current_maze)
    
    def _generate_around_path_hybrid(self) -> None:
        """Generate the rest of the maze using the Hybrid algorithm while preserving the user path."""
        # First generate using Eller's algorithm
        self._generate_around_path_eller()
        
        # Then rework random sections using backtracker
        sections_x = self.width // self.section_size
        sections_y = self.height // self.section_size
        
        for _ in range(max(1, min(10, sections_x * sections_y // 3))):
            # Choose a random section
            section_x = random.randint(0, max(0, sections_x - 1))
            section_y = random.randint(0, max(0, sections_y - 1))
            
            # Apply recursive backtracker to the section
            start_x = section_x * self.section_size
            start_y = section_y * self.section_size
            end_x = min(start_x + self.section_size, self.width)
            end_y = min(start_y + self.section_size, self.height)
            
            # Skip sections that heavily overlap with the user path
            path_cells_in_section = sum(1 for x, y in self.path 
                                      if start_x <= x < end_x and start_y <= y < end_y)
            section_area = (end_x - start_x) * (end_y - start_y)
            
            if path_cells_in_section / section_area < 0.3:  # Less than 30% overlap
                # Highlight the section being reworked
                highlight_maze = [row[:] for row in self.maze]
                for y in range(start_y, end_y):
                    for x in range(start_x, end_x):
                        if highlight_maze[2*y+1][2*x+1] == ' ':
                            highlight_maze[2*y+1][2*x+1] = 'H'  # Highlight cell
                self.generation_steps.append(highlight_maze)
                
                self._rework_section_preserving_path(start_x, start_y, end_x, end_y)
    
    def _rework_section_preserving_path(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        """Rework a section while preserving the user path."""
        # Create a set of cells that are part of the user path
        path_cells = set()
        path_walls = set()
        
        for i in range(len(self.path)):
            x, y = self.path[i]
            if start_x <= x < end_x and start_y <= y < end_y:
                path_cells.add((x, y))
                
                # If there's a next cell in the path, mark the wall between them
                if i < len(self.path) - 1:
                    nx, ny = self.path[i+1]
                    if abs(nx - x) + abs(ny - y) == 1:  # Adjacent cells
                        wall_x = 2*x + 1 + (nx - x)
                        wall_y = 2*y + 1 + (ny - y)
                        path_walls.add((wall_x, wall_y))
        
        # Reset the section to walls except for the cells
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                # Keep cells
                self.maze[2*y+1][2*x+1] = ' '
                
                # Reset walls unless they're part of the path
                if x < end_x - 1:
                    wall_pos = (2*x+2, 2*y+1)
                    if wall_pos not in path_walls:
                        self.maze[wall_pos[1]][wall_pos[0]] = '#'
                
                if y < end_y - 1:
                    wall_pos = (2*x+1, 2*y+2)
                    if wall_pos not in path_walls:
                        self.maze[wall_pos[1]][wall_pos[0]] = '#'
        
        # Save state after section reset
        self.generation_steps.append([row[:] for row in self.maze])
        
        # Apply recursive backtracker to the section
        stack = []
        visited = set(path_cells)  # Start with path cells already visited
        
        # Start at a random cell in the path if possible, otherwise any cell
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
            
            # Mark current cell for visualization
            current_maze = [row[:] for row in self.maze]
            current_maze[2*y+1][2*x+1] = 'C'  # Mark current cell
            self.generation_steps.append(current_maze)
            
            # Find unvisited neighbors within the section
            neighbors = []
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (start_x <= nx < end_x and start_y <= ny < end_y and 
                    (nx, ny) not in visited):
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                
                # Check if the wall is part of the path
                wall_x = 2*x + 1 + dx
                wall_y = 2*y + 1 + dy
                
                if (wall_x, wall_y) not in path_walls:
                    # Remove the wall
                    self.maze[wall_y][wall_x] = ' '
                
                stack.append((nx, ny))
                visited.add((nx, ny))
                
                # Save state after carving a passage
                self.generation_steps.append([row[:] for row in self.maze])
            else:
                stack.pop()
                
                # Save state after backtracking
                if stack:
                    x, y = stack[-1]
                    current_maze = [row[:] for row in self.maze]
                    current_maze[2*y+1][2*x+1] = 'B'  # Mark backtracking cell
                    self.generation_steps.append(current_maze)
        
        # Clean up temporary cell markers
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                if self.maze[2*y+1][2*x+1] in ['C', 'B', 'H']:
                    self.maze[2*y+1][2*x+1] = ' '
