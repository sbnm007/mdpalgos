import csv
import numpy as np
import argparse
import matplotlib.pyplot as plt
import random
from collections import deque

class MazeGenerator:
    """Generates a maze using the Recursive Backtracking (DFS) algorithm."""
    def __init__(self, width=21, height=21, seed=None):
        """
        :param width:  Width of the maze (number of cells, should be odd)
        :param height: Height of the maze (number of cells, should be odd)
        :param seed:   Optional seed for reproducibility
        """
        # Ensure width and height are odd
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        if seed is not None:
            random.seed(seed)
        self.start_x, self.start_y = 1, 1  # Always start at (1,1)
        self.exit_x, self.exit_y = self._random_exit()
        self.maze = self._generate_maze()
        self._mark_start_end()

    def _random_exit(self):
        """Selects a random exit position on any edge except the start."""
        edge_positions = []
        for i in range(1, self.width - 1, 2):  # Avoid corners
            edge_positions.append((1, i))  # Top row
            edge_positions.append((self.height - 1, i))  # Bottom row
        for j in range(1, self.height - 1, 2):
            edge_positions.append((j, 1))  # Left column
            edge_positions.append((j, self.width - 1))  # Right column
        
        # Remove start position from possible exits
        edge_positions = [(x, y) for (x, y) in edge_positions if (x, y) != (self.start_x, self.start_y)]
        return random.choice(edge_positions)

    def _generate_maze(self):
        """
        Generates a maze using a simple recursive backtracker (DFS).
        0 = open path, 1 = wall
        """
        maze = np.ones((self.height, self.width), dtype=int)
        maze[self.start_y][self.start_x] = 0
        stack = [(self.start_x, self.start_y)]
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        while stack:
            cx, cy = stack[-1]
            random.shuffle(directions)
            neighbors = []
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.width - 1 and 0 < ny < self.height - 1 and maze[ny][nx] == 1:
                    neighbors.append((nx, ny))
            if neighbors:
                nx, ny = random.choice(neighbors)
                maze[(ny + cy) // 2][(nx + cx) // 2] = 0
                maze[ny][nx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze
    
    def _mark_start_end(self):
        """Marks the start and randomly chosen exit positions on the maze."""
        self.maze[self.start_y][self.start_x] = 2  # Mark start with 2
        self.maze[self.exit_y][self.exit_x] = 3  # Mark exit with 3

    def save_to_csv(self, filename="maze.csv"):
        """Saves the maze to a CSV file."""
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.maze)

    def visualize(self):
        """Visualizes the maze using Matplotlib."""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.maze, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.title("Generated Maze with Start and End")
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=21, help="Width of the maze (odd number)")
    parser.add_argument("--height", type=int, default=21, help="Height of the maze (odd number)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--save", type=str, default=None, help="Save maze to CSV file")
    parser.add_argument("--display", action="store_true", help="Display the generated maze")
    args = parser.parse_args()
    
    mg = MazeGenerator(args.width, args.height, seed=args.seed)
    if args.save:
        mg.save_to_csv(args.save)
        print(f"Maze saved to {args.save}")
    if args.display:
        mg.visualize()

if __name__ == "__main__":
    main()
