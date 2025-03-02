import numpy as np
import random

class Maze:
    def __init__(self, cells_w, cells_h):
        self.cells_w = cells_w
        self.cells_h = cells_h
        self.grid_w = 2 * cells_w + 1
        self.grid_h = 2 * cells_h + 1
        self.grid = np.ones((self.grid_h, self.grid_w), dtype=int)

    def generate(self):
        visited = [[False for _ in range(self.cells_w)] for _ in range(self.cells_h)]
        stack = [(0, 0)]
        visited[0][0] = True
        self.grid[1, 1] = 0

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.cells_w and 0 <= ny < self.cells_h and not visited[ny][nx]:
                    neighbors.append((nx, ny, dx, dy))
            if neighbors:
                nx, ny, dx, dy = random.choice(neighbors)
                visited[ny][nx] = True
                # Remove wall between current cell and neighbor:
                grid_x, grid_y = 2 * x + 1, 2 * y + 1
                self.grid[grid_y + dy, grid_x + dx] = 0
                # Mark neighbor cell
                self.grid[2 * ny + 1, 2 * nx + 1] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return self.grid

if __name__ == "__main__":
    # Simple test of maze generation.
    maze = Maze(5, 5)
    grid = maze.generate()

    # Save grid to CSV file
    np.savetxt('maze_test.csv', grid, delimiter=',', fmt='%d')
    print(grid)
