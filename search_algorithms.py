from collections import deque
import heapq
import numpy as np

def get_neighbors(x, y, maze):
    """
    Returns valid 4-directional neighbors (up, down, left, right).
    """
    neighbors = []
    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.shape[1] and 0 <= ny < maze.shape[0]:
            if maze[ny][nx] == 0:  # open cell
                neighbors.append((nx, ny))
    return neighbors

def dfs(maze, start, goal):
    """
    Depth-First Search: non-optimal, but easy to implement.
    :param maze: 2D numpy array (0=open, 1=wall)
    :param start: (x, y) start position
    :param goal: (x, y) goal position
    :return: list of path coordinates from start to goal or None if not found
    """
    stack = [start]
    visited = set([start])
    parent = {start: None}  # For reconstructing path

    while stack:
        current = stack.pop()
        if current == goal:
            # Reconstruct path
            return _reconstruct_path(parent, current)

        for n in get_neighbors(current[0], current[1], maze):
            if n not in visited:
                visited.add(n)
                parent[n] = current
                stack.append(n)
    return None

def bfs(maze, start, goal):
    """
    Breadth-First Search: guarantees shortest path in an unweighted maze.
    """
    queue = deque([start])
    visited = set([start])
    parent = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            return _reconstruct_path(parent, current)

        for n in get_neighbors(current[0], current[1], maze):
            if n not in visited:
                visited.add(n)
                parent[n] = current
                queue.append(n)
    return None

def a_star(maze, start, goal):
    """
    A* Search: uses a heuristic (Manhattan distance) to guide search.
    """
    open_list = []
    heapq.heappush(open_list, (0, start))
    visited = set()
    parent = {start: None}

    g_score = {start: 0}  # Cost from start
    f_score = {start: _heuristic(start, goal)}

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            return _reconstruct_path(parent, current)

        visited.add(current)

        for neighbor in get_neighbors(current[0], current[1], maze):
            tentative_g = g_score[current] + 1
            if neighbor in visited and tentative_g >= g_score.get(neighbor, float('inf')):
                continue
            if tentative_g < g_score.get(neighbor, float('inf')):
                parent[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + _heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))
    return None

def _heuristic(a, b):
    """
    Manhattan distance heuristic.
    """
    (x1, y1), (x2, y2) = a, b
    return abs(x1 - x2) + abs(y1 - y2)

def _reconstruct_path(parent, current):
    """
    Reconstruct path from start to goal using the parent dictionary.
    """
    path = []
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path
