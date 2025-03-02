from collections import deque
import heapq

def get_neighbors(maze_grid, pos):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    r, c = pos
    rows, cols = maze_grid.shape
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and maze_grid[nr, nc] == 0:
            neighbors.append((nr, nc))
    return neighbors

def dfs(maze_grid, start, goal):
    stack = [start]
    visited = set()
    parent = {}
    nodes_expanded = 0

    while stack:
        current = stack.pop()
        nodes_expanded += 1
        if current == goal:
            break
        if current in visited:
            continue
        visited.add(current)
        for neighbor in get_neighbors(maze_grid, current):
            if neighbor not in visited:
                stack.append(neighbor)
                if neighbor not in parent:
                    parent[neighbor] = current

    if current != goal:
        return None, nodes_expanded

    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent[node]
    path.append(start)
    path.reverse()
    return path, nodes_expanded

def bfs(maze_grid, start, goal):
    queue = deque([start])
    visited = {start}
    parent = {}
    nodes_expanded = 0

    while queue:
        current = queue.popleft()
        nodes_expanded += 1
        if current == goal:
            break
        for neighbor in get_neighbors(maze_grid, current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    if current != goal:
        return None, nodes_expanded

    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent[node]
    path.append(start)
    path.reverse()
    return path, nodes_expanded

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(maze_grid, start, goal):
    
    import heapq

    open_set = []
    # The heap will store tuples of (f_score, g_score, node)
    heapq.heappush(open_set, (manhattan(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    nodes_expanded = 0
    closed_set = set()

    while open_set:
        f, current_g, current = heapq.heappop(open_set)
        nodes_expanded += 1

        if current == goal:
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path, nodes_expanded

        if current in closed_set:
            continue
        closed_set.add(current)

        for neighbor in get_neighbors(maze_grid, current):
            tentative_g = current_g + 1
            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                priority = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_set, (priority, tentative_g, neighbor))
                came_from[neighbor] = current

    return None, nodes_expanded
