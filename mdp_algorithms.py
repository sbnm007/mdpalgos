import numpy as np
import pandas as pd


def print_policy(policy, shape):
    actions = ['←', '→', '↑', '↓']
    policy_grid = np.array([[actions[a] if a != -1 else '#' for a in row] for row in policy])
    print("\nPolicy:")
    for row in policy_grid:
        print(" ".join(row))


def reconstruct_path(policy, start, goal):
    actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
    path = []
    current = start
    visited = set()
    while current != goal and current not in visited:
        path.append(current)
        visited.add(current)
        action = policy[current[0], current[1]]
        if action == -1:
            break
        dr, dc = actions[action]
        current = (current[0] + dr, current[1] + dc)
    if current == goal:
        path.append(goal)
    return path


def value_iteration(grid, start, goal, gamma, theta=1e-6, max_iterations=10000):
    rows, cols = grid.shape
    values = np.zeros((rows, cols), dtype=np.float64)
    policy = np.full((rows, cols), -1, dtype=np.int8)
    actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
    state_updates = 0

    for iteration in range(max_iterations):
        delta = 0
        new_values = values.copy()

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1 or (r, c) == goal:
                    continue

                best_value = float('-inf')
                best_action = -1

                for a, (dr, dc) in enumerate(actions):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 1:
                        next_state = (nr, nc)
                        if next_state == goal:
                            reward = 10 # Reaching the goal
                        else:
                            reward = -0.01  # Moving to a free space
                    else:
                        next_state = (r, c)  # Stay put if hitting wall or out of bounds
                        reward = -10 # Hitting a wall or out of bounds

                    v = reward + gamma * values[next_state[0]][next_state[1]]
                    if v > best_value:
                        best_value = v
                        best_action = a

                if best_action != -1:  # Only update if a valid action exists
                    if abs(new_values[r, c] - best_value) > theta:
                        state_updates += 1
                    new_values[r, c] = best_value
                    policy[r, c] = best_action
                delta = max(delta, abs(new_values[r, c] - values[r, c]))

        values = new_values
        if delta < theta:
            break
    else:
        print(f"Value Iteration stopped after max iterations ({max_iterations})")

    path = reconstruct_path(policy, start, goal)
    return values, policy, iteration + 1, state_updates, path


def policy_iteration(grid, start, goal, gamma, theta=1e-6, max_policy_iterations=1000, max_evaluation_iterations=10000):
    
    rows, cols = grid.shape
    values = np.zeros((rows, cols), dtype=np.float64)
    policy = np.full((rows, cols), -1, dtype=np.int8)
    actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
    state_updates = 0  # Initialize state updates

    # Initialize policy for open states to a default action (action 0: Left)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0 and (r, c) != goal:
                policy[r, c] = 0

    policy_iteration_count = 0
    while policy_iteration_count < max_policy_iterations:
        # **Policy Evaluation**
        evaluation_iteration = 0
        while evaluation_iteration < max_evaluation_iterations:
            delta = 0
            for r in range(rows):
                for c in range(cols):
                    if grid[r, c] == 1 or (r, c) == goal:  # Skip walls and goal
                        continue
                    action = policy[r, c]
                    dr, dc = actions[action]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 1:
                        next_state = (nr, nc)
                        if next_state == goal:
                            reward = 10  # Reaching the goal (adjusted to match value iteration)
                        else:
                            reward = -0.01  # Moving to a free space
                    else:
                        next_state = (r, c)  # Stay put if hitting wall or out of bounds
                        reward = -10  # Hitting a wall or out of bounds
                    new_v = reward + gamma * values[next_state[0]][next_state[1]]
                    if abs(new_v - values[r, c]) > theta:  # If state value changes significantly
                        state_updates += 1
                    delta = max(delta, abs(new_v - values[r, c]))
                    values[r, c] = new_v
            evaluation_iteration += 1
            if delta < theta:
                break
        if evaluation_iteration == max_evaluation_iterations:
            print(f"Policy evaluation did not converge after {max_evaluation_iterations} iterations")

        # **Policy Improvement**
        policy_stable = True
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1 or (r, c) == goal:  # Skip walls and goal
                    continue
                old_action = policy[r, c]
                best_action = None
                best_value = float('-inf')
                for a, (dr, dc) in enumerate(actions):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 1:
                        next_state = (nr, nc)
                        if next_state == goal:
                            reward = 10  # Reaching the goal (adjusted to match value iteration)
                        else:
                            reward = -0.01  # Moving to a free space
                    else:
                        next_state = (r, c)  # Stay put if hitting wall or out of bounds
                        reward = -10  # Hitting a wall or out of bounds
                    v = reward + gamma * values[next_state[0]][next_state[1]]
                    if v > best_value:
                        best_value = v
                        best_action = a
                if best_action is not None and best_action != old_action:
                    policy[r, c] = best_action
                    policy_stable = False

        policy_iteration_count += 1
        if policy_stable:
            break
    else:
        print(f"Policy Iteration did not converge after {max_policy_iterations} iterations")

    path = reconstruct_path(policy, start, goal)
    return values, policy, policy_iteration_count, state_updates, path


def load_grid_from_csv(filename):
    """Load a maze grid from a CSV file."""
    return np.array(pd.read_csv(filename, header=None), dtype=np.int8)


def main():
    filename = input("Enter CSV filename: ")
    grid = load_grid_from_csv(filename)
    print(f"Grid shape: {grid.shape}")

    start_input = input("Enter start state as 'row,col' (default: 1,1): ")
    goal_input = input(f"Enter goal state as 'row,col' (default: {grid.shape[0]-2},{grid.shape[1]-2}): ")

    if start_input:
        start_row, start_col = map(int, start_input.split(','))
        start_state = (start_row, start_col)
    else:
        start_state = (1, 1)

    if goal_input:
        goal_row, goal_col = map(int, goal_input.split(','))
        goal_state = (goal_row, goal_col)
    else:
        goal_state = (grid.shape[0]-2, grid.shape[1]-2)

    # Get gamma value
    gamma_input = input("Enter gamma value (default: 0.9): ")
    gamma = float(gamma_input) if gamma_input else 0.99

    # Run algorithms
    V_vi, policy_vi, iters_vi, state_updates_vi, path_vi = value_iteration(grid, start_state, goal_state, gamma)
    V_pi, policy_pi, iters_pi, state_updates_pi, path_pi = policy_iteration(grid, start_state, goal_state, gamma)

    # Print results
    print("\n=== VALUE ITERATION ===")
    print_policy(policy_vi, grid.shape)
    print(f"Iterations: {iters_vi}")
    print(f"State updates: {state_updates_vi}")
    print(f"Path: {path_vi}")

    print("\n=== POLICY ITERATION ===")
    print_policy(policy_pi, grid.shape)
    print(f"Iterations: {iters_pi}")
    print(f"State updates: {state_updates_pi}")
    print(f"Path: {path_pi}")

if __name__ == "__main__":
    main()