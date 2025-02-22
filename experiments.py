import csv
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
from collections import deque
from search_algorithms import dfs, bfs, a_star
from mdp_algorithms import build_mdp_model, value_iteration, policy_iteration

def load_maze(filename):
    """Loads a maze from a CSV file."""
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        maze = np.array([list(map(int, row)) for row in reader])
    return maze

def evaluate_algorithms(maze, start, goal):
    """Runs and times all algorithms on the given maze."""
    results = {}
    
    # DFS
    start_time = time.time()
    dfs_path = dfs(maze, start, goal)
    results['DFS'] = {'time': time.time() - start_time, 'path_length': len(dfs_path) if dfs_path else None}
    
    # BFS
    start_time = time.time()
    bfs_path = bfs(maze, start, goal)
    results['BFS'] = {'time': time.time() - start_time, 'path_length': len(bfs_path) if bfs_path else None}
    
    # A*
    start_time = time.time()
    a_star_path = a_star(maze, start, goal)
    results['A*'] = {'time': time.time() - start_time, 'path_length': len(a_star_path) if a_star_path else None}
    
    # MDP Value Iteration
    states, transitions, rewards = build_mdp_model(maze, goal, step_reward=-1, goal_reward=10)
    start_time = time.time()
    V_vi, policy_vi = value_iteration(states, transitions, rewards)
    results['Value Iteration'] = {'time': time.time() - start_time}
    
    # MDP Policy Iteration
    start_time = time.time()
    try:
        V_pi, policy_pi = policy_iteration(states, transitions, rewards)
        results['Policy Iteration'] = {'time': time.time() - start_time}
    except ValueError as e:
        results['Policy Iteration'] = {'time': None, 'error': str(e)}
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze", type=str, required=True, help="Path to the maze CSV file")
    parser.add_argument("--start", type=int, nargs=2, required=True, help="Start position (row col)")
    parser.add_argument("--goal", type=int, nargs=2, required=True, help="Goal position (row col)")
    parser.add_argument("--display", action="store_true", help="Display the maze solution")
    args = parser.parse_args()
    
    maze = load_maze(args.maze)
    start = tuple(args.start)
    goal = tuple(args.goal)
    
    results = evaluate_algorithms(maze, start, goal)
    
    for algo, res in results.items():
        if 'error' in res:
            print(f"{algo}: Error - {res['error']}")
        else:
            print(f"{algo}: Time = {res['time']:.6f} sec", end="")
            if 'path_length' in res:
                print(f", Path Length = {res['path_length']}")
            else:
                print()

if __name__ == "__main__":
    main()
