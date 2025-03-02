import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from maze_generator import Maze
from search_algorithms import dfs, bfs, astar
from mdp_algorithms import value_iteration, policy_iteration


RESULTS_DIR = "results1"
os.makedirs(RESULTS_DIR, exist_ok=True)
IMAGE_DIR = os.path.join(RESULTS_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

def visualize_path(maze_grid, path, algorithm_name, maze_size, run_id):
    
    plt.figure(figsize=(8, 8))
    plt.imshow(maze_grid, cmap="binary")
    if path:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        plt.plot(cols, rows, color="red", linewidth=2, label=f"{algorithm_name} Path")
    plt.title(f"Maze {maze_size}x{maze_size} - {algorithm_name} (Run {run_id})")
    plt.legend()
    filename = os.path.join(IMAGE_DIR, f"{algorithm_name.lower()}_maze{maze_size}_run{run_id}.png")
    plt.savefig(filename)
    plt.close()

def visualize_value(maze_grid, V, maze_size, run_id):
    value_array = np.full(maze_grid.shape, np.nan)
    for (r, c), val in np.ndenumerate(V):
        value_array[r, c] = val
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="black")
    plt.imshow(value_array, cmap=cmap)
    plt.colorbar(label="Value")
    plt.title(f"Maze {maze_size}x{maze_size} - Value Function (Run {run_id})")
    filename = os.path.join(IMAGE_DIR, f"value_maze{maze_size}_run{run_id}.png")
    plt.savefig(filename)
    plt.close()

def visualize_policy(maze_grid, policy, maze_size, run_id):
    X, Y, U, V_dir = [], [], [], []
    for (r, c), action in np.ndenumerate(policy):
        if action == -1 or maze_grid[r, c] != 0:
            continue
        X.append(c)
        Y.append(r)
        if action == 0:
            U.append(-0.5)
            V_dir.append(0)
        elif action == 1:
            U.append(0.5)
            V_dir.append(0)
        elif action == 2:
            U.append(0)
            V_dir.append(-0.5)
        elif action == 3:
            U.append(0)
            V_dir.append(0.5)
    if not U:
        print("No valid actions found for policy visualization.")
        return
    plt.figure(figsize=(8, 8))
    plt.imshow(maze_grid, cmap="binary")
    plt.quiver(X, Y, U, V_dir, color="blue", scale=1, scale_units="xy", angles="xy")
    plt.title(f"Maze {maze_size}x{maze_size} - Policy (Run {run_id})")
    filename = os.path.join(IMAGE_DIR, f"policy_maze{maze_size}_run{run_id}.png")
    plt.savefig(filename)
    plt.close()

def visualize_maze(maze_grid, maze_size, run_id):
    plt.figure(figsize=(8, 8))
    plt.imshow(maze_grid, cmap="binary")
    plt.title(f"Maze {maze_size}x{maze_size} (Run {run_id})")
    filename = os.path.join(IMAGE_DIR, f"maze_{maze_size}_run{run_id}.png")
    plt.savefig(filename)
    plt.close()

def run_single_experiment(maze_cells, run_id):
    maze_obj = Maze(maze_cells, maze_cells)
    maze_grid = maze_obj.generate()

    visualize_maze(maze_grid, maze_cells, run_id)

    start = (1, 1)
    goal = (maze_grid.shape[0] - 2, maze_grid.shape[1] - 2)
    metrics = {"maze_cells": maze_cells}

    # DFS
    t0 = time.time()
    path_dfs, nodes_dfs = dfs(maze_grid, start, goal)
    dfs_time = time.time() - t0
    metrics["DFS_time"] = dfs_time
    metrics["DFS_path_length"] = len(path_dfs) if path_dfs else None
    metrics["DFS_nodes_expanded"] = nodes_dfs

    # BFS
    t0 = time.time()
    path_bfs, nodes_bfs = bfs(maze_grid, start, goal)
    bfs_time = time.time() - t0
    metrics["BFS_time"] = bfs_time
    metrics["BFS_path_length"] = len(path_bfs) if path_bfs else None
    metrics["BFS_nodes_expanded"] = nodes_bfs

    # A*
    t0 = time.time()
    path_astar, nodes_astar = astar(maze_grid, start, goal)
    astar_time = time.time() - t0
    metrics["Astar_time"] = astar_time
    metrics["Astar_path_length"] = len(path_astar) if path_astar else None
    metrics["Astar_nodes_expanded"] = nodes_astar

    # MDP Value Iteration
    t0 = time.time()
    V_vi, policy_vi, iters_vi, state_updates_vi, path_vi = value_iteration(maze_grid, start, goal, 0.99)
    vi_time = time.time() - t0
    metrics["VI_time"] = vi_time
    metrics["VI_path_length"] = len(path_vi) if path_vi else None
    metrics["VI_iterations"] = iters_vi
    metrics["VI_state_updates"] = state_updates_vi

    # MDP Policy Iteration
    t0 = time.time()
    V_pi, policy_pi, iters_pi, state_updates_pi, path_pi = policy_iteration(maze_grid, start, goal, 0.99)
    pi_time = time.time() - t0
    metrics["PI_time"] = pi_time
    metrics["PI_path_length"] = len(path_pi) if path_pi else None
    metrics["PI_iterations"] = iters_pi
    metrics["PI_state_updates"] = state_updates_pi

    return (metrics, maze_grid, path_dfs, path_bfs, path_astar,
            path_vi, path_pi, V_vi, policy_vi, V_pi, policy_pi)

def run_experiments(maze_sizes, num_runs=5):
    all_results = []
    detailed_results = []

    for size in maze_sizes:
        metrics_list = []
        print(f"Running experiments for maze size: {size}x{size}")
        for run in range(1, num_runs + 1):
            (metrics, maze_grid, path_dfs, path_bfs, path_astar,
             path_vi, path_pi, V_vi, policy_vi, V_pi, policy_pi) = run_single_experiment(size, run)

            metrics["run"] = run
            metrics_list.append(metrics)
            detailed_results.append(metrics)

            visualize_path(maze_grid, path_dfs, "DFS", size, run)
            visualize_path(maze_grid, path_bfs, "BFS", size, run)
            visualize_path(maze_grid, path_astar, "A*", size, run)
            visualize_path(maze_grid, path_vi, "ValueIteration", size, run)
            visualize_path(maze_grid, path_pi, "PolicyIteration", size, run)
            visualize_value(maze_grid, V_vi, size, run)
            visualize_policy(maze_grid, policy_pi, size, run)

        # Aggregate metrics for this maze size.
        agg = {"maze_cells": size}
        keys = [k for k in metrics_list[0].keys() if k not in ["maze_cells", "run"]]
        for key in keys:
            values = [m[key] for m in metrics_list if m[key] is not None]
            if values:
                agg[f"{key}_mean"] = np.mean(values)
                agg[f"{key}_std"]  = np.std(values)
            else:
                agg[f"{key}_mean"] = None
                agg[f"{key}_std"]  = None
        all_results.append(agg)

    # Save detailed results to CSV inside the results folder.
    csv_file = os.path.join(RESULTS_DIR, "maze_experiment_data_detailed.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detailed_results[0].keys())
        writer.writeheader()
        for row in detailed_results:
            writer.writerow(row)
    print(f"Detailed experiment data saved to {csv_file}")

    return all_results, detailed_results

def plot_aggregated_results(aggregated_results):
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 14})

    sizes = [res["maze_cells"] for res in aggregated_results]

    algo_mapping = {
        "DFS": "DFS",
        "BFS": "BFS",
        "A*":  "Astar",
        "VI":  "VI",
        "PI":  "PI"
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    for display_name, key_name in algo_mapping.items():
        times = [res.get(f"{key_name}_time_mean", 0) for res in aggregated_results]
        stds  = [res.get(f"{key_name}_time_std", 0)  for res in aggregated_results]
        ax.errorbar(
            sizes, times, yerr=stds,
            marker='o', markersize=8,
            linewidth=2, capsize=5,
            label=f"{display_name} Time"
        )

    ax.set_xlabel("Maze Size (cells)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Average Runtime vs Maze Size (Log Scale)")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    filename = os.path.join(IMAGE_DIR, "aggregated_runtime.png")
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    maze_sizes = [10, 15, 20, 25, 50]
    num_runs = 5
    aggregated_results, detailed_results = run_experiments(maze_sizes, num_runs=num_runs)
    plot_aggregated_results(aggregated_results)
    csv_agg_file = os.path.join(RESULTS_DIR, "maze_experiment_data_aggregated.csv")
    with open(csv_agg_file, "w", newline="") as f:
        fieldnames = aggregated_results[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_results:
            writer.writerow(row)
    print(f"Aggregated experiment data saved to {csv_agg_file}")
