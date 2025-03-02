import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use the same RESULTS_DIR and IMAGE_DIR as in experiments.py.
RESULTS_DIR = "results"
IMAGE_DIR = os.path.join(RESULTS_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

def plot_nodes_expanded(aggregated_results):
    algo_mapping = {
        "DFS": "DFS",
        "BFS": "BFS",
        "A*":  "Astar",
        "VI":  "VI",
        "PI":  "PI"
    }
    sizes = [res["maze_cells"] for res in aggregated_results]

    plt.figure(figsize=(10, 6))
    for display_name, key_name in algo_mapping.items():
        means = [res.get(f"{key_name}_nodes_expanded_mean", 0) for res in aggregated_results]
        stds  = [res.get(f"{key_name}_nodes_expanded_std", 0)  for res in aggregated_results]
        plt.errorbar(sizes, means, yerr=stds, marker='o', capsize=5,
                     label=f"{display_name} Nodes Expanded")

    plt.xlabel("Maze Size (cells)")
    plt.ylabel("Nodes Expanded")
    plt.title("Average Number of Nodes Expanded vs Maze Size")
    plt.legend()
    plt.grid(True)
    filename = os.path.join(IMAGE_DIR, "aggregated_nodes_expanded.png")
    plt.savefig(filename)
    plt.close()

def plot_vi_stats(aggregated_results):
    sizes = [res["maze_cells"] for res in aggregated_results]

    # VI Iterations
    vi_iters_mean = [res.get("VI_iterations_mean", 0) for res in aggregated_results]
    vi_iters_std  = [res.get("VI_iterations_std", 0)  for res in aggregated_results]

    plt.figure(figsize=(10,6))
    plt.errorbar(sizes, vi_iters_mean, yerr=vi_iters_std, marker='o', capsize=5,
                 label="VI Iterations")
    plt.xlabel("Maze Size (cells)")
    plt.ylabel("Iterations")
    plt.title("Value Iteration: Iterations vs Maze Size")
    plt.legend()
    plt.grid(True)
    filename = os.path.join(IMAGE_DIR, "vi_iterations.png")
    plt.savefig(filename)
    plt.close()

    # VI State Updates
    vi_updates_mean = [res.get("VI_state_updates_mean", 0) for res in aggregated_results]
    vi_updates_std  = [res.get("VI_state_updates_std", 0)  for res in aggregated_results]

    plt.figure(figsize=(10,6))
    plt.errorbar(sizes, vi_updates_mean, yerr=vi_updates_std, marker='o', capsize=5,
                 label="VI State Updates")
    plt.xlabel("Maze Size (cells)")
    plt.ylabel("Number of Updates")
    plt.title("Value Iteration: State Updates vs Maze Size")
    plt.legend()
    plt.grid(True)
    filename = os.path.join(IMAGE_DIR, "vi_state_updates.png")
    plt.savefig(filename)
    plt.close()

def plot_pi_stats(aggregated_results):
    sizes = [res["maze_cells"] for res in aggregated_results]

    # PI Iterations
    pi_iters_mean = [res.get("PI_iterations_mean", 0) for res in aggregated_results]
    pi_iters_std  = [res.get("PI_iterations_std", 0)  for res in aggregated_results]

    plt.figure(figsize=(10,6))
    plt.errorbar(sizes, pi_iters_mean, yerr=pi_iters_std, marker='o', capsize=5,
                 label="PI Iterations")
    plt.xlabel("Maze Size (cells)")
    plt.ylabel("Iterations")
    plt.title("Policy Iteration: Iterations vs Maze Size")
    plt.legend()
    plt.grid(True)
    filename = os.path.join(IMAGE_DIR, "pi_iterations.png")
    plt.savefig(filename)
    plt.close()

    # PI State Updates
    pi_updates_mean = [res.get("PI_state_updates_mean", 0) for res in aggregated_results]
    pi_updates_std  = [res.get("PI_state_updates_std", 0)  for res in aggregated_results]

    plt.figure(figsize=(10,6))
    plt.errorbar(sizes, pi_updates_mean, yerr=pi_updates_std, marker='o', capsize=5,
                 label="PI State Updates")
    plt.xlabel("Maze Size (cells)")
    plt.ylabel("State Updates")
    plt.title("Policy Iteration: State Updates vs Maze Size")
    plt.legend()
    plt.grid(True)
    filename = os.path.join(IMAGE_DIR, "pi_state_updates.png")
    plt.savefig(filename)
    plt.close()

def plot_path_length_vs_time(detailed_results):
    algo_keys = ["DFS", "BFS", "Astar", "VI", "PI"]

    plt.figure(figsize=(10, 6))
    for algo in algo_keys:
        path_lengths = []
        times = []
        for row in detailed_results:
            length_key = f"{algo}_path_length"
            time_key   = f"{algo}_time"
            if row.get(length_key) is not None and row.get(time_key) is not None:
                path_lengths.append(row[length_key])
                times.append(row[time_key])
        if path_lengths and times:
            plt.scatter(path_lengths, times, alpha=0.7, label=algo)
    plt.xlabel("Path Length (steps)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Path Length vs. Runtime (All Maze Sizes & Runs)")
    plt.legend()
    plt.grid(True)
    filename = os.path.join(IMAGE_DIR, "path_length_vs_runtime.png")
    plt.savefig(filename)
    plt.close()

def plot_runtime_boxplots(detailed_results):
    algo_keys = ["DFS", "BFS", "Astar", "VI", "PI"]
    time_data = []
    for row in detailed_results:
        for algo in algo_keys:
            time_key = f"{algo}_time"
            if row.get(time_key) is not None:
                time_data.append({
                    "algorithm": algo,
                    "time": row[time_key]
                })
    if not time_data:
        print("No time data found for boxplots. Skipping.")
        return
    df_time = pd.DataFrame(time_data)
    plt.figure(figsize=(10, 6))
    df_time.boxplot(column="time", by="algorithm", grid=True)
    plt.yscale('log')
    plt.title("Runtime Distribution by Algorithm")
    plt.suptitle("")
    plt.xlabel("Algorithm")
    plt.ylabel("Runtime (seconds)")
    filename = os.path.join(IMAGE_DIR, "runtime_boxplot.png")
    plt.savefig(filename)
    plt.close()

def plot_bar_charts_by_size(aggregated_results):
    algo_keys = ["DFS", "BFS", "Astar", "VI", "PI"]
    for res in aggregated_results:
        size = res["maze_cells"]
        means = [res.get(f"{algo}_time_mean", 0) for algo in algo_keys]
        plt.figure(figsize=(8,6))
        x_positions = np.arange(len(algo_keys))
        plt.bar(x_positions, means, width=0.6, align='center')
        plt.xticks(x_positions, algo_keys)
        plt.yscale('log')
        plt.ylabel("Runtime (seconds)")
        plt.title(f"Runtime Comparison for Maze Size = {size}")
        plt.grid(True, axis='y')
        filename = os.path.join(IMAGE_DIR, f"bar_runtime_size_{size}.png")
        plt.savefig(filename)
        plt.close()

def plot_nodes_expanded_vs_path_length(detailed_results):
    algo_keys = ["DFS", "BFS", "Astar", "VI", "PI"]
    plt.figure(figsize=(10, 6))
    for algo in algo_keys:
        expanded_key = f"{algo}_nodes_expanded"
        length_key   = f"{algo}_path_length"
        xs = []
        ys = []
        for row in detailed_results:
            if row.get(expanded_key) is not None and row.get(length_key) is not None:
                xs.append(row[expanded_key])
                ys.append(row[length_key])
        if xs and ys:
            plt.scatter(xs, ys, alpha=0.7, label=algo)
    plt.xlabel("Nodes Expanded")
    plt.ylabel("Path Length (steps)")
    plt.title("Nodes Expanded vs. Path Length (All Maze Sizes & Runs)")
    plt.legend()
    plt.grid(True)
    filename = os.path.join(IMAGE_DIR, "nodes_expanded_vs_path_length.png")
    plt.savefig(filename)
    plt.close()
