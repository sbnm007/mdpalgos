
from experiments import run_experiments, plot_aggregated_results
import metrics 

def main():
    # Define maze sizes and number of runs.
    maze_sizes = [5,10,15,25,50]  
    num_runs = 5

    # Run experiments, returning both aggregated and detailed results
    aggregated_results, detailed_results = run_experiments(maze_sizes, num_runs=num_runs)

    # Basic aggregated plots (runtime, path length)
    plot_aggregated_results(aggregated_results)


    # Nodes Expanded vs. Maze Size
    metrics.plot_nodes_expanded(aggregated_results)

    # Value Iteration Stats (iterations, state updates)
    metrics.plot_vi_stats(aggregated_results)

    #Path Length vs. Runtime (Scatter)
    metrics.plot_path_length_vs_time(detailed_results)

    #Runtime Distribution (Box Plot)
    metrics.plot_runtime_boxplots(detailed_results)

    # Side-by-Side Bar Charts by Maze Size (for runtime)
    metrics.plot_bar_charts_by_size(aggregated_results)

    #Nodes Expanded vs. Path Length
    metrics.plot_nodes_expanded_vs_path_length(detailed_results)

    print("Algorithms Evaluated, Check Plots in Results Folder.")

if __name__ == "__main__":
    main()
