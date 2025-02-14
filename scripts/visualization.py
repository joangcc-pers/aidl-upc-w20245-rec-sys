import matplotlib.pyplot as plt

def plot_evaluation_metrics(results, top_k_values=[5, 10]):
    """
    Plot the evaluation metrics (MRR@K, Recall@K, Precision@K) over different top-K values.

    Args:
        results: Dictionary containing evaluation metrics for each top-K value.
        top_k_values: List of top-K values to plot.
    """
    # Unpack results for plotting
    mrr_at_k_vals = [results[top_k]['mrr'] for top_k in top_k_values]
    recall_at_k_vals = [results[top_k]['recall'] for top_k in top_k_values]
    precision_at_k_vals = [results[top_k]['precision'] for top_k in top_k_values]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(top_k_values, mrr_at_k_vals, label="MRR@K", marker='o', linestyle='-', color='blue')
    plt.plot(top_k_values, recall_at_k_vals, label="Recall@K", marker='o', linestyle='-', color='green')
    plt.plot(top_k_values, precision_at_k_vals, label="Precision@K", marker='o', linestyle='-', color='red')

    plt.xlabel('Top-K')
    plt.ylabel('Metric Value')
    plt.title('Evaluation Metrics vs Top-K')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
