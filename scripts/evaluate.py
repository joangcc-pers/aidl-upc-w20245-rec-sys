def evaluate_and_print_results(model, test_loader, top_k_values=[5, 10]):
    """
    Evaluate the model on the test set using different values of K (e.g., 5, 10) and print the results.

    Args:
        model: The trained model.
        test_loader: The DataLoader for the test dataset.
        top_k_values: List of top-K values to evaluate (e.g., [5, 10]).
    """
    for top_k in top_k_values:
        print(f"\nEvaluating with Top-{top_k} Recommendations...")
        mrr_at_k, recall_at_k, precision_at_k = test_model(model, test_loader, top_k=top_k)

        # Print results for this top-k evaluation
        print(f"\n--- Results for Top-{top_k} ---")
        print(f"MRR@{top_k}: {mrr_at_k:.4f}")
        print(f"Recall@{top_k}: {recall_at_k:.4f}")
        print(f"Precision@{top_k}: {precision_at_k:.4f}")
