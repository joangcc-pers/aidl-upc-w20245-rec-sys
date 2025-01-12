import pandas as pd

def clean_data_hierarchical_rnn(data, cleaning_params):
    if cleaning_params.get("remove_na", False):
        data = data.dropna()
    if cleaning_params.get("remove_duplicates", False):
        data = data.drop_duplicates()
    if cleaning_params.get("normalize_columns"):
        for column in cleaning_params["normalize_columns"]:
            data[column] = (data[column] - data[column].mean()) / data[column].std()
    if cleaning_params.get("filter_rows"):
        filter_config = cleaning_params["filter_rows"]
        data = data[data[filter_config["column"]].isin(filter_config["values"])]
    return data

def clean_data_graph_nn(data, cleaning_params):
    if cleaning_params.get("one_hot_encode_columns"):
        for column in cleaning_params["one_hot_encode_columns"]:
            data = pd.get_dummies(data, columns=[column])
    if cleaning_params.get("create_graph_edges", False):
        # Example: Create graph edges from some logic
        data["edges"] = data.apply(lambda x: (x["node1"], x["node2"]), axis=1)
    return data

def clean_data(input_path, output_path, cleaning_params, model_name):
    # Load raw data
    data = pd.read_csv(input_path)
    
    # Apply model-specific cleaning
    if model_name == "hierarchical_rnn":
        data = clean_data_hierarchical_rnn(data, cleaning_params)
    elif model_name == "graph_nn":
        data = clean_data_graph_nn(data, cleaning_params)
    elif model_name == "temporal_transformer":
        # Add custom cleaning logic for transformers
        pass
    
    # Save cleaned data
    data.to_csv(output_path, index=False)
    print(f"Cleaned data for {model_name} saved to {output_path}")