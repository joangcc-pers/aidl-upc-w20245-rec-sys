# Project: Artificial Intelligence with Deep Learning

This repository contains the development of a project carried out as part of the **Artificial Intelligence with Deep Learning** postgraduate program.

## Advisor

- Oscar Pina

## Authors

- Joan González-Conde Cantero
- Judit Llorens Diaz
- Miguel Palos Pou
- Xavier Rodriguez Moreno

## Project Description

This project aims to explore and apply Deep learning architectures for Recommender Systems, intrasession. In this repository, we provide the code to:

- Implement architectures for deep learning in RecSys.
- Process the needed data.
- Training, evaluation, visualize and compare the diffrenet architectures and configurations.

The implementaiton of these architectures will use pytorch, and visualize results through TensorBoard.

[Project repository](https://github.com/joangcc-pers/aidl-upc-w20245-rec-sys)

# 1. Motivation

//TODO

# 2. Problem to solve

//TODO

# 3. Dataset

//TODO

# 4. Achitectures

//TODO

# 5. Preprocessing and training

//TODO

# 6. Model evolutions

//TODO

# 7. Models benchmarking

The table below summarises the results benchmarking different model configurations using the same dataset (Oct 2019 and Nov 2019).

| Model | R@20 | MRR@20 |
|-------|------|--------|
| GRU4Rec (baseline) | 0.5293 | 0.2008 |
| SR_GNN (own implementation) | 0.5703| 0.3200 |
| Graph with Embeddings and Attentional Aggregation | 0.6003 | 0.3462 |

We created our own implementation of SR_GNN because the SR_GNN does not report results on the dataset we are using for the baseline and the assessment of our model.

We can see how our model "Graph with Embeddings and Attentional Aggregation" represents:
- A 5.26% of improvement in the Recall@20 
- A 8.19% of improvement in the MRR@20

# 8. Repository structure and MLOPS features

## Repository structure

The repository is organized as follows:
- **data**: Contains the data used in the project.
    - **raw**: Stores the original, unprocessed dataset downloaded from Kaggle.
    - **processed**: Stores the data after preprocessing and feature engineering. This data is ready to be used for training and evaluation.
- **experiments**: Contains configuration files and results for different experiments. 
    - **experiment_folder**: Each experiment has its own subfolder, where the following artifacts will be stored:
        - **graphsdb**: lmdb files storing the preprocessing outcome for each experiment.
        - **train_dataset.pth**: Train dataset generated during the data preprocessing.
        - **val_dataset.pth**: Validation dataset generated during the data preprocessing.
        - **test_dataset.pth**: Test dataset generated during the data preprocessing.
        - **logs**: Tensorboard files for the training of each experiment.
        - **trained_model_000x.pth**: Model checkpoints after each training epoch.
    - **config.yaml**: Contains the configuration for preprocessing and executing each experiment using `run_experiment.py`.
    - **config_hyp.yaml**: Containst the configuration for preprocessing and executing hyperparameter tuning using `run_optim.py`
- **models**: Contains the different deep leanning models implementations.
- **notebooks**: Folder to store Jupyter notebooks that can be useful to explore different parts of this project. 
- **scripts**: The scripts folder contains all the logic needed to implement the whole lifecycle of this machine learning project. Contains `preprocessing_scripts`, `train_scripts`, `evaluate_scripts` and `test_scripts`.
- **utils**: Helper classes and methods used in by other classes in the repository.
**run_experiment.py**: Entry point for running experiments defined in `experiments/config.yaml` file.
**run_optim.py**: Entry point for running grid search defined in `experiments/config-hyp.yaml` file.


## Contributing to the repository

- Create a new branch from `develop` to work on your contribution to this repository.
- Create a merge request and assign it to at least one of the authors.
- Merge requests will be reviewed, approved and merged to `develop`and `main` by the authors.
- Do **NOT** merge to `main` or `develop` directly.

Not sure where to start? Take a look at the [open issues](https://github.com/joangcc-pers/aidl-upc-w20245-rec-sys/issues) in our repository.

## Project setup
### Install project dependencies

To install the requirements, run:
```bash
pip install -r requirements.txt
```

### Download the dataset

Download the dataset files from [kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) and place the CSV uncompressed files inside the `data/raw/` directory. Create the `raw` directory if needed. Make sure to keep the original CSV file names.

## Experiments

### Definition

Define experiments in `experiments/config.yaml`. The recommended approach is to create a new experiment using as a reference one of the already defined experiments. For instance, a valid experiment definition for the sr_gnn with attentional aggregation model is: 

```yaml
experiment_name:
    model_name: "graph_with_embeddings_and_attentional_aggregation"
    model_params:
      lr: 0.001                         # learning rate
      epochs: 30                        # number of epochs
      optimizer: "Adam" 
      num_iterations: 1                 # number of iterations within the GGNN
      embedding_dim: 64                 # embedding dimension for product date embeddings.
      hidden_dim: 100                   # hidden dimension value for the GGNN
      batch_size: 50
      shuffle: False
      weight_decay: 0.0001
      dropout_rate: 0.3
    data_params:
      input_folder_path: "data/raw/"
      output_folder_artifacts: "experiments/experiment_name/" 
    preprocessing:
      start_month: '2019-10'            # Start month of data to consider for the dataset generation
      end_month: '2019-10'              # End month of data to consider for the dataset generation
      test_sessions_first_n: 10000      # Limit number of sessions. Remove to use the whole dataset.
      limit_to_view_event: True         # Limiting to "view" events or not.
      drop_listwise_nulls: True
      min_products_per_session: 3       # Smaller sessions will be discarded.
      normalization_method: 'zscore'
      train_split: 0.8
      val_split: 0.1
      test_split: 0.1
      split_method: 'temporal'          # Allowed values are 'temporal' and 'random'
    evaluation:
      top_k: [1, 5, 10, 20]             # top K values to take into account for metrics calculation
```

### Execution

Command to execute an experiment.
```bash
python run_experiment.py --config experiments/config.yaml --experiment your_experiment_name --task preprocess train
```

## Grid Search

### Definition
Define the grid search scenarios in `experiments/config-hyp.yaml`.  The definition is quite similar to the experiments definintion. 

```yaml
model_graph_with_embeddings_and_attentional_aggregation:
    model_name: "graph_with_embeddings_and_attentional_aggregation"
    model_params:
      epochs: 5
      optimizer: "Adam"
      num_iterations: 1
      embedding_dim: 64
      hidden_dim: 100
      batch_size: 50
      shuffle: False
    data_params:
      input_folder_path: "data/raw/"
      output_folder_artifacts: "experiments/graph_with_embeddings_and_attentional_aggregation/"
    preprocessing:
      start_month: '2019-10'
      end_month: '2019-10'
      test_sessions_first_n: 500000
      limit_to_view_event: True
      drop_listwise_nulls: True
      min_products_per_session: 3
      normalization_method: 'zscore'
      train_split: 0.8
      val_split: 0.1
      test_split: 0.1
      split_method: 'temporal'
    evaluation:
      top_k: [1, 5, 10, 20]
```

### Execution
```bash
python run_optim.py --model your_model_name --task preprocess train --force_rerun_train yes/no --resume yes/no 
```

- **--force_rerun_train**
  - "no": Will check the local directory for existing runs. If a complete training has been done before (i.e., the pertinent .pth file exists), it will not rerun. New or partially calculated scenarios will be executed.
  - "all": Will execute all scenarios, regardless of whether they have been calculated before.
  - "rerun_list": Will only overwrite those scenarios in the list.
    
- **--resume** : Resume training. If "yes":
  - Will skip fully executed scenarios
  - Will complete started scenarios from the last checkpoint

# License



# Contact

For any inquiries, you can contact us at:

- Joan González-Conde Cantero: [jgonzalezconde90@gmail.com](mailto:jgonzalezconde90@gmail.com)

- Judit Llorens Diaz: [juditllorens1998@gmail.com](mailto:juditllorens1998@gmail.com)

- Miguel Palos Pou: [miguelpalospou@gmail.com](mailto:miguelpalospou@gmail.com)

- Xavier Rodriguez Moreno: [xeeevi@gmail.com](mailto:xeeevi@gmail.com)

