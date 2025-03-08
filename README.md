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

# 7. Repository structure and MLOPS features

// TODO Xavi

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

## Requirements

WORK IN PROGRESS...

To install the dependencies, run:
```bash
pip install -r requirements.txt
```

## Execution

### Executing experiments (DEPRECATED)
```bash
python run_experiment.py --config experiments/config.yaml --experiment experiment_1 --task preprocess train
```

- Tasks must be in correct order: preprocess and train. If orders are ntot placed in the correct order, process will fail and raise an error.
- Tasks can be omitted if performed before.

### Executing Grid Search optimization (DEPRECATED)
```bash
python run_optim.py --model your_model_name --task preprocess train
```

- Tasks must be in correct order: preprocess and train. If orders are ntot placed in the correct order, process will fail and raise an error.
- Tasks can be omitted if performed before.

# License



# Contact

For any inquiries, you can contact us at:

- Joan González-Conde Cantero: [jgonzalezconde90@gmail.com](mailto:jgonzalezconde90@gmail.com)

- Judit Llorens Diaz: [juditllorens1998@gmail.com](mailto:juditllorens1998@gmail.com)

- Miguel Palos Pou: [miguelpalospou@gmail.com](mailto:miguelpalospou@gmail.com)

- Xavier Rodriguez Moreno: [xeeevi@gmail.com](mailto:xeeevi@gmail.com)

