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

The implementaiton of these architectures will use pytorch, and visualize results through TensorFlow.

## Contents (TO BE UPDATED)

The repository is organized as follows:
- **data**: Contains the data used in the project.
    - **raw**: Stores the original, unprocessed dataset downloaded from Kaggle.
    - **processed**: Stores the data after preprocessing and feature engineering. This data is ready to be used for training and evaluation.
- **experiments**: Contains configuration files and results for different experiments. Each experiment has its own subfolder, where the results of the experiment will be stored.
    **config.yaml**: Contains the configuration for preprocessing and executing each experiment.
- **models**: Stores model_registry.py, which has the correspondence of each model class to each model called in config.yaml (within experiments). It also stores each individual model code, with the corresponding amethods to train, evaluate and construct the visualization artifacts.
- **notebooks**: where to store all notebooks.
- **results** Placeholder for resulting artifacts. We may delete it and use expriments folder instead.
- **scripts**: Code called by the run_experiments (main) function, that is reponsible for calling the corresponding methods of architectures' class.
- **test**: unit and integrity test we might need.
- **utils**: auxiliary utils we might need.
**run_experiment.py**: the "main" function we will call to train a model and store and visualize its results.


## Workflow

- Branch from develop when testing a new feature.
- After new feature has been added, merge to develop, NOT to main.
- Test everything in develop in the cloud, and when it is working, merge to main.
- Do **NOT** merge to main directly, so that we are sure that before marging to main, the test works in develop first.

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

## License



## Contact

For any inquiries, you can contact us at:

- Joan González-Conde Cantero: [jgonzalezconde90@gmail.com](mailto:jgonzalezconde90@gmail.com)

- Judit Llorens Diaz: [juditllorens1998@gmail.com](mailto:juditllorens1998@gmail.com)

- Miguel Palos Pou: [miguelpalospou@gmail.com](mailto:miguelpalospou@gmail.com)

- Xavier Rodriguez Moreno: [xeeevi@gmail.com](mailto:xeeevi@gmail.com)

