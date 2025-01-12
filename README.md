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

## Contents

The repository is organized as follows:
- **tentative_name**: Work in progress...

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

<!-- 1. Step 1:

2. Step 2: -->

python run_experiment.py --config experiments/config.yaml --experiment experiment_1 --task clean train evaluate visualize

- Tasks must be in correct order: clean, train, evaluate, and visualize. If orders are ntot placed in the correct order, process will fail and raise an error.
- Tasks can be omitted if performed before.

## License



## Contact

For any inquiries, you can contact us at:

- Joan González-Conde Cantero: [jgonzalezconde90@gmail.com](mailto:jgonzalezconde90@gmail.com)

- Judit Llorens Diaz: [juditllorens1998@gmail.com](mailto:juditllorens1998@gmail.com)

- Miguel Palos Pou: [miguelpalospou@gmail.com](mailto:miguelpalospou@gmail.com)

- Xavier Rodriguez Moreno: [xeeevi@gmail.com](mailto:xeeevi@gmail.com)

