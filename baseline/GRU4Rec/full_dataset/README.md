# Steps to execute the baseline

This README defines the final working setup to execute and run the baseline data preprocessing and downloading.

3 different GCP components are needed:
- A memory-intensive instance for data preprocessing (referred as `preproc` in this documentation).
- A deeplearning-optimised instance for training and inference (referred as `train` in this documentation).
- An SSD persistent storage to store the data for both instances (one at a time).

All VMs and persistent storage are created in the location `europe-central-2b`. The most limiting factor here are GPUs, and after several retries, this is the location where there are GPUs available to provision the

## Data preprocessing
In order to process all the dataset, and since the current implementation of `rees46_preproc.py` loads all the data in memory, it requires a lot of memory. Since 128Gb RAM instances were not available, we found a good compromise with the following setup:
* Machine type: `n2-highmem-8`
* 64Gb RAM
* 64Gb Swap (on an attached SSD disk)

Especially when merging, filtering and deduplicating the data, all the RAM is used and also most of the swap, so a VM with 128Gb RAM or a more efficient implementation would significantly reduce the preprocessing time (~5hr). 

In order to detach the data from the compute, we created an SSD persistent storage of 300Gb that we attached and mounted to the `preproc` VM to preprocess the data and to the `train` vm for training and inference. Data preprocessing step on a non-SDD disk takes too long.

### Download the files
In order to download the files, execute the `download-dataset.sh` script on a folder named `dataset` in the SSD disk.

### Install dependencies
The dependencies needed for the data preprocessing are `pandas` and `numpy`. We recommend creating a miniconda environment with this dependencies. 

### Create folder structure
Training and inference expects the following folders at the working directory:
* `dataset`
* `trained-models`
* `trained-models/gru4rec_pytorch_oob_bprmax`

Make sure to make all folders and subfolders user-writable.

### Run the data preprocessing script
In order to run the data preprocessing, execute the following command: 
`$> python rees46_preproc.py -p dataset/`

Since the process will take a significant amount of time, it's recommended to:
* Protect against user hangup using `nohup`, `tmux`, `screen` or equivalent approaches. 
* Log the output in a file.

Once finished, the script will create the following files:
* `rees46_processed_view_userbased_full.tsv`
* `rees46_processed_view_userbased_test.tsv`
* `rees46_processed_view_userbased_train_full.tsv`
* `rees46_processed_view_userbased_train_tr.tsv`
* `rees46_processed_view_userbased_train_valid.tsv`
* `stats.tsv`

Commands like `lsof` or `top` are useful to monitor the process.

### Tidying up
* Unmount the SSD disk.
* Detach the SSD disk from the instance.
* Stop the instance.

## Model training
* Create a machine based on the (DeepLearning image)[https://console.cloud.google.com/marketplace/product/click-to-deploy-images/deeplearning] with 23Gb RAM.
* Attach the SSD disk containing the datasets and the output of the processed dataset.

### Run the train script 
In order to train the model, execute the following script:
`$> python rees46_train.py`

Since the process will take a significant amount of time, it's recommended to:
* Protect against user hangup using `nohup`, `tmux`, `screen` or equivalent approaches. 
* Log the output in a file.

Commands like `top` or `nvidia-smi` are useful to monitor the process.

### Tidying up
* Unmount the SSD disk.
* Detach the SSD disk from the instance.
* Stop the instance.