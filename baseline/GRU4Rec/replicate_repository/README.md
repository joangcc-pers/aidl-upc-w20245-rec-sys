Steps to replicate the results from the repository [GRU4Rec_PyTorch_Official](https://github.com/hidasib/gru4rec_pytorch_official)

In order to replicate the results, we should download the October and November datasets from [kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store).

In order to preprocess the datasets, and create the .tsv files, we need to run the file [rees46_preproc.py](https://github.com/hidasib/gru4rec_third_party_comparison/blob/master/rees46_preproc.py)

The command to execute the training and the evaluation is: 
`python run.py /mnt/datadisk/dataset/oct_nov/rees46_processed_view_userbased_train_full.tsv -t /mnt/datadisk/dataset/oct_nov/rees46_processed_view_userbased_test.tsv -m 1 5 10 20 -pf ./paramfiles/rees46_xe_shared_best.py -s /mnt/datadisk/trained_models/run_py_manual.pt
Loaded parameters from file: /mnt/datadisk/GRU4Rec_PyTorch_Official/paramfiles/rees46_xe_shared_best.py`

This command uses the parametres defined in `rees46_xe_shared_best.py` which returns the best performance using our dataset: 
- MRR@20: 0.2008
- R@20: 0.5293

Using the command above, we are able to reproduce assimilable performance with our dataset, concisely:
- Recall@1: 0.111772 MRR@1: 0.111772
- Recall@5: 0.301062 MRR@5: 0.178018
- Recall@10: 0.413486 MRR@10: 0.192999
- __Recall@20: 0.529393 MRR@20: 0.201067__

Execution log: 

```
(base) xevi_r_m@deeplearning-3-vm:/mnt/datadisk/GRU4Rec_PyTorch_Official$ python run.py /mnt/datadisk/dataset/oct_nov/rees46_processed_view_userbased_train_full.tsv -t /mnt/datadisk/dataset/oct_nov/rees46_processed_view_userbased_test.tsv -m 1 5 10 20 -pf ./paramfiles/rees46_xe_shared_best.py -s /mnt/datadisk/trained_models/run_py_manual.pt
Loaded parameters from file: /mnt/datadisk/GRU4Rec_PyTorch_Official/paramfiles/rees46_xe_shared_best.py
Creating GRU4Rec model on device "cuda:0"
SET   loss                    TO   cross-entropy   (type: <class 'str'>)
SET   constrained_embedding   TO   True            (type: <class 'bool'>)
SET   embedding               TO   0               (type: <class 'int'>)
SET   elu_param               TO   0.0             (type: <class 'float'>)
SET   layers                  TO   [512]           (type: <class 'list'>)
SET   n_epochs                TO   10              (type: <class 'int'>)
SET   batch_size              TO   240             (type: <class 'int'>)
SET   dropout_p_embed         TO   0.45            (type: <class 'float'>)
SET   dropout_p_hidden        TO   0.0             (type: <class 'float'>)
SET   learning_rate           TO   0.065           (type: <class 'float'>)
SET   momentum                TO   0.0             (type: <class 'float'>)
SET   n_sample                TO   2048            (type: <class 'int'>)
SET   sample_alpha            TO   0.5             (type: <class 'float'>)
SET   bpreg                   TO   0.0             (type: <class 'float'>)
SET   logq                    TO   1.0             (type: <class 'float'>)
Loading training data...
Loading data from TAB separated file: /mnt/datadisk/dataset/oct_nov/rees46_processed_view_userbased_train_full.tsv
Started training
The dataframe is already sorted by SessionId, Time
Created sample store with 4882 batches of samples (type=GPU)
Epoch1 --> loss: 6.440980       (838.44s)       [285.23 mb/s | 68456 e/s]
Epoch2 --> loss: 6.142780       (836.66s)       [285.84 mb/s | 68601 e/s]
Epoch3 --> loss: 6.058142       (835.72s)       [286.16 mb/s | 68679 e/s]
Epoch4 --> loss: 6.008482       (834.91s)       [286.44 mb/s | 68745 e/s]
Epoch5 --> loss: 5.973498       (829.38s)       [288.35 mb/s | 69203 e/s]
Epoch6 --> loss: 5.947220       (828.33s)       [288.71 mb/s | 69291 e/s]
Epoch7 --> loss: 5.925884       (828.27s)       [288.73 mb/s | 69296 e/s]
Epoch8 --> loss: 5.908268       (827.47s)       [289.01 mb/s | 69363 e/s]
Epoch9 --> loss: 5.892942       (827.58s)       [288.98 mb/s | 69354 e/s]
Epoch10 --> loss: 5.880402      (826.38s)       [289.39 mb/s | 69454 e/s]
Total training time: 8361.81s
Saving trained model to: /mnt/datadisk/trained_models/run_py_manual.pt
Loading test data...
Loading data from TAB separated file: /mnt/datadisk/dataset/oct_nov/rees46_processed_view_userbased_test.tsv
Starting evaluation (cut-off=[1, 5, 10, 20], using standard mode for tiebreaking)
Using existing item ID map
The dataframe is already sorted by SessionId, Time
Evaluation took 62.93s
Recall@1: 0.111772 MRR@1: 0.111772
Recall@5: 0.301062 MRR@5: 0.178018
Recall@10: 0.413486 MRR@10: 0.192999
Recall@20: 0.529393 MRR@20: 0.201067
```