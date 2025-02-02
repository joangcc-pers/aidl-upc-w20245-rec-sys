import os
import torch
from experiment_setup import setups

dataset_path = "./dataset"
model_path = "./trained_models"

params = setups["rees46"]["params_xe"]

train_path = os.path.join(dataset_path,"rees46_processed_view_userbased_train_full.tsv")
test_path = os.path.join(dataset_path,"rees46_processed_view_userbased_test.tsv")

print(f"Train path: {train_path}")
print(f"Test path: {test_path}")

def create_gru4rec_pytorch_script(model_name, train_folder, train_data, test_data, model_path, loss, optim, final_act, layers, batch_size, dropout_p_embed, dropout_p_hidden, learning_rate, n_epochs, m, eval_hidden_reset, use_correct_loss, use_correct_mask_reset):
    checkpoint_dir = f"{model_path}/{model_name}"
    s_train_full = f"CUDA_VISIBLE_DEVICES=0 python ./GRU4REC-pytorch/main.py --data_folder {train_folder} --train_data {train_data} --valid_data {test_data} --checkpoint_dir {checkpoint_dir} --num_layers {1} --embedding_dim {layers} --hidden_size {layers} --loss_type {'BPR-max' if loss=='bpr-max' else 'CrossEntropy'} --final_act {final_act} --n_epochs {n_epochs} --batch_size {batch_size} --dropout_input {dropout_p_embed} --dropout_hidden {dropout_p_hidden} --lr {learning_rate} --momentum {0.0} --optimizer_type {'Adagrad' if optim=='adagrad' else ''}{' --eval_hidden_reset' if eval_hidden_reset else ''}{' --use_correct_loss' if use_correct_loss else ''}{' --use_correct_mask_reset' if use_correct_mask_reset else ''}"
    s_test_full = s_train_full + f" --is_eval --load_model {checkpoint_dir}/model_0000{n_epochs-1}.pt --m {m}"
    return s_train_full, s_test_full

loss = params["loss"]
optim = params["optim"]
const_emb = params["constrained_embedding"]
embed = params["embedding"]
final_act = params["final_act"]
layers = params["layers"]
batch_size = params["batch_size"]
dropout_p_embed = params["dropout_p_embed"]
dropout_p_hidden = params["dropout_p_hidden"]
learning_rate = params["learning_rate"]
momentum = params["momentum"]
sample_alpha = params["sample_alpha"]
bpreg = params["bpreg"]
logq = params["logq"]
hidden_act = params["hidden_act"]
n_epochs = 5
m = '1 5 10 20'

train_folder, train_data = '/'.join(train_path.split('/')[:-1]), train_path.split('/')[-1]
test_folder, test_data = '/'.join(test_path.split('/')[:-1]), test_path.split('/')[-1]

print(f"Train folder: {train_folder}")
print(f"Train data: {train_data}")
print(f"Test folder: {test_folder}")
print(f"Test data: {test_data}")

train_script_oob, test_script_oob = create_gru4rec_pytorch_script(model_name='gru4rec_pytorch_oob_bprmax', train_folder=train_folder, train_data=train_data, test_data=test_data, model_path=model_path, loss=loss, optim=optim, final_act=final_act, layers=layers, batch_size=batch_size, dropout_p_embed=0.0, dropout_p_hidden=0.0, learning_rate=learning_rate, n_epochs=n_epochs, m=m, eval_hidden_reset=False, use_correct_loss=False, use_correct_mask_reset=False)

print("Train script")
print(train_script_oob)

print("Test script")
print(test_script_oob)

print(f"Cuda: {torch.cuda.is_available()}")

print("train out of the box model...")
print(f"Train Script OOB command: {train_script_oob}")
print(" ")
print(f"Test Script OOB command: {test_script_oob}")
print(" ")

os.system(train_script_oob)
os.system(test_script_oob)

print("train and test inference fix model...")
train_script_inffix, test_script_inffix = create_gru4rec_pytorch_script(model_name='gru4rec_pytorch_inffix_bprmax', train_folder=train_folder, train_data=train_data, test_data=test_data, model_path=model_path, loss=loss, optim=optim, final_act=final_act, layers=layers, batch_size=batch_size, dropout_p_embed=0.0, dropout_p_hidden=0.0, learning_rate=learning_rate, n_epochs=n_epochs, m=m, eval_hidden_reset=True, use_correct_loss=False, use_correct_mask_reset=False)

print(f"Train Script Inference Fix Model command: {train_script_inffix}")
print(" ")
print(f"Test Script Inference Fix Model command: {test_script_inffix}")
print(" ")


print("train the out of the box eval fix model")
os.system(train_script_inffix)

print("test the out of the box eval fix model")
os.system(test_script_inffix)

print("train and test the major fix model")
train_script_majorfix, test_script_majorfix = create_gru4rec_pytorch_script(model_name='gru4rec_pytorch_majorfix_bprmax', train_folder=train_folder, train_data=train_data, test_data=test_data, model_path=model_path, loss=loss, optim=optim, final_act=final_act, layers=layers, batch_size=batch_size, dropout_p_embed=dropout_p_embed, dropout_p_hidden=dropout_p_hidden, learning_rate=learning_rate, n_epochs=n_epochs, m=m, eval_hidden_reset=True, use_correct_loss=True, use_correct_mask_reset=True)

print(f"Train Script Major Fix Model command: {train_script_majorfix}")
print(" ")
print(f"Test Script Major Fix Model command: {test_script_majorfix}")
print(" ")

print("train the major fix model")
os.system(train_script_majorfix)

print("test the major fix model")
os.system(test_script_majorfix)