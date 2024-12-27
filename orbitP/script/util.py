import os, shutil
import numpy as np
# save train or validation loss
def log_loss(epoch:int, loss_val : float, path_to_save_loss : str, train : bool = True):
    if train:
        file_name = "train_loss.txt"
    else:
        file_name = "test_loss.txt"

    path_to_file = path_to_save_loss+file_name
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "a") as f:
        f.write("Epoch"+str(epoch+1)+": "+str(loss_val)+"\n")
        f.close()

# Exponential Moving Average, https://en.wikipedia.org/wiki/Moving_average
def EMA(values, alpha=0.1):
    ema_values = [values[0]]
    for idx, item in enumerate(values[1:]):
        ema_values.append(alpha*item + (1-alpha)*ema_values[idx])
    return ema_values

def Pml(src,pred):
    res = 0
    for i in range(len(pred)):
        res +=abs(src[i]-pred[i])
    res /= np.sum(np.abs(src))
    return res

# Remove all files from previous executions and re-run the model.
def clean_directory(path_to_save_dir):

    if os.path.exists(path_to_save_dir+'save_loss'):
        shutil.rmtree(path_to_save_dir+'save_loss')
    if os.path.exists(path_to_save_dir+'save_model'):
        shutil.rmtree(path_to_save_dir+'save_model')
    if os.path.exists(path_to_save_dir+'save_predictions'):
        shutil.rmtree(path_to_save_dir+'save_predictions')
    if os.path.exists(path_to_save_dir + 'load_model'):
        shutil.rmtree(path_to_save_dir + 'load_model')

    if not os.path.exists(path_to_save_dir):
        os.mkdir(path_to_save_dir)
    if not os.path.exists(path_to_save_dir+"save_loss"):
        os.mkdir(path_to_save_dir+"save_loss")
    if not os.path.exists(path_to_save_dir+"save_model"):
        os.mkdir(path_to_save_dir+"save_model")
    if not os.path.exists(path_to_save_dir+"save_predictions"):
        os.mkdir(path_to_save_dir+"save_predictions")
    if not os.path.exists(path_to_save_dir+"load_model"):
        os.mkdir(path_to_save_dir+"load_model")

