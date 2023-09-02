import os
import pandas as pd
import pickle
import yaml
import torch
from datetime import datetime
import numpy as np
import wandb
import re
from shutil import rmtree
import logging
import matplotlib.pyplot as plt


def maybe_make_dir(path):
    if(not(os.path.isdir(path))):
        os.makedirs(path)


def load_configs(PATH):
    with open(PATH) as file:
        configs = yaml.safe_load(file)

    return configs


def load_input_data(configs):
    '''
    Loads input data
    Inputs:
        configs: dict
    Outputs:
        df: Pandas DataFrame (dataset info)
        data: dict with images
        
    NOTE: We won't take into account patient_id, age_yo and sex columns from CSV file (DataFrame). You can omit that info or put to 0.
          If cross validation (CV) is enabled, please make sure that "fold_cv" column is properly included.
    '''
    
    # 1. Read DF. Note:
    df = pd.read_csv(configs['data_in']['csv'], index_col=0)

    # Read PKL
    with open(os.path.join(configs['data_in']['pkl']), 'rb') as handle:
        data = pickle.load(handle)

    return df, data


def init_log(configs, test=False):
    if(test):
        log_dir = os.path.join(os.path.join(configs['out_dir'],configs['experimentDescription']['experiment_name']))
    else:
        log_dir = os.path.join(configs['out_training_dir'],configs['experimentDescription']['experiment_name'])
    maybe_make_dir(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, "log.txt"), filemode='w', level=logging.DEBUG, format='%(message)s')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.info("###### LOGGER INITIALIZED ######")
    configs['logger'] = logger
    
    return configs
    

def init(configs, test=False):
    '''
    Function with some initialization steps
    '''
    if(test):
        if configs['experimentDescription']['experiment_name'] == "" or configs['data_in']['csv'] == "" or configs['data_in']['pkl'] == "" or configs['data_in']['models_ckpts_dir'] == "" or configs['out_dir'] == "":
            raise Exception("Please check you have entered all the information required in the corresponding config YAML file.")
        configs['out_dir'] = os.path.join(configs['out_dir'],configs['experimentDescription']['experiment_name'])
    else:
        if configs['out_training_dir'] == "":
            raise Exception("Please check you have entered all the information required in the corresponding config YAML file.")
        if(not(configs['experimentDescription']['experiment_name']) and not(test)):
            now = datetime.now() # current date and time
            d = now.strftime("%d%m%Y_%H%M%S")
            configs['experimentDescription']['experiment_name'] = f"exp_{d}"
    if(configs['log_output']):    
        configs = init_log(configs, test)
    print_cite(configs)
    torch.manual_seed(configs['random_state'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return configs, device


def get_best_epoch(history,metric,trim_epochs_ratio=1):
    '''
    trim_epochs_ratio: above which epoch we must search for this epoch. 
    Example: 0.75. That means that, for example, if the number of epochs used for training were 400, the best epoch that we are looking for will among the last 300 epochs (75% of 400).
    
    Default value: 1. That is, all epochs are considered
    
    best_epoch: Best epoch knowing that epochs begin by 0
    '''
    a = [[epoch,value] for epoch,value in zip(range(0,len(history[metric])),history[metric]) if epoch in range(int((1-trim_epochs_ratio)*len(history[metric])),len(history[metric]))]  # Considering trim_epochs_ratio
    # a = [[epoch,value] for epoch,value in zip(range(0,len(history[metric])),history[metric])]  # Without considering trim_epochs_ratio
    a = np.array(a) # To numpy
    if(metric=='val_loss'):
        a = a[a[:,1].argsort()]  # Order by second column in ascendent order (best at the top)
        a = a[a[:,1]==a[0,1]]  # Get all cases where the metric value is the highest (could be more than one)
        best_epoch_values = a[np.argmax(a[:,0])]  # Among the ones with the best metric, take the highest epoch
        best_epoch = int(best_epoch_values[0])  # Best epoch to int
    elif(metric in ['val_acc','val_auc']):
        a = a[a[:,1].argsort()][::-1]  # Order by second column in descendent order (best at the top)
        a = a[a[:,1]==a[0,1]]  # Get all cases where the metric value is the highest (could be more than one)
        best_epoch_values = a[np.argmax(a[:,0])]  # Among the ones with the best metric, take the highest epoch
        best_epoch = int(best_epoch_values[0])  # Best epoch to int
    return best_epoch

def get_best_model_checkpoint(history,configs,model_name,optimizer_name,lossFn_name,fold,metric='val_acc',trim_epochs_ratio=1):
    best_epoch = get_best_epoch(history,metric,trim_epochs_ratio)
    best_epoch_metrics = {
        "val_loss": history["val_loss"][best_epoch],
        "val_acc": history["val_acc"][best_epoch],
        "val_f1" : history["val_f1"][best_epoch],
        "val_precision" : history["val_precision"][best_epoch],
        "val_recall" : history["val_recall"][best_epoch],
        "val_auc" : history["val_auc"][best_epoch],
        "val_sn" : history["val_sn"][best_epoch],
        "val_sp" : history["val_sp"][best_epoch]
    }
    if(configs['wandb']['enable_tracking']):
        wandb.log({
            "best_epoch": best_epoch,  # Epoch starting with 0
            "best_val_loss": best_epoch_metrics["val_loss"],
            "best_val_acc": best_epoch_metrics["val_acc"],
            "best_val_f1" : best_epoch_metrics["val_f1"],
            "best_val_precision" : best_epoch_metrics["val_precision"],
            "best_val_recall" : best_epoch_metrics["val_recall"],
            "best_val_auc" : best_epoch_metrics["val_auc"],
            "best_val_sn" : best_epoch_metrics["val_sn"],
            "best_val_sp" : best_epoch_metrics["val_sp"]
        })
    # Get best model
    best_model_path = os.path.join(configs['out_training_dir'],configs['experimentDescription']['experiment_name'],model_name,optimizer_name,lossFn_name,f"fold_{str(fold)}",f"epoch_{str(best_epoch).zfill(len(str(configs['experimentEnv']['epochs'])))}","model.pth")

    return best_epoch, best_epoch_metrics, best_model_path


def log_wandb(wandb_dict):
    '''
    Log info in WandB
    '''
    wandb.log(wandb_dict)


def remove_checkpoints(PATH, configs):
    '''
    Removes subfolders inside PATH that match regex pattern "epoch_\d+"
    '''
    pattern = re.compile("epoch_\d+")
    for root, subdirs, files in os.walk(PATH, True):
        for subdir in subdirs:
            if(re.match(r"epoch_(\d+)", subdir)):
                print_and_log(f"Deleting... {os.path.join(root,subdir)}", configs)
                rmtree(os.path.join(root,subdir), ignore_errors=True)


def print_cite(configs):
    print_and_log("\nPlease cite the following paper when using LightTBNet:\n\nD. Capellán-Martín, J. J. Gómez-Valverde, D. Bermejo-Peláez and M. J. Ledesma-Carbayo. "
      "\"A Lightweight, Rapid and Efficient Deep Convolutional Network for Chest X-Ray Tuberculosis Detection,\" "
      "2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), Cartagena, Colombia, 2023, pp. 1-5, doi: 10.1109/ISBI53787.2023.10230500.\n", configs)
    print_and_log("If you have questions or suggestions, feel free to open an issue at https://github.com/dani-capellan/LightTBNet\n", configs)
    

def print_and_log(s: str, configs):
    print(s)
    if(configs['log_output']):
        configs['logger'].info(s)
        

# def plot_step(H, epoch, configs, model_name, optimizer_name, lossFn_name):
#     fig = plt.figure(figsize=(30, 24))
#     ax1 = fig.gca()
#     ax2 = ax1.twinx()
#     ax1.plot(list(range(epoch+1)), H['train_loss'], 'b-', label='Training loss')
#     ax1.plot(list(range(epoch+1)), H['val_loss'], 'r-', label='Validation loss')
#     ax2.plot(list(range(epoch+1)), H['val_auc'], 'g-', label='Validation AUC')
#     ax1.legend()
#     ax2.legend(loc='upper right', bbox_to_anchor=(1, -0.15, 0, 0), ncol=2)
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Loss")
#     ax2.set_ylabel("Evaluation Metric")
#     out_dir = os.path.join(configs['out_training_dir'], configs['experimentDescription']['experiment_name'], model_name, optimizer_name, lossFn_name)
#     maybe_make_dir(out_dir)
#     fig.savefig(os.path.join(out_dir, 'progress.svg'))
#     plt.close(fig)

def plot_step(H, epoch, configs, model_name, optimizer_name, lossFn_name):
    fig = plt.figure(figsize=(30, 24))
    ax1 = fig.gca()
    ax2 = ax1.twinx()
    ax1.plot(list(range(epoch+1)), H['train_loss'], 'b-', label='Training loss')
    ax1.plot(list(range(epoch+1)), H['val_loss'], 'r-', label='Validation loss')
    ax2.plot(list(range(epoch+1)), H['val_auc'], 'g-', label='Validation AUC')

    lns1 = ax1.get_lines()
    lns2 = ax2.get_lines()
    labs1 = [l.get_label() for l in lns1]
    labs2 = [l.get_label() for l in lns2]

    lns = lns1 + lns2
    labs = labs1 + labs2
    ax1.legend(lns, labs, loc='upper right')

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Evaluation Metric")
    out_dir = os.path.join(configs['out_training_dir'], configs['experimentDescription']['experiment_name'], model_name, optimizer_name, lossFn_name)
    maybe_make_dir(out_dir)
    fig.savefig(os.path.join(out_dir, 'progress.svg'))
    plt.close(fig)
