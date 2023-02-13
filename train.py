import os
from utils import load_input_data, load_configs, init, maybe_make_dir, get_best_model_checkpoint, remove_checkpoints, log_wandb, print_and_log, plot_step
from evaluate import evaluate_model, compute_val_metrics
from models.models_utils import getModel, getOptimizer, getLossFn
from dataloader.dataloader import MCSZDataset
import torch
from utils import summary, summary_string
import time
import wandb
from shutil import copy


def train(configs, df, data, device):
    # Define folds
    if configs['experimentEnv']['cross_validation']['enabled']:
        folds = configs['experimentEnv']['cross_validation']['k']
    else:
        folds = 1
    # Training
    for model_name in configs['experimentEnv']['models']:
        for optimizer_name in configs['experimentEnv']['optimizers']:
            for lossFn_name in configs['experimentEnv']['losses']:
                for fold in range(folds):
                    # Initialize model and configs
                    print_and_log("[INFO] initializing the model...", configs)
                    print_and_log(f"\t Model: {model_name}", configs)
                    print_and_log(f"\t Optimizer: {optimizer_name}", configs)
                    print_and_log(f"\t Loss function: {lossFn_name}", configs)
                    if(configs['experimentEnv']['cross_validation']['enabled']):
                        print_and_log(f"\t Fold: {fold}", configs)

                    if(configs['experimentEnv']['cross_validation']['enabled'] and configs['experimentEnv']['use_validation_split']):
                        # Training
                        df_train = df[(df['split']=='training') & (df["filename"].notnull()) & (df['fold_cv']!=fold)].reset_index()
                        train_data = MCSZDataset(df_train, data, configs, do_transform=configs['experimentEnv']['data_augmentation'], one_hot_encoding=True)
                        # Validation
                        df_val = df[(df['split']=='training') & (df["filename"].notnull()) & (df['fold_cv']==fold)].reset_index()
                        val_data = MCSZDataset(df_val, data, configs, do_transform=False, one_hot_encoding=True)
                        # Loaders
                        trainDataLoader = torch.utils.data.DataLoader(train_data, batch_size=configs['experimentEnv']['batch_size'], shuffle=True, num_workers=0)
                        valDataLoader = torch.utils.data.DataLoader(val_data, batch_size=configs['experimentEnv']['batch_size'], shuffle=False, num_workers=0)
                        # calculate steps per epoch for training and validation set
                        trainSteps = len(trainDataLoader.dataset) // configs['experimentEnv']['batch_size']
                        valSteps = len(valDataLoader.dataset) // configs['experimentEnv']['batch_size']
                    elif(not(configs['experimentEnv']['use_validation_split'])):
                        fold='all'
                        valDataLoader = None
                    else:
                        fold = 'all'

                    # Model definition
                    model = getModel(model_name, configs, device)
                    # Summary of the model
                    try:
                        if(model_name not in ['DenseNet12']):
                            summary(model, (1, 256, 256))
                            ss = summary_string(model, (1, 256, 256))
                            configs['logger'].info(ss[0])
                        else:
                            print_and_log("Summary not displayed for DenseNet121, some errors may occur.", configs)
                    except:
                        print_and_log("Pytorch-Summary failed. Please check if anything is wrong. Please note that for DenseNet121, some errors may occur.", configs)

                    # Initialize optimizer and loss function
                    opt = getOptimizer(optimizer_name, model, configs)
                    lossFn = getLossFn(lossFn_name)

                    # initialize dictionary to store training history
                    if(configs['experimentEnv']['use_validation_split']):   
                        H = {
                            "train_loss": [],
                            "train_acc": [],
                            "val_loss": [],
                            "val_acc": [],
                            "val_f1": [],
                            "val_precision": [],
                            "val_recall": [],
                            "val_auc": [],
                            "val_sn": [],
                            "val_sp": []
                        }
                    else:
                        H = {
                            "train_loss": [],
                            "train_acc": []
                        }

                    # Run ID
                    configs['experimentDescription']['run_id'] = wandb.util.generate_id()

                    if(configs['wandb']['enable_tracking']):    
                        # WandB – Config is a variable that holds and saves hyperparameters and inputs
                        cfg = {
                            'batch_size': configs['experimentEnv']['batch_size'],
                            'test_batch_size': configs['experimentEnv']['test_batch_size'],
                            'epochs': configs['experimentEnv']['epochs'],
                            'lr': configs['experimentEnv']['optim_args'][optimizer_name]['learning_rate'],
                            'weight_decay': configs['experimentEnv']['optim_args'][optimizer_name]['weight_decay'],
                            'random_state': configs['random_state'],
                            'model': model_name,
                            'optimizer': optimizer_name,
                            'description': configs['experimentDescription']['experiment_description'],
                            'run_id': configs['experimentDescription']['run_id'],
                            'fold': fold if configs['experimentEnv']['cross_validation']['enabled'] else None,
                            'lossFn': lossFn_name
                        }

                        # WandB – Initialize a new run
                        wandb.init(project=configs['experimentDescription']['project_name'], group=configs['experimentDescription']['experiment_name'], job_type=model_name, id=configs['experimentDescription']['run_id'], save_code=True, tags=[model_name, optimizer_name, lossFn_name], config=cfg, reinit=True, resume='allow')
                        wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

                        # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
                        # Using log="all" log histograms of parameter values in addition to gradients
                        wandb.watch(model, log="all")

                    # measure how long training is going to take
                    print_and_log("[INFO] training the network...", configs)
                    startTime = time.time()

                    for epoch in range(configs['experimentEnv']['epochs']):
                        wandb_dict = {}  # wandb to log
                        # Training step
                        train_step(H, configs, wandb_dict, model, device, trainDataLoader, opt, lossFn, epoch, lossFn_name, trainSteps)
                        # Validation step
                        val_step(H, configs, wandb_dict, model, device, valDataLoader, lossFn, lossFn_name, valSteps)
                        # Plot learning curve (real time)
                        plot_step(H, epoch, configs, model_name, optimizer_name, lossFn_name)
                        # WandB 
                        if(configs['wandb']['enable_tracking']):
                            log_wandb(wandb_dict)
                        # Save model checkpoint
                        if(configs['experimentEnv']['save_model_on_epoch']):
                            DIR = os.path.join(configs['out_training_dir'], configs['experimentDescription']['experiment_name'],model_name,optimizer_name,lossFn_name,f"fold_{str(fold)}",f"epoch_{str(epoch).zfill(len(str(configs['experimentEnv']['epochs'])))}")
                            maybe_make_dir(DIR)
                            PATH = os.path.join(DIR,f"model.pth")
                            print_and_log(f"[INFO] Saving model checkpoint at: {PATH} \n", configs)
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': opt.state_dict(),
                            }, PATH)

                    # Finish measuring how long training took
                    endTime = time.time()
                    print_and_log(f"[INFO] total time taken to train the model: {(endTime - startTime):.2f}s \n", configs)

                    print_and_log('Finished Training \n', configs)

                    # Save model - last epoch
                    DIR = os.path.join(configs['out_training_dir'], configs['experimentDescription']['experiment_name'],model_name,optimizer_name,lossFn_name,f"fold_{str(fold)}")
                    maybe_make_dir(DIR)
                    PATH = os.path.join(DIR,'model_last.pth')
                    print_and_log(f"[INFO] Saving model... \n", configs)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                    }, PATH)
                    ## WandB
                    if(configs['wandb']['enable_tracking']):
                        wandb.save('model.pth')

                    # Select best model checkpoint
                    if(configs['experimentEnv']['save_model_on_epoch'] and configs['experimentEnv']['use_validation_split']):
                        print_and_log("[INFO] Selecting best epoch...\n", configs)
                        best_epoch, best_epoch_metrics, best_model_path = get_best_model_checkpoint(H,configs,model_name,optimizer_name,lossFn_name,fold,metric='val_auc',trim_epochs_ratio=configs['experimentEnv']['trim_epochs_ratio'])
                        print_and_log(f"Best epoch starting with 0: {best_epoch}", configs)
                        print_and_log(f"Best epoch starting with 1: {best_epoch+1}", configs)
                        best_model_path_dst = os.path.join(configs['out_training_dir'], configs['experimentDescription']['experiment_name'],model_name,optimizer_name,lossFn_name,f"fold_{str(fold)}","model_best.pth")
                        # Copy pth
                        copy(best_model_path,best_model_path_dst)
                        # Load best model
                        model = getModel(model_name, configs, device)
                        opt = getOptimizer(optimizer_name,model,configs)
                        checkpoint = torch.load(best_model_path)
                        epoch = checkpoint['epoch']
                        model.load_state_dict(checkpoint['model_state_dict'])
                        opt.load_state_dict(checkpoint['optimizer_state_dict'])

                    ## WandB
                    if(configs['wandb']['enable_tracking']):
                        wandb.save('model_best.pth')

                    # Test model - best epoch
                    if(configs['experimentEnv']['test_model']):
                        print_and_log("[INFO] testing network...\n", configs)
                        df_test = df[(df['split']=='test') & (df["filename"].notnull())].reset_index()
                        test_data = MCSZDataset(df_test, data, configs, do_transform=False, one_hot_encoding=True)
                        testDataLoader = torch.utils.data.DataLoader(test_data, batch_size=configs['experimentEnv']['test_batch_size'], shuffle=False, num_workers=0)
                        testSteps = len(testDataLoader.dataset) // configs['experimentEnv']['test_batch_size']
                        evaluate_model(configs, model, lossFn, device, testDataLoader, testSteps, classes=configs['experimentEnv']['classes'], lossFn_name=lossFn_name, apply_softmax=configs['experimentEnv']['apply_softmax'], threshold=configs['experimentEnv']['pred_thresh'])

                    # Finish WandB run
                    if(configs['wandb']['enable_tracking']):
                        wandb.finish()

                    # Post-training clean model checkpoints
                    if(configs['experimentEnv']['post_training_remove_checkpoints']):
                        remove_checkpoints(os.path.join(configs['out_training_dir'], configs['experimentDescription']['experiment_name'],model_name,optimizer_name,lossFn_name,f"fold_{str(fold)}"), configs)


def train_step(H, configs, wandb_dict, model, device, trainDataLoader, optimizer, lossFn, epoch, lossFn_name, trainSteps):
    '''
    Training step with validation.
    '''
    # set the model in training mode
    model.train()
    
    # initialize the total training and validation loss
    totalTrainLoss = 0
    
    # initialize the number of correct predictions in the training and validation step
    trainCorrect = 0
    
    # loop over the training set
    for (x, y) in trainDataLoader:
        # send the input to the device
        (x, y) = (x.to(device=device, dtype=torch.float), y.to(device=device, dtype=torch.long))
        
        # Adapt for CrossEntropyLoss
        if(lossFn_name=='CrossEntropy'):
            y = y[:,0]
        
        # Zeros the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        pred = model(x)
        loss = lossFn(pred, y)
        loss.backward()
        optimizer.step()
        
        # add the loss to the total training loss so far and calculate the number of correct predictions
        totalTrainLoss += loss
        if(lossFn_name=='CrossEntropy'):
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
        else:
            trainCorrect += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            
    # Calculate the average training loss
    avgTrainLoss = totalTrainLoss / trainSteps
    # Calculate the training accuracy
    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    
    # Update our training history
    train_loss = float(avgTrainLoss.cpu().detach().numpy())
    train_acc = trainCorrect
    H["train_loss"].append(train_loss)
    H["train_acc"].append(train_acc)
    
    # Update wandb dict
    wandb_dict.update({
        "train_loss": train_loss,
        "train_acc": train_acc
    })
    
    # Print the model training/validation info
    print_and_log(f"[INFO] EPOCH: {epoch + 1}/{configs['experimentEnv']['epochs']}", configs)
    print_and_log(f"Train loss: {avgTrainLoss:.6f}, Train accuracy: {trainCorrect:.4f}", configs)
        
    return H, wandb_dict

   
def val_step(H, configs, wandb_dict, model, device, valDataLoader, lossFn, lossFn_name, valSteps):
    # initialize the total training and validation loss
    totalValLoss = 0
    
    # initialize the number of correct predictions in the training and validation step
    valCorrect = 0
    
    # Switch off autograd for evaluation - Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward(). Reduces memory consumption.
    with torch.no_grad():
        # Set the model in evaluation mode
        model.eval()
        
        # Initialize variables
        preds, preds_logits, gt = [], [], []
        val_loss = 0
        
        # Loop over the validation set
        for (x, y) in valDataLoader:
            # Send the input to the device
            (x, y) = (x.to(device, dtype=torch.float), y.to(device, dtype=torch.long))
            
            # Adapt for CrossEntropyLoss
            if(lossFn_name=='CrossEntropy'):
                y = y[:,0]
            
            # Make the predictions, compute validation loss and add them to the lists
            pred = model(x)
            totalValLoss += lossFn(pred, y)
            
            pred_label = pred.argmax(axis=1).cpu().numpy()
            preds.extend(pred_label)
            preds_logits.extend(pred.cpu().numpy())
            gt.extend(y.cpu().numpy())
            
            # Calculate the number of correct predictions
            if(lossFn_name=='CrossEntropy'):
                valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            else:
                valCorrect += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    
    # Calculate the average validation loss
    avgValLoss = totalValLoss / valSteps
    
    # Calculate the training and validation accuracy
    valCorrect = valCorrect / len(valDataLoader.dataset)
    
    # Update validaton history
    val_loss = float(avgValLoss.cpu().detach().numpy())
    val_acc = valCorrect
    H["val_loss"].append(val_loss)
    H["val_acc"].append(val_acc)
    
    # Update wandb dict
    wandb_dict.update({
        "val_loss": val_loss,
        "val_acc": val_acc
    })
    
    # Calculate extra validation metrics
    val_metrics = compute_val_metrics(H, wandb_dict, gt, preds, preds_logits, lossFn_name)
    
    # Print the model training/validation info
    print_and_log(f"Val loss: {avgValLoss:.6f}, Val accuracy: {valCorrect:.4f}, Val F1: {val_metrics['val_f1']:.4f}, Val AUC: {val_metrics['val_auc']:.4f}\n", configs)
    
    return H, wandb_dict


if __name__ == "__main__":
    # 1. Load configs
    configs = load_configs("./config_train.yaml")
    # 2. Load input data
    df, data = load_input_data(configs)
    # 3. Define device and init
    configs, device = init(configs)
    # 4. Print and log configs
    print_and_log("### CONFIGURATIONS ###\n", configs)
    print_and_log(configs, configs)
    print_and_log("\n######################\n", configs)
    # 5. Train
    train(configs, df, data, device)