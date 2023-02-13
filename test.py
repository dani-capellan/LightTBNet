import os
from utils import load_input_data, load_configs, init, print_and_log
from evaluate import evaluate_model, compute_test_metrics
from models.models_utils import getModel, getOptimizer, getLossFn
from dataloader.dataloader import MCSZDataset
import torch
import numpy as np


def test(configs, df, data, device):
    # Define folds
    if configs['experimentEnv']['cross_validation']['enabled']:
        folds = configs['experimentEnv']['cross_validation']['k']
    else:
        folds = 1
    # Load testing dataset and DataLoader
    df_test = df[(df['split']=='test') & (df["filename"].notnull())].reset_index()
    test_data = MCSZDataset(df_test, data, configs, do_transform=False, one_hot_encoding=True)
    testDataLoader = torch.utils.data.DataLoader(test_data, batch_size=configs['experimentEnv']['test_batch_size'], shuffle=False, num_workers=0)
    testSteps = len(testDataLoader.dataset) // configs['experimentEnv']['test_batch_size']
    # Initialize test_metrics dict
    test_metrics = {model_name: {optimizer_name: {lossFn_name: {} for lossFn_name in configs['experimentEnv']['losses']} for optimizer_name in configs['experimentEnv']['optimizers']} for model_name in configs['experimentEnv']['models']}
    # Testing
    for model_name in configs['experimentEnv']['models']:
        for optimizer_name in configs['experimentEnv']['optimizers']:
            for lossFn_name in configs['experimentEnv']['losses']:
                # Models paths
                models_paths = []
                MODEL_DIR = os.path.join(configs['data_in']['models_ckpts_dir'],configs['experimentDescription']['experiment_name'],model_name,optimizer_name,lossFn_name)
                for root, dirs, files in os.walk(MODEL_DIR):
                    if(configs['experimentEnv']['ckpt_filename'] in files):
                        models_paths.append(os.path.join(root,configs['experimentEnv']['ckpt_filename']))
                # Evaluation
                preds_logits_all = []
                for fold in range(folds):
                    # Initialize model and configs
                    print_and_log("[INFO] initializing the model...", configs)
                    print_and_log(f"\t Model: {model_name}", configs)
                    print_and_log(f"\t Optimizer: {optimizer_name}", configs)
                    print_and_log(f"\t Loss function: {lossFn_name}", configs)
                    if(configs['experimentEnv']['cross_validation']['enabled']):
                        print_and_log(f"\t Fold: {fold}", configs)
                    model = getModel(model_name, configs, device)
                    opt = getOptimizer(optimizer_name,model,configs)
                    lossFn = getLossFn(lossFn_name)
                    # Load model
                    model_path = [p for p in models_paths if f"fold_{fold}" in p][0]
                    checkpoint = torch.load(model_path)
                    epoch = checkpoint['epoch']
                    model.load_state_dict(checkpoint['model_state_dict'])
                    opt.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Test process
                    print_and_log("[INFO] testing network...\n", configs)
                    gt, preds_logits, preds, test_loss = evaluate_model(
                        configs, 
                        model, 
                        lossFn, 
                        device, 
                        testDataLoader,
                        testSteps, 
                        classes=configs['experimentEnv']['classes'], 
                        lossFn_name=lossFn_name, return_preds = True, 
                        apply_softmax=configs['experimentEnv']['apply_softmax'], 
                        threshold=configs['experimentEnv']['pred_thresh']
                    )
                    # Process outputs
                    preds_logits_processed = np.vstack(preds_logits)
                    preds_logits_all.append(preds_logits_processed)

                # Adapt outputs (logits)
                preds_logits_all = np.stack(preds_logits_all)
                y_pred_logits = np.mean(preds_logits_all, axis=0)
                y_pred = y_pred_logits[:,1]>=configs['experimentEnv']['pred_thresh']
                
                # Metrics
                info = {
                    'model_name': model_name,
                    'optimizer_name': optimizer_name,
                    'lossFn_name': lossFn_name,
                    'OUT_DIR': os.path.join(configs['out_dir'], model_name, optimizer_name, lossFn_name)
                }
                test_metrics[model_name][optimizer_name][lossFn_name] = compute_test_metrics(gt, y_pred, y_pred_logits, info, configs)
                
    return test_metrics


if __name__ == "__main__":
    # 1. Load configs
    configs = load_configs("config_test.yaml")
    # 2. Load input data
    df, data = load_input_data(configs)
    # 3. Define device and init
    configs, device = init(configs, test=True)
    # 4. Print and log configs
    print_and_log("### CONFIGURATIONS ###\n", configs)
    print_and_log(configs, configs)
    print_and_log("\n######################\n", configs)
    # 5. Test
    test_metrics = test(configs, df, data, device)