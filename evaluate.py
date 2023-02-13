import torch
import pprint
import numpy as np
# import wandb
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, recall_score, accuracy_score, classification_report, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt
import os
import pandas as pd
from utils import print_and_log, maybe_make_dir


def evaluate_model(configs, model, lossFn, device, testDataLoader, testSteps, classes, lossFn_name, return_preds=False, apply_softmax=True, threshold=0.5):
    '''
    Testing
    '''
    # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.eval()

    # Initialize variables
    preds, preds_logits, gt = [], [], []
    test_loss = 0
    example_images = []

    # Testing process
    with torch.no_grad():
        # loop over the test set
        for (x, y) in testDataLoader:
            # Send the input to the device
            x, y = (x.to(device, dtype=torch.float), y.to(device, dtype=torch.long))

            # Adapt for CrossEntropyLoss
            if(lossFn_name=='CrossEntropy'):
                y = y[:,0]

            # Make the predictions, compute loss and add them to the lists
            pred = model(x)
            test_loss += lossFn(pred, y)
            
            if(apply_softmax):
                pred = torch.nn.Softmax(1)(pred)
                
            if(threshold!=0.5):
                pred_label = (pred[:,1].cpu().numpy()>threshold).astype(int)
            else:
                pred_label = pred.argmax(axis=1).cpu().numpy()
            preds.extend(pred_label)
            preds_logits.extend(pred.cpu().numpy())
            gt.extend(y.cpu().numpy())

            # # (optional) WandB – Log images in your test dataset automatically, along with predicted and true labels by passing pytorch tensors with image data into wandb.Image
            # if(configs['wandb']['enable_tracking'] and configs['wandb']['save_test_predictions'] and not(return_preds)):
            #     example_images.append(wandb.Image(x[0], caption="Pred: {} Truth: {}".format(classes[pred_label[0]], classes[y[0]])))

    # Compute avg test loss
    test_loss /= testSteps
    
    if return_preds:
        return gt, preds_logits, preds, test_loss
    
    # Classification report and metrics
    print_and_log("[INFO] Displaying test metrics & results (single model)...", configs)
    if(lossFn_name=='CrossEntropy'):
        y_true = np.array(gt)
    else:
        y_true = np.array(gt).argmax(1)
    y_pred = np.array(preds)
    y_pred_logits = np.array(preds_logits)
    print_and_log(classification_report(y_true,y_pred), configs)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    test_metrics = {
        "test_acc": accuracy_score(y_true,y_pred),
        "test_f1": f1_score(y_true,y_pred),
        "test_precision": precision_score(y_true,y_pred),
        "test_recall": recall_score(y_true,y_pred),
        "test_auc": roc_auc_score(y_true,y_pred_logits[:,1]),
        "test_sn": recall_score(y_true,y_pred),
        "test_sp": tn / (tn+fp),
    }
    pprint.pprint(test_metrics)
    configs['logger'].info(f"TEST METRICS: {test_metrics}")
    print_and_log('\n', configs)  # Linebreak

    # # WandB – wandb.log(a_dict) logs the keys and values of the dictionary passed in and associates the values with a step.
    # # You can log anything by passing it to wandb.log, including histograms, custom matplotlib objects, images, video, text, tables, html, pointclouds and other 3D objects.
    # # Here we use it to log test accuracy, loss and some test images (along with their true and predicted labels).
    # if(configs['wandb']['enable_tracking']):
    #     if(configs['wandb']['save_test_predictions']):
    #         wandb.log({
    #             "test_examples": example_images
    #         })
    #     wandb.log(test_metrics)


def compute_val_metrics(H, wandb_dict, gt, preds, preds_logits, lossFn_name):

    # Other Metrics
    if(lossFn_name=='CrossEntropy'):
        y_true = np.array(gt)
    else:
        y_true = np.array(gt).argmax(1)
    y_pred = np.array(preds)
    y_pred_logits = np.array(preds_logits)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    val_metrics = {
        "val_f1": f1_score(y_true,y_pred),
        "val_precision": precision_score(y_true,y_pred),
        "val_recall": recall_score(y_true,y_pred),
        "val_auc": roc_auc_score(y_true,y_pred_logits[:,1]),
        "val_sn": recall_score(y_true,y_pred),
        "val_sp": tn / (tn+fp),
    }
    
    ###### CHECK
    assert accuracy_score(y_true,y_pred) == H['val_acc'][-1], 'Hey!'

    # Update validaton history with other metrics
    H["val_f1"].append(val_metrics["val_f1"])
    H["val_precision"].append(val_metrics["val_precision"])
    H["val_recall"].append(val_metrics["val_recall"])
    H["val_auc"].append(val_metrics["val_auc"])
    H["val_sn"].append(val_metrics["val_sn"])
    H["val_sp"].append(val_metrics["val_sp"])

    # Update wandb dict
    wandb_dict.update({
        "val_f1" : val_metrics["val_f1"],
        "val_precision" : val_metrics["val_precision"],
        "val_recall" : val_metrics["val_recall"],
        "val_auc" : val_metrics["val_auc"],
        "val_sn" : val_metrics["val_sn"],
        "val_sp" : val_metrics["val_sp"]
    })
        
    return val_metrics


def compute_test_metrics(gt, y_pred, y_pred_logits, info, configs):
    '''
    Compute test metrics and save results (confusion matrix, roc curve, etc.)
    Inputs:

    Outputs:
        test_metrics: [dict]
    '''

    # Metrics
    if(info['lossFn_name']=='CrossEntropy'):
        y_true = np.array(gt)
    else:
        y_true = np.array(gt).argmax(1)
        
    print_and_log(classification_report(y_true,y_pred), configs)
    if(configs['log_output']):
        configs['logger'].info(f" CLASSIFICATION REPORT: {classification_report(y_true,y_pred)}")

    # Maybe create out dir
    maybe_make_dir(info['OUT_DIR'])
    
    # Confusion Martix - Save figure in output folder
    fig = plt.figure()
    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, labels=[0,1], display_labels=[0,1], cmap=plt.cm.Blues)
    disp.plot()
    plt.title('Confusion matrix - test set')
    plt.savefig(os.path.join(info['OUT_DIR'],"test_confusion_matrix.svg"))
    plt.close(fig)

    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_logits[:,1])
    test_metrics = {
        "test_acc": accuracy_score(y_true,y_pred),
        "test_f1": f1_score(y_true,y_pred),
        "test_precision": precision_score(y_true,y_pred),
        "test_recall": recall_score(y_true,y_pred),
        "test_auc": roc_auc_score(y_true,y_pred_logits[:,1]),
        "test_sn": recall_score(y_true,y_pred),
        "test_sp": tn / (tn+fp),
        "roc": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds
        }
    }
    print_and_log("### TEST METRICS OBTAINED", configs)
    pprint.pprint(test_metrics)
    if(configs['log_output']):
        configs['logger'].info(f"TEST METRICS: {test_metrics}")
    print_and_log("#########################", configs)
    
    # ROC curve
    fig = plt.figure()
    lw = 2  # adjust line width
    plt.plot(test_metrics['roc']['fpr'], test_metrics['roc']['tpr'], color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % test_metrics['test_auc'])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1-SP)')
    plt.ylabel('True Positive Rate (SN)')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(info['OUT_DIR'],"test_roc_auc.svg"))
    plt.close(fig)

    # Save metrics to CSV
    test_metrics_df = pd.DataFrame(columns=['test_acc','test_f1','test_auc','test_precision','test_sn','test_sp'], dtype=float)
    # test_metrics_df = test_metrics_df.append({k:test_metrics[k] for k in test_metrics if k not in ['roc']},ignore_index=True)
    test_metrics_df = pd.concat([test_metrics_df,pd.DataFrame.from_dict({k:[test_metrics[k]] for k in test_metrics if k not in ['roc']})],ignore_index=True)
    test_metrics_df = test_metrics_df.round(4)
    test_metrics_df.to_csv(os.path.join(info['OUT_DIR'],f"metrics_{info['model_name']}_{info['optimizer_name']}_{info['lossFn_name']}.csv"))

    return test_metrics