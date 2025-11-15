import json
import matplotlib.pyplot as plt

def plt_training_metrics(path):
    metrics = None
    with open(path, 'r') as f:
        metrics = json.load(f)
    fig, axis = plt.subplots(1,3,figsize=(20,5))
    for k in metrics.keys():
        if k != 'epoch':
            if k == 'regression loss':
                axis[1].plot(metrics[k], label= "reg_loss_train")
            elif k == 'regression_loss':
                axis[1].plot(metrics['epoch'], metrics[k], label="reg_loss_val")
            elif k == 'classification loss':
                axis[2].plot(metrics[k], label="cls_loss_train")
            elif k == 'classification_loss':
                axis[2].plot(metrics['epoch'],metrics[k], label="cls_loss_val")
            else:
                axis[0].plot(metrics['epoch'], metrics[k], label=k)
    fig.tight_layout()

    axis[0].legend()
    axis[1].legend()
    axis[2].legend()
    plt.show()