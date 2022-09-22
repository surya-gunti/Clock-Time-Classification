from cProfile import label
from matplotlib import pyplot as plt
from . import config

def plot_accuracy(train_acc_li, val_acc_li):

    epochs = [int(i) for i in list(range(0, len(train_acc_li)))]
    plt.plot(epochs, train_acc_li, label='Train Acc')
    plt.plot(epochs, val_acc_li, label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(config.PLOT_PATH)