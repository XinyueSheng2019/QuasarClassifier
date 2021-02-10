from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import sys
from train import *


def plot_confusion_matrix(cm, savename, title='Confusion Matrix', classes = int):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title,fontsize=16)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label', fontsize=16)
    plt.xlabel('Predict label',fontsize=16)
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename+'.png', format='png')



def plot_ROC(actual, predict, path):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predict)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure(figsize=(10, 10), dpi=100)
    plt.title('Receiver Operating Characteristic',fontsize=16)
    plt.plot(false_positive_rate, true_positive_rate, 'b',label='ROC curve (area = %0.2f)' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.savefig(path+'ROC.png', format = 'png')

    with open(path+'roc_record.txt','w+') as f:
        f.write('false_positive_rate: ')
        f.write(str(list(false_positive_rate)))
        f.write('\ntrue_positive_rate: ')
        f.write(str(list(true_positive_rate)))
        f.write('\nthresholds: ')
        f.write(str(list(thresholds)))
        f.write('\nroc_auc: ')
        f.write(str(roc_auc))
    f.close()
    return roc_auc


def plot_two_graph(value1, value2, savename, x_label, y_label, y1, y2):
    '''
    plot for epochs
    '''
    plt.figure(figsize=(12, 8), dpi=100)
    plt.clf()
    plt.plot([n for n in list(range(1,len(value1)+1))], value1, marker='.', label = y1)
    plt.plot([n for n in list(range(1,len(value2)+1))], value2, marker='.', label = y2)
    plt.legend(loc='upper left')
    plt.xlabel(x_label,fontsize=16)
    plt.ylabel(y_label,fontsize=16)
    if 'epoch' in x_label and len(value1)<=50:
        ax = plt.gca()
        x_major_locator=MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(1, len(value1))
    if 'loss' not in savename:
        plt.ylim(0,1)
    plt.savefig(savename, format = 'png')

def plot_one_graph(value, savename, x_label, y_label):
    '''
    plot for batches
    '''
    plt.figure(figsize=(12, 8), dpi=100)
    plt.clf()
    plt.plot([n for n in list(range(1,len(value)+1))], value, label = y_label)
    plt.legend(loc='upper left')
    plt.xlabel(x_label,fontsize=16)
    plt.ylabel(y_label,fontsize=16)
    plt.xlim(1, len(value))
    if 'loss' not in savename:
        plt.ylim(0,1)
    plt.savefig(savename, format = 'png')   

    