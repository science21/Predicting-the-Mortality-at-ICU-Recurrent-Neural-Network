import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots_adjust
import numpy as np
from sklearn.metrics import confusion_matrix



def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(8.5, 10.5, forward=True)
    plt.rcParams.update({'font.size': 12})
    subplots_adjust(hspace=0.5)
    plt.subplot(2,1,1)
    plt.plot(np.arange(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation Loss')
    
    plt.title('Loss Curve')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(loc="best")

    plt.subplot(2,1,2)
    plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train Accuracy')
    plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    pass

# the plot_confusion_matrix code is modified based on the website:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(results, class_names):
    y_test =[a[0] for a in results]
    y_pred=[a[1] for a in results]
    cmat = confusion_matrix(y_test, y_pred)
    cm = cmat / cmat.astype(np.float).sum(axis=1)
    np.set_printoptions(precision=2)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, format = '%.2f')
   
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           
           xticklabels=class_names, yticklabels=class_names,
           title='Normalized Confusion Matrix',
           ylabel='True',
           xlabel='Predicted')

   
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
   
     
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")    
    fig.tight_layout()
      

    pass
