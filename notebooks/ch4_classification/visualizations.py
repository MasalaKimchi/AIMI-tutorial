import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix with seaborn heatmap."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['COVID', 'NonCOVID'],
                yticklabels=['COVID', 'NonCOVID'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_sample_images(X_test, y_test, y_pred, n_samples=4):
    """Plot sample images with their true and predicted labels."""
    plt.figure(figsize=(10, 8))
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 2, i+1)
        img = X_test[idx].reshape(64, 64)
        plt.imshow(img, cmap='gray')
        plt.title(f'True: {"COVID" if y_test[idx]==0 else "NonCOVID"}\nPred: {"COVID" if y_pred[idx]==0 else "NonCOVID"}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_score):
    """Plot ROC curve with AUC score."""
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_metrics_table(y_true, y_pred):
    """Plot metrics table using seaborn heatmap."""
    from sklearn.metrics import classification_report
    import pandas as pd
    
    metrics = classification_report(y_true, y_pred, target_names=['COVID','NonCOVID'], output_dict=True)
    metrics_df = pd.DataFrame(metrics).transpose()
    metrics_df = metrics_df.drop('support', axis=1)
    
    plt.figure(figsize=(10, 4))
    sns.heatmap(metrics_df, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Detailed Classification Metrics')
    plt.tight_layout()
    plt.show() 