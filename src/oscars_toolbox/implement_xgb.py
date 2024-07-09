import time
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


def evaluate_xgb(xgb_model, X_val, y_val, return_extra_metrics=False):
    t0 = time.time()
    y_pred = xgb_model.predict(X_val)
    tf = time.time()
    conf_matrix = confusion_matrix(y_val, y_pred, normalize='true')

    if not return_extra_metrics:
        return conf_matrix
    
    else:
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_val, y_pred, average='weighted')

        time_per_sample = (tf - t0) / len(y_val)

        return conf_matrix, accuracy, precision, recall, f1, roc_auc, time_per_sample
    
import matplotlib.pyplot as plt

def plot_confusion_matrix(confusion_matrix_val, confusion_matrix_test, allowed_categories, accuracy_val, accuracy_test, save_path, show_percentages=False):
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    cb1 = ax[0].imshow(confusion_matrix_val, cmap='viridis')
    fig.colorbar(cb1, ax=ax[0])
    ax[0].set_xticks(range(len(allowed_categories)))
    ax[0].set_yticks(range(len(allowed_categories)))
    ax[0].set_xticklabels(allowed_categories, rotation=90)
    ax[0].set_yticklabels(allowed_categories)
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    ax[0].set_title(f'Confusion Matrix, Validation Accuracy: {accuracy_val*100:.1f}%')

    cb2 = ax[1].imshow(confusion_matrix_test, cmap='viridis')
    fig.colorbar(cb2, ax=ax[1])
    ax[1].set_xticks(range(len(allowed_categories)))
    ax[1].set_yticks(range(len(allowed_categories)))
    ax[1].set_xticklabels(allowed_categories, rotation=90)
    ax[1].set_yticklabels(allowed_categories)
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('Actual')
    ax[1].set_title(f'Confusion Matrix, Test Accuracy: {accuracy_test*100:.1f}%')

    # Add percentage annotations
    if show_percentages:
        for i in range(len(allowed_categories)):
            for j in range(len(allowed_categories)):
                # fraction in that cell divided by total for column
                ax[0].text(j, i, f'{confusion_matrix.iloc[i, j]*100:.1f}%', ha='center', va='center', color='black')
                ax[1].text(j, i, f'{confusion_matrix_test.iloc[i, j]*100:.1f}%', ha='center', va='center', color='black')       

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

