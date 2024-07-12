import time
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import numpy as np


def evaluate_xgb(xgb_model, X_val, y_val, return_extra_metrics=True):
    t0 = time.perf_counter()
    y_pred = xgb_model.predict(X_val)
    tf = time.perf_counter()
    conf_matrix = confusion_matrix(y_val, y_pred, normalize='true')

    if not return_extra_metrics:
        return conf_matrix
    
    else:
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        y_pred_proba = xgb_model.predict_proba(X_val)

        # One-hot encode y_val if it's not already one-hot encoded
        if len(y_val.shape) == 1:
            y_val_one_hot = np.eye(np.max(y_val) + 1)[y_val]
        elif len(y_val.shape) == 2 and y_val.shape[1] == 1:
            y_val_one_hot = np.eye(np.max(y_val) + 1)[y_val.squeeze()]
        else:
            y_val_one_hot = y_val

        # Calculate the ROC AUC score
        roc_auc = roc_auc_score(y_val_one_hot, y_pred_proba, average='weighted', multi_class='ovr')

        time_per_sample = (tf - t0) / len(y_val)

        return conf_matrix, accuracy, precision, recall, f1, roc_auc, time_per_sample
    
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def plot_confusion_matrix(confusion_matrix_val, confusion_matrix_test, allowed_categories, accuracy_val, accuracy_test, save_path, show_percentages=False, val_Xp=None, test_Xp=None, y_val_pred=None, y_test_pred=None, y_val_true=None, y_test_true=None, features=None, show_plot=False, only_features=False):

    if len(allowed_categories) > 19 and len(allowed_categories) < 40:
            colors_tab20b = cm.get_cmap('tab20b', 20)
            colors_tab20c = cm.get_cmap('tab20c', 20)

            # Combine the colormaps to reach 39 colors (we will use the first 19 colors from tab20c to avoid duplication)
            new_colors = np.vstack((colors_tab20b(np.linspace(0, 1, 20)),
                                    colors_tab20c(np.linspace(0, 1, len(allowed_categories) - 20))))
    elif len(allowed_categories) <= 19:
        new_colors = cm.get_cmap('tab20b', 20)(np.linspace(0, 1, len(allowed_categories)))
    else:
        raise ValueError('Too many categories to plot, need to add more colors to the colormap. **will be fixed in future versions**')

    if val_Xp is None or test_Xp is None or y_val_pred is None or y_test_pred is None or features is None:
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

    
    elif not only_features:
        # Generate colors from the 'tab20b' and 'tab20c' colormap

        # Create a colormap from these colors
        custom_cmap = mcolors.ListedColormap(new_colors)

        # Create the combined figure
        fig = plt.figure(figsize=(20, 15))

          # Confusion matrix for validation set
        ax1 = fig.add_subplot(221)
        cb1 = ax1.imshow(confusion_matrix_val, cmap='viridis')
        fig.colorbar(cb1, ax=ax1)
        ax1.set_xticks(range(len(allowed_categories)))
        ax1.set_yticks(range(len(allowed_categories)))
        ax1.set_xticklabels(allowed_categories, rotation=90)
        ax1.set_yticklabels(allowed_categories)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_title(f'Confusion Matrix, Validation Accuracy: {accuracy_val*100:.1f}%')

        # Add percentage annotations to validation confusion matrix
        if show_percentages:
            for i in range(len(allowed_categories)):
                for j in range(len(allowed_categories)):
                    ax1.text(j, i, f'{confusion_matrix_val[i, j]*100:.1f}%', ha='center', va='center', color='black')

        # Confusion matrix for test set
        ax2 = fig.add_subplot(222)
        cb2 = ax2.imshow(confusion_matrix_test, cmap='viridis')
        fig.colorbar(cb2, ax=ax2)
        ax2.set_xticks(range(len(allowed_categories)))
        ax2.set_yticks(range(len(allowed_categories)))
        ax2.set_xticklabels(allowed_categories, rotation=90)
        ax2.set_yticklabels(allowed_categories)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.set_title(f'Confusion Matrix, Test Accuracy: {accuracy_test*100:.1f}%')

        # Add percentage annotations to test confusion matrix
        if show_percentages:
            for i in range(len(allowed_categories)):
                for j in range(len(allowed_categories)):
                    ax2.text(j, i, f'{confusion_matrix_test[i, j]*100:.1f}%', ha='center', va='center', color='black')

        # 3D scatter plot for validation set
        ax3 = fig.add_subplot(223, projection='3d')
        sc1 = ax3.scatter(val_Xp[:, 0], val_Xp[:, 1], val_Xp[:, 2], c=y_val_pred, cmap=custom_cmap, s=5)
        ax3.set_title('Validation Set')
        ax3.set_xlabel(features[0])
        ax3.set_ylabel(features[1])
        ax3.set_zlabel(features[2])
        cbar1 = fig.colorbar(sc1, ax=ax3)
        cbar1.set_label('Validation Prediction')

        # 3D scatter plot for test set
        ax4 = fig.add_subplot(224, projection='3d')
        sc2 = ax4.scatter(test_Xp[:, 0], test_Xp[:, 1], test_Xp[:, 2], c=y_test_pred, cmap=custom_cmap, s=5)
        ax4.set_title('Test Set')
        ax4.set_xlabel(features[0])
        ax4.set_ylabel(features[1])
        ax4.set_zlabel(features[2])
        cbar2 = fig.colorbar(sc2, ax=ax4)
        cbar2.set_label('Test Prediction')

        plt.tight_layout()
        plt.savefig(save_path)
        if show_plot:
            plt.show()
        plt.close()

    elif y_val_true is not None and y_test_true is not None:
        # just the 3d scatter plots
        import plotly.graph_objects as go

        # Create 3D scatter plot for validation set
        custom_cmap = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' for r, g, b, _ in new_colors]

        # Create 3D scatter plot for validation set
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=val_Xp[:, 0],
            y=val_Xp[:, 1],
            z=val_Xp[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=[custom_cmap[val] for val in y_val_true],
                opacity=0.8
            ),
            name='Validation Set, True Colors'
        ))

        # Create 3D scatter plot for test set
        fig.add_trace(go.Scatter3d(
            x=test_Xp[:, 0],
            y=test_Xp[:, 1],
            z=test_Xp[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=[custom_cmap[test] for test in y_test_true],
                opacity=0.8
            ),
            name='Test Set, True Colors'
        ))

        fig.update_layout(
            title='3D Scatter Plots',
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Feature 3'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        fig.show()




