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


