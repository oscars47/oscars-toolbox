import torch
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def train_only(model, device, train_loader, val_loader, num_epochs=5, learning_rate=1e-3, weight_decay=1e-4, loss_func=nn.CrossEntropyLoss()):
    model.to(device)
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Define learning rate scheduler
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    train_acc_ls = []
    val_acc_ls = []

    for epoch in range(num_epochs):
        # Train
        model.train()
        with tqdm(train_loader) as pbar:
            for i, (data, labels) in enumerate(pbar):
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_func(output, labels.to(device))
                loss.backward()
                optimizer.step()
                train_accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=train_accuracy.item(), lr=optimizer.param_groups[0]['lr'])

        # at the end of each epoch, save the accuracy
        train_acc_ls.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                val_loss += loss_func(output, labels.to(device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)

        val_acc_ls.append(val_accuracy)

        # Update learning rate
        # scheduler.step()
        optimizer.step()

        print(
            f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
        )

        # if torch tensor is on GPU, move it to CPU
        # first check if float
        try:
            if train_accuracy.is_cuda:
                train_accuracy = train_accuracy.cpu()
                # convert to float
                train_accuracy = train_accuracy.item()
        except:
            pass
        try:
            if val_accuracy.is_cuda:
                val_accuracy = val_accuracy.cpu()
                # convert to float
                val_accuracy = val_accuracy.item()
        except:
            pass

    return model, train_accuracy, val_accuracy, train_acc_ls, val_acc_ls

def train_model(model_func, device, train_loader, val_loader, input_size, output_size, neurons_ls,  num_epochs=5, learning_rate=1e-3, weight_decay=1e-4, use_cnn=False, loss_func=nn.CrossEntropyLoss(), img_channels=3):
    '''note: this function reuses much of the code from https://github.com/ZiyaoLi/fast-kan/blob/master/examples/train_mnist.py'''

    # Define model
    # concatenate the input size at beginning of neurons_ls
    if not(use_cnn):
        neurons_ls = [input_size] + neurons_ls + [output_size]
        model = model_func(neurons_ls)
    else:
        model = model_func(num_classes=output_size, num_channels=img_channels) # here neurons_ls is (out_channels, kernel_size, and stride)
    return train_only(model, device, train_loader, val_loader, num_epochs, learning_rate, weight_decay, loss_func)

def evaluate(model, test_loader, num_classes, device, return_extra_metrics=False):
    model.to(device)
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predictions = outputs.max(1)
            all_preds.append(predictions.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # normalize by row, true labels
    conf_matrix = confusion_matrix(all_targets, all_preds, labels=range(num_classes), normalize='true') 

    if not return_extra_metrics:
        return conf_matrix
    
    else:
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        f1 = f1_score(all_targets, all_preds, average='weighted')

        # ensure they are all floats
        accuracy = float(accuracy)
        precision = float(precision)
        recall = float(recall)
        f1 = float(f1)

        return conf_matrix, accuracy, precision, recall, f1

def count_parameters_torch(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    pass