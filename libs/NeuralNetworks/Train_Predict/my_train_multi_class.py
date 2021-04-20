
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
# from libs.NeuralNetworks.Helper.my_is_inception import is_inception_model

# Why # activation_train, activation_valid ,
# chose CrossEntropyLoss combines LogSoftmax and NLLLoss in one single class.
#nn.NLLLoss(Negative Log Likelihood)
#because of numerical stability

def train(model, loader_train, criterion, optimizer, scheduler,
          epochs_num, activation_before_loss=None, activation_valid=None,
          log_interval_train=10, log_interval_valid=None,
          save_model_dir=None,
          loader_valid=None, loader_test=None, accumulate_grads_times=None):

    assert activation_before_loss in [None, 'softmax'], 'activation error!'
    # is_inception = is_inception_model(model)
    # if is_inception:
    #     model.AuxLogits.training = False
    #     num_filters = model.AuxLogits.fc.in_features
    #     num_class = model.last_linear.out_features
    #     model.AuxLogits.fc = nn.Linear(num_filters, num_class)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model.to(device)
    if torch.cuda.device_count() > 1: # and (not is_inception):
        model = nn.DataParallel(model)

    for epoch in range(epochs_num):
        print(f'Epoch {epoch}/{epochs_num - 1}')
        model.train()
        epoch_loss, epoch_sample_num, epoch_corrects = 0, 0, 0
        running_loss, running_sample_num, running_corrects = 0, 0, 0

        list_labels, list_preds = [], []

        for batch_idx, (inputs, labels) in enumerate(loader_train):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # if is_inception:
            #     # inception v3 use an auxiliary losses function
            #     if activation == None:
            #         outputs, aux_outputs = model(inputs)
            #     if activation == 'softmax':
            #         outputs, aux_outputs = torch.softmax(model(inputs))
            #     loss1 = criterion(outputs, labels)
            #     # loss2 = criterion(aux_outputs, labels)
            #     # losses = loss1 + 0.4 * loss2
            #     loss = loss1
            # else:
            outputs = model(inputs)
            if activation_before_loss == 'softmax':
                outputs = torch.softmax(outputs)
            loss = criterion(outputs, labels)

            if accumulate_grads_times is None:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                if batch_idx % accumulate_grads_times == 0:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            #region statistics
            if activation_valid == 'softmax':
                outputs = torch.softmax(outputs)

            _, preds = torch.max(outputs, 1)

            list_labels += labels.cpu().numpy().tolist()
            list_preds += preds.cpu().numpy().tolist()

            #show average losses instead of batch total losses  *inputs.size(0) total losses
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).cpu().numpy()
            running_sample_num += len(inputs)

            epoch_loss += loss.item()
            epoch_corrects += torch.sum(preds == labels.data).cpu().numpy()
            epoch_sample_num += len(inputs)

            if log_interval_train is not None:
                if batch_idx % log_interval_train == log_interval_train - 1:
                    print(f'[epoch:{epoch}, batch:{batch_idx}] losses:{running_loss / log_interval_train:8.2f}, acc:{running_corrects / running_sample_num:8.2f}')
                    running_loss, running_corrects, running_sample_num = 0, 0, 0

            #endregion

        print(f'epoch{epoch} losses:{epoch_loss / (batch_idx+1):8.2f}, acc:{epoch_corrects / epoch_sample_num:8.2f}')
        print('Confusion Matrix of training dataset:', confusion_matrix(list_labels, list_preds))

        scheduler.step()

        if loader_valid:
            print('compute validation dataset...')
            validate(model, activation_valid, loader_valid, log_interval_valid)
        if loader_test:
            print('compute test dataset...')
            validate(model, activation_valid, loader_test, log_interval_valid)

        if save_model_dir:
            save_model_file = os.path.join(save_model_dir, f'epoch{epoch}.pth')
            os.makedirs(os.path.dirname(save_model_file), exist_ok=True)
            print('save model:', save_model_file)
            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()

            torch.save(state_dict, save_model_file)



def validate(model, activation_valid, dataloader, log_interval=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    running_sample_num, running_corrects = 0, 0
    list_labels, list_preds = [], []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if activation_valid == 'softmax':
                outputs = torch.softmax(outputs)

            _, preds = torch.max(outputs, 1)

            list_labels += labels.cpu().numpy().tolist()
            list_preds += preds.cpu().numpy().tolist()

            running_corrects += torch.sum(preds == labels.data).cpu().numpy()
            running_sample_num += len(inputs)

            if log_interval is not None:
                if batch_idx % log_interval == log_interval - 1:
                    print(f'batch:{batch_idx + 1} acc:{ running_corrects / running_sample_num:8.2f}')
                    running_corrects, running_sample_num = 0, 0

    print('Confusion Matrix:', confusion_matrix(list_labels, list_preds))


