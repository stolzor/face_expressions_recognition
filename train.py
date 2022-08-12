import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from models.model import init_model
from matplotlib import pyplot as plt
from tqdm import trange
import tqdm
from preproc_data.load_data import CustomDataset, valid_data, val_labels, train_data, train_labels, test_data
import pickle

def train(model, data, loss, optim, device):
    res_acc = 0.0
    res_loss = 0.0
    n = 0

    for X, y in data:
        optim.zero_grad()

        X = X.to(device)
        y = y.to(device)
        output = model(X)

        curr_loss = loss(output, y)

        curr_loss.backward()
        optim.step()

        pred = torch.argmax(output, dim=-1)
        res_acc += torch.sum(pred == y.data).cpu()
        res_loss += curr_loss.item() * X.shape[0]
        n += X.size(0)

    return res_loss / n, res_acc / n


def eval(model, data, loss, device):
    res_acc = 0.0
    res_loss = 0.0
    n = 0

    with torch.no_grad():
        for X, y in data:
            X = X.to(device)
            y = y.to(device)
            output = model(X)

            curr_loss = loss(output, y)

            pred = torch.argmax(output, dim=-1)
            res_acc += torch.sum(pred == y.data).cpu()
            res_loss += curr_loss.item() * X.shape[0]
            n += X.size(0)

    return res_loss / n, res_acc / n


def fit(model, train_data, valid_data, loss, optim, scheduler, device, epoch):
    loss_history = {'train': [], 'valid': []}
    acc_history = {'train': [], 'valid': []}


    for i in trange(epoch, desc='EPOCH'):
        info_train = train(model, train_data, loss, optim, device)
        info_valid = eval(model, valid_data, loss, device)

        loss_history['train'].append(info_train[0])
        loss_history['valid'].append(info_valid[0])
        acc_history['train'].append(info_train[1])
        acc_history['valid'].append(info_valid[1])

        tqdm.tqdm.write(f'\ntrain_loss: {info_train[0]}, kfold_valid_loss: {info_valid[0]}\ttrain_acc: {info_train[1]}, '
                        f'kfold_valid_acc: {info_valid[1]}')

        scheduler.step(info_valid[1])

    return loss_history, acc_history


def main():
    random_seed = 69
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_fold = 5
    batch_size = 64
    epoch = 1
    best_acc = 0.0
    best_weight = 0

    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_seed)
    label_encoder = pickle.load(open('preproc_data/label_encoder.pkl', 'rb'))
    for i, (train_idx, valid_idx) in enumerate(skf.split(train_data, label_encoder.transform(train_labels))):
        train_loader = DataLoader(CustomDataset(train_data[train_idx], train_labels[train_idx]), shuffle=True, batch_size=batch_size)
        fold_valid_loader = DataLoader(CustomDataset(train_data[valid_idx], train_labels[valid_idx]), shuffle=True, batch_size=batch_size)
        valid_loader = DataLoader(CustomDataset(valid_data, val_labels), shuffle=True, batch_size=batch_size)

        clf = init_model('train')
        clf.to(device)

        optim = torch.optim.AdamW(clf.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.1, patience=3, verbose=False)
        loss = nn.CrossEntropyLoss()

        loss_history, acc_history = fit(clf, train_loader, fold_valid_loader, loss, optim, scheduler, device, epoch)
        loss_valid, acc_valid = eval(clf, valid_loader, loss, device)

        print(f'valid_loss: {loss_valid}, valid_acc: {acc_valid}')
        if acc_valid > best_acc:
            best_acc = acc_valid
            best_weight = clf.state_dict()

        fig, ax = plt.subplots(figsize=(15, 9), nrows=1, ncols=2)
        ax[0].plot(loss_history['train'], label="train_loss")
        ax[0].plot(loss_history['valid'], label="val_loss")
        ax[0].legend(loc='best')
        ax[0].xlabel("epochs")
        ax[0].ylabel("loss")
        ax[1].plot(acc_history['train'], label="train_acc")
        ax[1].plot(acc_history['valid'], label="valid_acc")
        ax[1].legend(loc='best')
        ax[1].xlabel("epochs")
        ax[1].ylabel("accuracy")
        plt.show()
        plt.close()

    torch.save(best_weight, "models/after_train.pth")
    loss = nn.CrossEntropyLoss()
    loss_valid, acc_valid = eval(clf, valid_loader, loss, device)
    print(f'\ntest_loss: {loss_valid}, test_acc: {acc_valid}')

if __name__ == '__main__':
    print('START TRAIN')
    main()