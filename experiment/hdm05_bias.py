import sys
import time

sys.path.insert(0, '..')
from pathlib import Path
import os
import random

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils import data
from torch.optim.lr_scheduler import StepLR

import spd.nn as nn_spd
from spd.optimizers import MixOptimizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def hdm05(data_loader):
    # main parameters
    lr = 2e-1 # 8e-2
    n = 93  # dimension of the data
    C = 117  # number of classes
    threshold_reeig = 1e-6  # threshold for ReEig layer
    epochs = 200

    if not Path(data_path).is_dir():
        print("Error: dataset not found")
        print(
            "Please download and extract the file at the following url: http://www-connex.lip6.fr/~schwander/datasets/hdm05.tgz")
        sys.exit(2)

    # setup data and model
    class HDMNet(nn.Module):
        def __init__(self):
            super(__class__, self).__init__()
            dim = 93
            dim4 = 30
            classes = 117
            self.bimap1 = nn_spd.BiMap(1, 1, dim, dim4, 0)
            self.chode = nn_spd.CholeskyDe()
            self.chomu = nn_spd.CholeskyMu()
            self.batchnorm1 = nn_spd.CholeskyBatchNormSPD(dim)
            self.pt = nn_spd.CholeskyPt(dim)
            self.logeig = nn_spd.LogEig()
            self.linear = nn.Linear(dim4 ** 2, classes).double()
            self.re = nn_spd.ReEig()



        def forward(self, x):
            x_spd = self.re(x)
            x_l = self.chode(x_spd)
            x_l = self.batchnorm1(x_l)
            x_pt = self.pt(x_l)
            x_spd = self.chomu(x_pt)
            x_spd = self.bimap1(x_spd)
            x_vec = self.logeig(x_spd).view(x_spd.shape[0], -1)
            y = self.linear(x_vec)
            return y

    model = HDMNet()

    # setup loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opti = MixOptimizer(model.parameters(), lr=lr)
    scheduler = StepLR(opti.optim, step_size=50, gamma=1.0)
    learning_rates = []

    # initial validation accuracy
    log_file = open('log_hdm05.txt', 'a')
    loss_val, acc_val = [], []
    y_true, y_pred = [], []
    gen = data_loader._test_generator
    model.eval()
    for local_batch, local_labels in gen:
        out = model(local_batch)
        l = loss_fn(out, local_labels)
        predicted_labels = out.argmax(1)
        y_true.extend(list(local_labels.cpu().numpy()))
        y_pred.extend(list(predicted_labels.cpu().numpy()))
        acc, loss = (predicted_labels == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
        loss_val.append(loss)
        acc_val.append(acc)
    acc_val = np.asarray(acc_val).mean()
    loss_val = np.asarray(loss_val).mean()
    print('Initial validation accuracy: ' + str(100 * acc_val) + '%')

    # training loop
    for epoch in range(epochs):
        current_lr = opti.optim.param_groups[0]['lr']
        learning_rates.append(current_lr)

        scheduler.step()

        startTime = time.time()
        # train one epoch
        loss_train, acc_train = [], []
        model.train()
        for local_batch, local_labels in data_loader._train_generator:
            opti.zero_grad()
            out = model(local_batch)
            l = loss_fn(out, local_labels)
            acc, loss = (out.argmax(1) == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
            loss_train.append(loss)
            acc_train.append(acc)
            l.backward()
            opti.step()
        acc_train = np.asarray(acc_train).mean()
        loss_train = np.asarray(loss_train).mean()
        print('train loss: ' + str(loss_train) + '; train acc: ' + str(100 * acc_train) + '% at epoch ' + str(epoch + 1) + '/'
               + str(epochs))
        log_file.write('%f,%f,%s\n' % (loss_train, 100 * acc_train, str(epoch + 1)))
        endTime = time.time()
        aa = endTime - startTime
        print('training time:' + str(aa))

        # validation
        acc_val_list, loss_val = [], []
        y_true, y_pred = [], []
        model.eval()
        for local_batch, local_labels in data_loader._test_generator:
            out = model(local_batch)
            l = loss_fn(out, local_labels)
            predicted_labels = out.argmax(1)
            y_true.extend(list(local_labels.cpu().numpy()))
            y_pred.extend(list(predicted_labels.cpu().numpy()))
            acc, loss = (predicted_labels == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
            acc_val_list.append(acc)
            loss_val.append(loss)
        xx = None
        acc_val = np.asarray(acc_val_list).mean()
        loss_val = np.asarray(loss_val).mean()
        print('Val loss: ' + str(loss_val) + '; Val acc: ' + str(100 * acc_val) + '% at epoch ' + str(epoch + 1) + '/'
              + str(epochs))
        log_file.write('%f,%f,%s\n' % (loss_val, 100 * acc_val, str(epoch + 1)))
        log_file.flush()
    log_file.close()
    print('Final validation accuracy: ' + str(100 * acc_val) + '%')
    return 100 * acc_val


if __name__ == "__main__":

    data_path = 'data/hdm05/'
    pval = 0.5
    batch_size = 30  # batch size

    class DatasetHDM05(data.Dataset):
        def __init__(self, path, names):
            self._path = path
            self._names = names

        def __len__(self):
            return len(self._names)

        def __getitem__(self, item):
            x = np.load(self._path + self._names[item])[None, :, :].real
            x = th.from_numpy(x).double()
            y = int(self._names[item].split('.')[0].split('_')[-1])
            y = th.from_numpy(np.array(y)).long()
            return x, y

    class DataLoaderHDM05:
        def __init__(self, data_path, pval, batch_size):
            for filenames in os.walk(data_path):
                names = sorted(filenames[2])
            random.Random().shuffle(names)
            N_test = int(pval * len(names))
            train_set = DatasetHDM05(data_path, names[N_test:])
            test_set = DatasetHDM05(data_path, names[:N_test])
            self._train_generator = data.DataLoader(train_set, batch_size=batch_size, shuffle='True')
            self._test_generator = data.DataLoader(test_set, batch_size=batch_size, shuffle='False')

    log_file = open('log_hdm05.txt', 'a')
    hdm05(DataLoaderHDM05(data_path, pval, batch_size))
    log_file.close()
