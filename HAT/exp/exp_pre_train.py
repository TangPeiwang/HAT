# from data_provider.data_factory import data_provider
from data_provider.all_data_loader import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Pre_Train(Exp_Basic):
    def __init__(self, args):
        super(Exp_Pre_Train, self).__init__(args)

    def _build_model(self):
        model = self.model_dict['HAT'].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            print("use multi gpu!!")
            model = nn.DataParallel(model)
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                print(f"GPUs: {model.device_ids}")
                for i in model.device_ids:
                    print(torch.cuda.get_device_name(i))
            else:
                print(f"Single GPU: {next(model.parameters()).device}")
        return model

    def _get_data(self, flag):
        data_loader = data_provider(self.args, flag)
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_loader = self._get_data(flag='train')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        print("ATTENTION!!!train_steps is", train_steps)
        model_optim = self._select_optimizer()
        # criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, mask_x) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            loss = self.model(batch_x, mask_x, batch_x, mask_x)[0]
                        else:
                            loss = self.model(batch_x, mask_x, batch_x, mask_x)
                        # loss = criterion(outputs, batch_x)
                        
                        loss=loss.mean()
                        # print(loss)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        loss = self.model(batch_x, mask_x, batch_x, mask_x)[0]
                    else:
                        loss = self.model(batch_x, mask_x, batch_x, mask_x)
                    # loss = criterion(outputs, batch_x)
                    
                    loss=loss.mean()
                    # print(loss)
                    train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            torch.save(self.model.state_dict(), path + '/' + str(epoch) + 'checkpoint.pth')

        return self.model
