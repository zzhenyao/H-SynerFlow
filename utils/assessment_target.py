import sys
import argparse
import yaml
import json
import pickle
#=========================
import numpy as np
import copy
from scipy import stats
import math
import time
import pprint
#=========================
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
import torch.nn.functional as F

# torchlight
import torchlight
from torchlight.io import str2bool
from torchlight.io import DictAction
from torchlight.io import import_class
from utils.status import ProgressBar

from utils.processor import Processor
try:
    import wandb
except ImportError:
    wandb = None

import cProfile
import pstats
from net.net_utils import DEBUG_PROFILE

from ptflops import get_model_complexity_info


class ASESS_processor(Processor):
    def __init__(self, argv=None):
        super().__init__(argv)
    

    def train(self, progress_bar):
        self.model.train()
        loader = self.data_loader['train']
        loss_value = []
        epoch = self.meta_info['epoch']

        len_loader = len(loader)

        scaler = amp.GradScaler("cuda")
        for i, onedata in enumerate(loader):
            data, label = onedata
            data = data.to(self.dev, non_blocking=True)
            label = label.to(self.dev, non_blocking=True)
            self.train_step += 1

            if self.arg.use_amp:
                with amp.autocast("cuda"):
                    
                    output, target_loss = self.model(data, phase='train')
                    output = output.view(-1)
                    loss = self.loss1(output, label)
                    alpha = 0.1
                    with torch.no_grad():
                        epsilon = 1e-8
                        lambda_atts = (alpha * loss.detach()) / (target_loss.detach() + epsilon)
                        if lambda_atts > 1:
                            lambda_atts = torch.tensor(1.0)
                    loss += lambda_atts * target_loss
            else:
                output, target_loss = self.model(data, phase='train')
                output = output.view(-1)
                loss = self.loss1(output, label)
                alpha = 0.1
                with torch.no_grad():
                    epsilon = 1e-8
                    lambda_atts = (alpha * loss.detach()) / (target_loss.detach() + epsilon)
                    if lambda_atts > 1:
                        lambda_atts = torch.tensor(1.0)
                loss += lambda_atts * target_loss
            self.iter_info['loss'] = loss.data.item()
            loss = loss / self.arg.accumulation_steps

            scaler.scale(loss).backward()
             
            if (i + 1) % self.arg.accumulation_steps == 0 or (i + 1) == len_loader:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

            if not self.arg.nowandb:
                wandb.log({
                    'train_loss': loss * self.arg.accumulation_steps,
                    'train_step': self.train_step,
                    'train_labels': label.mean().item(),
                    'train_outputs': output.mean().item(),
                    'target_loss': target_loss.item()
                })

            # statistics
            loss_value.append(self.iter_info['loss'])
            progress_bar.prog(i, len(loader), epoch,loss.item() * self.arg.accumulation_steps)

        scheduler_lr = self.scheduler.optimizer.param_groups[0]['lr']
        self.arg.file_logging.info(f'Train '
                    f'epoch {epoch}:'
                    f'  train_loss:{np.mean(loss_value):5.2f}')
        if not self.arg.nowandb:
            d2 = {
                'train_lr': scheduler_lr,
                'train_step': self.train_step,
                'epoch': epoch,
                'train_mean_loss': np.mean(loss_value),
                'train_epoch': self.train_epoch
            }
            if not self.arg.nowandb:
                wandb.log(d2)
        if epoch < self.arg.warm_up_epoch + self.arg.opt_T_max:         
            if self.arg.opt_lr_scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(metric=self.epoch_info['mean_loss'])
            else:
                self.scheduler.step()
        self.epoch_info['mean_loss'] = np.mean(loss_value)

        #self.show_epoch_info()
        

    def test(self):
        status = self.model.training
        self.model.eval()
        self.test_step += 1
        loader = self.data_loader['test']
        loss_value = []
        totaloutput = []

        for i, onedata in enumerate(loader):
            data, label = onedata
            # get data
            data = data.to(self.dev, non_blocking=True)
            label = label.to(self.dev, non_blocking=True)

            # inference
            with torch.no_grad():
                output, target_loss = self.model(data, phase='test')
                
            output = output.view(-1)
            loss = self.loss1(output, label)
            loss += target_loss
            loss_value.append(loss.data.item())

            totaloutput.append(output.cpu().numpy().tolist())

        src, rl2, mse, l_score, o_score, save_best_flag = self.show_src(totaloutput)
        self.epoch_info['mean_loss'] = mse

        if not self.arg.nowandb:
            wandb.log({
                'test_step': self.test_step,
                'test_p': src,
                'test_mse': mse,
                'test_rl2': rl2,
                'test_labels': np.mean(l_score),
                'test_outputs': np.mean(o_score),
                'test_epoch': self.test_epoch,
                'test_mean_loss': np.mean(loss_value)
            })



        self.model.train(status)
        return save_best_flag


    def start(self):
        backbone_unfreeze_epoch = self.arg.backbone_unfreeze_epoch
        progress_bar = ProgressBar(verbose=True)
        self.arg.logging.info(f'|'.center(36, '-'))
        self.model.freeze_backbone()
        if self.arg.phase == 'train':
            
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch
                if epoch == backbone_unfreeze_epoch:
                    self.model.unfreeze_backbone()
                if epoch == 8000:
                    self.model.head.freeze_targets()
                # training
                self.train(progress_bar)
                self.train_epoch += 1
                # save model
                if (epoch + 1) > self.arg.warm_up_epoch:
                    if ((epoch + 1 - self.arg.warm_up_epoch) % self.arg.save_interval == 0):
                        self.save_checkpoint(epoch + 1)

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    save_best_flag = self.test()
                    if save_best_flag:
                        self.save_checkpoint(epoch + 1, best = True)
                    self.test_epoch += 1
                    if self.arg.check_best_epoch > 0 and epoch == self.arg.check_best_epoch:
                        if self.best_test_src < self.arg.check_best_src:
                            print(f"Best SRC {self.best_test_src} reached the target {self.arg.check_best_src}. Early stopping at epoch {epoch}.")
                            sys.exit(0)

            self.arg.logging.info("\tbestsrc: {}".format(self.best_test_src))
            self.arg.logging.info("\tbestmse: {}".format(self.best_test_mse))
            self.arg.logging.info("\tbestrl2: {:.2f}".format(self.best_test_rl2))
            self.arg.logging.info("\tbestsrc_epoch: {}".format(self.src_test_epoch))
            if not self.arg.nowandb:
                wandb.log({
                    'bestsrc': self.best_test_src,
                    'bestmse': self.best_test_mse,
                    'bestrl2': self.best_test_rl2,
                })
        else:
            self.test()

    def show_src(self, totalpred):
        output = sum(totalpred, [])
        test_label = []
        test_pred = []
        real_test_video_num = 0
        mse_sum = []
        med_sum = []
        save_best_flag = False
        if self.arg.data_clip:
            test_video_num = len(self.test_label_dict)
            max_num = max(self.test_label_dict.keys()) + 1
            totalscore = [[] for i in range(max_num)]
            for i in range(len(output)):
                totalscore[int(self.test_video_index[i])].append(output[i])
            for key in self.test_label_dict:
                if len(totalscore[int(key)]) == 0:
                    continue
                real_test_video_num += 1
                test_label.append(round(self.test_label_dict[key], 2))
                test_pred.append(round(sum(totalscore[int(key)]) / len(totalscore[int(key)]), 2))     


        else:
            test_video_num = len(output)
            for i in range(len(output)):
                test_label.append(round(self.test_label_dict[i], 2))
                test_pred.append(round(output[i], 2))
            real_test_video_num = test_video_num
        r, p = stats.spearmanr(test_label, test_pred)
        for k in range(real_test_video_num):
            if self.feeder_score_normalization:
                score_scale = self.feeder_max_score
                mse_sum.append(round((test_pred[k] * score_scale - test_label[k] * score_scale) * (test_pred[k] * score_scale - test_label[k] * score_scale), 4))
                med_sum.append(round(abs(test_pred[k] * score_scale - test_label[k] * score_scale), 8))
            else:
                mse_sum.append(round((test_pred[k] - test_label[k]) * (test_pred[k] - test_label[k]), 4))
                med_sum.append(round(abs(test_pred[k] - test_label[k]), 8))

        true_scores = np.array(test_label) / 25.0
        pred_scores = np.array(test_pred) / 25.0

        rl2 = 100 * np.power((pred_scores - true_scores) / 
                            (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
        rl2 = float(np.trunc(rl2 * 100) / 100)
        mse = round(sum(mse_sum) / real_test_video_num, 8)
        med = round(sum(med_sum) / real_test_video_num, 8) 
        if r > self.best_test_src:
            self.best_test_src = r
            self.src_test_epoch = self.meta_info['epoch']
            self.best_test_mse = mse
            self.best_test_med = med
            self.best_test_rl2 = rl2
            if self.arg.save_best > 0 and self.meta_info['epoch'] >= self.arg.save_best:
                save_best_flag = True
        if self.arg.phase == 'test':
            self.arg.logging.info(f'Test: '
                               f'rho:{r * 100:5.2f}%, rl2: {rl2:<8.2f},'
                               f' mse: {mse:<8.6f}, med: {med:<8.6f}')
        else:
            self.arg.logging.info(f'Evaluate: '
                                f'rho:{r * 100:5.2f}%, rl2: {rl2:<8.2f},'
                                f' *best:rho:{self.best_test_src * 100:5.2f}%， Rl2:{self.best_test_rl2:<8.2f}')

        return r, rl2,  mse, test_label, test_pred, save_best_flag

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')



        return parser

