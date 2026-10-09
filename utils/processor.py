#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
import random
import os
import datetime
import pprint

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import logging
# torchlight
import torchlight
from torchlight.io import str2bool
from torchlight.io import DictAction
from torchlight.io import import_class

from utils.io import IO
from utils.misc import GradualWarmupScheduler, copy_dir, read_pickle,init_seed


try:
    import wandb
except ImportError:
    wandb = None

class Processor(IO):
    """
        Base Processor
    """

    def __init__(self, argv=None):
        self.best_test_src = 0
        self.best_test_mse = 10000
        self.best_test_med = 10000
        self.best_test_rl2 = 10000
        self.src_test_epoch = 0
        self.mse_test_epoch = 0
        self.med_test_epoch = 0
        self.o_score=[]
        self.l_score = []
        self.train_step = 0
        self.test_step = 0
        self.train_epoch = 0
        self.test_epoch = 0
        self.batch_one_steps = 0
        self.feeder_score_normalization = False
        self.feeder_max_score = 1.0
        self.load_arg(argv)
        self.check_args()
        self.init_seed()
        self.test_label_dict = {}
        self.test_video_index = []
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()
        self.get_log()
        if self.arg.pretrain or self.arg.phase == 'test':
            self.load_checkpoint(self.arg.weight_path)

        self.test_epoch = 0 if self.arg.start_epoch == 0 else self.arg.start_epoch / self.arg.eval_interval 
        self.train_epoch = self.arg.start_epoch
        

        if not self.arg.nowandb:
            assert wandb is not None, "Wandb not installed, please install it or run without wandb"
            run = wandb.init(project=self.arg.wandb_project,
                       config=vars(self.arg),
                       name=self.arg.wandb_name,
                       mode="offline")
            self.arg.wandb_url = wandb.run.get_url()
            wandb.define_metric('train_*', step_metric='train_step')
            wandb.define_metric('epoch_train_*', step_metric='train_epoch')
            wandb.define_metric('epoch_test_*', step_metric='test_epoch')
            wandb.define_metric('test_*', step_metric='test_step')
            self.arg.logging.info('wandb: ' + run.path)
        self.back_proj()
        if self.arg.phase == 'train':
            self.arg.logging.info('Args: ' + pprint.pformat(vars(self.arg))) 

    def get_log(self):

        # logger: CRITICAL > ERROR > WARNING > INFO > DEBUG
        logger = logging.getLogger(self.arg.proj_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # stream handler
        log_sh = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(message)s', "%y-%m-%d %H:%M:%S")
        log_sh.setFormatter(formatter)

        logger.addHandler(log_sh)

        # file handler
        log_fh = logging.FileHandler(self.arg.log_file, mode='a')
        formatter = logging.Formatter('%(asctime)s | %(message)s', "%y-%m-%d %H:%M:%S")
        log_fh.setFormatter(formatter)

        logger.addHandler(log_fh)

        self.arg.logging = logger

        file_logger = logging.getLogger(self.arg.proj_name + "_file")
        file_logger.setLevel(logging.INFO)
        file_logger.addHandler(log_fh)
        self.arg.file_logging = file_logger

        # logging info
        self.arg.logging.info(f'New {self.arg.log_file}.')

    def init_seed(self):
        if self.arg.seed is not None:
            print(f'Initialize seed with {self.arg.seed}')
            init_seed(self.arg.seed)

    def init_environment(self):

        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.epoch_info['mean_loss'] = float('inf')
        self.meta_info = dict(epoch=0, iter=0)

    def load_optimizer(self):
        backbone_params = filter(lambda p: p.requires_grad, self.model.backbone.parameters())

        neck_head_params = filter(lambda p: p.requires_grad, self.model.neck.parameters())
        neck_head_params = list(neck_head_params) + list(self.model.head.parameters())  
        if self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                [
                    {'params': neck_head_params, 'lr': self.arg.base_lr},
                    {'params': backbone_params, 'lr': self.arg.backbone_lr},
                ],
                betas=(self.arg.opt_betas1, self.arg.opt_betas2),
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                [
                    {'params': neck_head_params, 'lr': self.arg.base_lr},
                    {'params': backbone_params, 'lr': self.arg.backbone_lr},
                ],
                betas=(self.arg.opt_betas1, 0.999),
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                [
                    {'params': neck_head_params, 'lr': self.arg.base_lr},
                    {'params': backbone_params, 'lr': self.arg.backbone_lr},
                ],
                momentum=self.arg.optim_mom,
                weight_decay=self.arg.weight_decay)
            '''
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=self.arg.optim_mom,
                weight_decay=self.arg.weight_decay)
            '''
        else:
            raise ValueError()
        scheduler = None
        if self.arg.opt_lr_scheduler == 'StepLR':
            scheduler =optim.lr_scheduler.StepLR(self.optimizer,
                                                       step_size=self.arg.opt_step_size,
                                                       gamma=self.arg.opt_gamma)
        elif self.arg.opt_lr_scheduler == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.arg.opt_T_max,
                    eta_min=self.arg.opt_eta_min
            )
        elif self.arg.opt_lr_scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.arg.opt_T_max,
                T_mult=self.arg.opt_T_mult,
                eta_min=self.arg.opt_eta_min,
                last_epoch=-1
            )
        elif self.arg.opt_lr_scheduler == 'MultiStepLR':
            scheduler =  optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.arg.opt_step,gamma=self.arg.opt_gamma)
        elif self.arg.opt_lr_scheduler == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=self.arg.opt_factor,
                    patience=self.arg.opt_patience, threshold=self.arg.opt_threshold,
                    threshold_mode='rel', cooldown=0, min_lr=1e-06, eps=1e-08, verbose=False)
        elif self.arg.opt_lr_scheduler == 'OneCycleLR':
            scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.arg.base_lr,
                    total_steps=self.arg.opt_T_max,
                    pct_start=0.000001,
                    three_phase=False,
                    anneal_strategy='cos',
                    div_factor=1.0,
                    final_div_factor=1e2,
                    cycle_momentum=True,
                    base_momentum=self.arg.opt_betas1,
                    max_momentum=self.arg.opt_betas2,
            )
        else:
            raise ValueError()          
        
        if self.arg.pretrain is False and self.arg.warm_up_epoch > 0:
            self.scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                after_scheduler=scheduler)
        else:
            self.scheduler = scheduler
    

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers= self.arg.num_worker,
                drop_last=True,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=48)
        if self.arg.test_feeder_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers= self.arg.num_worker,
                pin_memory=True 
            )
        
        self.batch_one_steps = len(self.data_loader[self.arg.phase])
        test_labels = read_pickle(self.arg.test_feeder_args['label_path'])
        self.test_label_dict, self.test_video_index = self.get_videoID_to_label_unique(test_labels)


    def get_videoID_to_label_unique(self, test_labels):
        self.feeder_score_normalization = self.arg.test_feeder_args.get('score_normalization', False)
        self.feeder_max_score = float(self.arg.test_feeder_args.get('max_score', 1))
        score_normalization = self.feeder_score_normalization
        max_score = self.feeder_max_score
        if self.arg.data_clip:
            modified_first_labels = [int(label.split('_')[0]) for label in test_labels[0]]
            test_video_index = modified_first_labels
            new_test_labels_tuple = (modified_first_labels, test_labels[1])
            seen = set()
            unique_first_labels = []
            unique_second_labels = []

            for first, second in zip(new_test_labels_tuple[0], new_test_labels_tuple[1]):
                if first not in seen:
                    seen.add(first)
                    unique_first_labels.append(first)
                    

                    if score_normalization:
                        second = second / max_score
                    second = round(second, 2)
                    unique_second_labels.append(second)

            deduplicated_dict = {key: value for key, value in zip(unique_first_labels, unique_second_labels)}
        else:
            if score_normalization:
                deduplicated_dict = {idx: round(label/max_score, 2) for idx, label in enumerate(test_labels[1])}
            else:
                deduplicated_dict = {idx: round(label, 2)  for idx, label in enumerate(test_labels[1])}
            test_video_index = list(range(len(test_labels[1])))

        return deduplicated_dict, test_video_index


        

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.arg.logging.info('\t{}: {}'.format(k, v))
       


    def show_iter_info(self):
        '''
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.arg.logging.info(info)'
        '''
        return None


    def train(self):
        return None

    def test(self):
        return None

    def start(self):
        return None

    def check_args(self):
        self.arg.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        parser = self.get_parser()
        replaceable_keys = [
            action.dest for action in parser._actions if '.' in action.dest
        ]

        for key in replaceable_keys:

            value = getattr(self.arg, key, None)


            if value is not None:
                if isinstance(value, str):
                    if value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            pass

                keys = key.split('.')

                def get_child(obj, k):
                    if isinstance(obj, dict) and k in obj:
                        return obj[k]
                    if hasattr(obj, k):
                        return getattr(obj, k)
                    return None

                def deep_search(obj, k):
                    if isinstance(obj, dict):
                        for kk, vv in obj.items():
                            if kk == k:
                                return vv
                            res = deep_search(vv, k)
                            if res is not None:
                                return res

                    attrs = []
                    try:
                        attrs = dir(obj)
                    except:
                        pass
                    
                    for attr in attrs:
                        if attr.startswith("_"):
                            continue
                        try:
                            v = getattr(obj, attr)
                        except:
                            continue
                        if attr == k:
                            return v
                        res = deep_search(v, k)
                        if res is not None:
                            return res

                    return None


                # ============================
                # ============================
                d = self.arg
                for k in keys[:-1]:

                    next_d = get_child(d, k)

                    if next_d is None:
                        next_d = deep_search(d, k)

                    if next_d is None:
                        d = None
                        break

                    d = next_d


                # ============================
                # ============================
                if d is not None:
                    last_key = keys[-1]
                    if isinstance(d, dict):
                        d[last_key] = value
                    elif hasattr(d, last_key):
                        setattr(d, last_key, value)
                    else:
                        pass
                else:
                    pass



        if self.arg.debug is True:
            self.arg.nowandb = 1
            self.arg.accmulation_steps = 1
        if self.arg.phase == 'test':
            self.arg.nowandb = 1

        
        # default exp name
        self.arg.proj_name = os.path.abspath(__file__).split('/')[-3]
        if not hasattr(self.arg, f'user_non_default'):
            self.arg.user = os.getenv('USER', 'default_user')
        if self.arg.exp_name == 'default':
            self.arg.exp_name = self.arg.model

        # add user and version to the exp name
        self.arg.exp_dir = os.path.join(
            './outputs', self.arg.user ,
            self.arg.dataset, self.arg.exp_name)
        os.makedirs(self.arg.exp_dir, exist_ok=True)

        base_exp_path = self.arg.exp_dir

        existing_dirs = [d for d in os.listdir(base_exp_path) if d.startswith("exp") and d[3:].isdigit()]

        if existing_dirs:
            max_index = max(int(d[3:]) for d in existing_dirs)
            new_dir_name = f"exp{max_index + 1}"
        else:
            new_dir_name = "exp1"
        self.arg.exp_dir = os.path.join(self.arg.exp_dir, new_dir_name)
        os.makedirs(self.arg.exp_dir, exist_ok=True)

        # logger file and dir
        self.arg.log_dir = os.path.join(self.arg.exp_dir, 'logs')
        os.makedirs(self.arg.log_dir, exist_ok=True)

        self.arg.log_file = os.path.join(self.arg.log_dir,
                                          f'{self.arg.phase}-{self.arg.timestamp}.log')
        self.arg.weight_save_path = os.path.join(self.arg.exp_dir, f'weights')
        os.makedirs(self.arg.weight_save_path, exist_ok=True)


        if self.arg.dataset_repath:
            self.arg.test_feeder_args['label_path'] = self.arg.test_feeder_args['label_path'].replace('DATASET', self.arg.dataset)
            self.arg.test_feeder_args['data_path'] = self.arg.test_feeder_args['data_path'].replace('DATASET', self.arg.dataset, 1)
            self.arg.train_feeder_args['label_path'] = self.arg.train_feeder_args['label_path'].replace('DATASET', self.arg.dataset)
            self.arg.train_feeder_args['data_path'] = self.arg.train_feeder_args['data_path'].replace('DATASET', self.arg.dataset, 1)
 
            if self.arg.inputfeature is True:
                string_clips = self.arg.model_args['clip_num']
                string_dataname = f'{self.arg.dataset}_{string_clips}_f'
                self.arg.test_feeder_args['data_path'] = self.arg.test_feeder_args['data_path'].replace('DATASET', string_dataname)
                self.arg.train_feeder_args['data_path'] = self.arg.train_feeder_args['data_path'].replace('DATASET', string_dataname)
            else:
                self.arg.test_feeder_args['data_path'] = self.arg.test_feeder_args['data_path'].replace('DATASET', self.arg.dataset)
                self.arg.train_feeder_args['data_path'] = self.arg.train_feeder_args['data_path'].replace('DATASET', self.arg.dataset)
     
        if self.arg.dataset_M_repath:
            if self.arg.num_people == 1:
                string_num = ''
            else:
                string_num = f'_{self.arg.num_people}'
            self.arg.test_feeder_args['label_path'] = self.arg.test_feeder_args['label_path'].replace('PEOPLE', string_num)
            self.arg.test_feeder_args['data_path'] = self.arg.test_feeder_args['data_path'].replace('PEOPLE', string_num)
            self.arg.train_feeder_args['label_path'] = self.arg.train_feeder_args['label_path'].replace('PEOPLE', string_num)
            self.arg.train_feeder_args['data_path'] = self.arg.train_feeder_args['data_path'].replace('PEOPLE', string_num)  
            


    def back_proj(self):
        if self.arg.debug:
            return
        # save the configuration
        config = os.path.join(self.arg.exp_dir, f'configs/{self.arg.phase}.yaml')
        os.makedirs(os.path.dirname(config), exist_ok=True)
        
        # self.reuse_config()

        with open(config, 'w') as f:
            yaml.dump(vars(self.arg), f)
            self.arg.logging.info(f'Save {config}.')

        # save main file and dirs
        for dir in ['main.py', 'utils', 'net', 'feeder']:
            new_dir = os.path.join(self.arg.exp_dir, dir)
            if os.path.exists(dir):
                copy_dir(dir, new_dir)
        config_src = self.arg.config
        config_dst = os.path.join(self.arg.exp_dir, f'{self.arg.config}')
        os.makedirs(os.path.dirname(config_dst), exist_ok=True)
        copy_dir(config_src, config_dst)
        self.arg.logging.info(f'Copied {self.arg.config} file to: {config_dst}')
                    
    def save_checkpoint(self, epoch, best=False):
        r"""
            Save the model checkpoint.
            :param epoch: Current epoch number.
            :param best: Boolean flag indicating if this is the best model so far.
        """
        if best:
            checkpoint_file = os.path.join(self.arg.weight_save_path, f'best_model.pt')
        else:
            checkpoint_file = os.path.join(self.arg.weight_save_path, f'epoch{epoch}_model.pt')
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(), 
            'epoch': epoch,
            'src_test_epoch': self.src_test_epoch,
            'bestsrc': self.best_test_src,
            'train_step': self.train_step,
            'test_step': self.test_step,
            'train_epoch': self.train_epoch,
            'test_epoch': self.test_epoch
        }, checkpoint_file)
        self.arg.logging.info(f'Save checkpoint to {checkpoint_file}')

    def load_checkpoint(self, weight_path=None):
        if weight_path is None:
            return

        if os.path.exists(weight_path):
            state = torch.load(weight_path)
            self.model.load_state_dict(state['model'], strict=False)
            self.optimizer.load_state_dict(state['optimizer'])
            self.arg.start_epoch = state['epoch']
            self.src_test_epoch = state['src_test_epoch']
            self.best_test_src = state['bestsrc']
            self.train_step = state['train_step']
            self.test_step = state['test_step']
            self.train_epoch = state['train_epoch']
            self.test_epoch = state['test_epoch']

            self.arg.logging.info(f'Load pretrained model state from {weight_path}')

    def load_model(self):
        model_args = {
            'model_args': self.arg.model_args,
            'featureadapter_args': self.arg.featureadapter_args,
            'backbone_args': self.arg.backbone_args,
            'neck_args': self.arg.neck_args,
            'head_args': self.arg.head_args
        }
        self.model = self.io.load_model(self.arg.model,
                                        **model_args)
        #self.model.apply(weights_init)
        self.loss1 = nn.MSELoss()

    @staticmethod
    def get_parser(add_help=False):

        #region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser( add_help=add_help, description='Base Processor')

        parser.add_argument('--exp_dir', default='./outputs', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--exp_name', type=str, default='default', help='Work dir.')
        parser.add_argument('--weight_path', type=str, default=None, help='pretrain weight')
        parser.add_argument('--pretrain', type=str2bool, default=False, help='pretrain')
        parser.add_argument('--use_amp', type=str2bool, default=True, help='use amp for training')
        parser.add_argument('--save_result', type=str2bool, default=False, help='if ture, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
        parser.add_argument('--seed', type=int, default=1024, help='The random seed.')
        parser.add_argument('--nowandb', default=0, choices=[0, 1], type=int, help='Inhibit wandb logging')
        parser.add_argument('--wandb_name', type=str, default='', help='Wandb experiment name')
        parser.add_argument('--wandb_project', type=str, default='AQA', help='Wandb project name')
        
        # visulize and debug
        parser.add_argument('--save_best', type=int, default=-1, help='save best model before N epoch')
        parser.add_argument('--save_interval', type=int, default=5, help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#iteration)')

        parser.add_argument('--check_best_epoch', type=int, default=0, help='if >0, check if best performance is achieved in the last N epochs')
        parser.add_argument('--check_best_src', type=float, default=0, help='the best src to reach')

        parser.add_argument('--accumulation_steps', type=int, default=1, help='accumulation steps')
        # feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')
        parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.0005, help='initial learning rate')
        parser.add_argument('--backbone_lr', type=float, default=0.00005, help='initial learning rate for backbone')
        parser.add_argument('--optimizer', default='Adam', help='type of optimizer')
        parser.add_argument(
                '--opt_step',
                type=int,
                default=[20, 40, 60],
                nargs='+',
                help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--opt_gamma', type=float, default=0.7,
                            help='optimizer  gamma of StepLR.')
        parser.add_argument('--opt_betas1', type=float, default=0.95,
                            help='optimizer  betas1 of Adam.') 
        parser.add_argument('--opt_betas2', type=float, default=0.999,
                            help='optimizer  betas2 of Adam.') 
        parser.add_argument('--opt_step_size', type=int, default=70.,
                            help='optimizer  step_size of StepLR.')
        parser.add_argument('--warm_up_epoch', type=int, default=0.,
                            help='optimizer warm up.')
        parser.add_argument('--optim_mom', type=float, default=0.9,
                            help='optimizer momentum.')
        parser.add_argument('--opt_lr_scheduler', type=str, default='StepLR',
                            help='the scheduler will be used.') 
        
        parser.add_argument('--opt_eta_min', type=float, default=1e-6,
                            help='optimizer  eta_min of CosineAnnealingLR or CosineAnnealingWarmRestarts.')
        parser.add_argument('--opt_T_max', type=int, default=200.,
                            help='optimizer  T_max of CosineAnnealingLR or CosineAnnealingWarmRestarts.')
        parser.add_argument('--opt_T_mult', type=int, default=2.,
                            help='optimizer  T_mult of CosineAnnealingWarmRestarts.')
        parser.add_argument('--opt_patience', type=int, default=5,
                            help='optimizer  patience of ReduceLROnPlateau.')
        parser.add_argument('--opt_factor', type=float, default=0.1,
                            help='optimizer  factor of ReduceLROnPlateau.')      
        parser.add_argument('--opt_threshold', type=float, default=0.0001,  
                            help='optimizer  threshold of ReduceLROnPlateau.')
    
        parser.add_argument('--dataset', type=str, default=None)
        parser.add_argument('--dataset_repath', type=str2bool, default=False, help='change dataset path for RG or Fisv')
        parser.add_argument('--inputfeature', type=str2bool, default=False, help='use input feature or not')
      
        parser.add_argument('--num_people', type=int, default=1, help='the number of people in the dataset')
        parser.add_argument('--dataset_M_repath', type=str2bool, default=False, help='change dataset num of people')
       
        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--backbone_unfreeze_epoch', type=int, default=0, help='the epoch to unfreeze the backbone')
        
        parser.add_argument('--backbone_args', action=DictAction, default=dict(), help='the arguments of backbone')
        parser.add_argument('--neck_args', action=DictAction, default=dict(), help='the arguments of neck')
        parser.add_argument('--head_args', action=DictAction, default=dict(), help='the arguments of head')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--featureadapter_args', action=DictAction, default=dict(), help='the arguments of feature adapter')
        parser.add_argument('--featureadapter_args.type', type=str, help="type for feature adapter.")

        parser.add_argument('--backbone_args.type', type=str, help="type for model.")
        parser.add_argument('--model_args.activation', type=str, help="Activation function for the model.")
        parser.add_argument('--model_args.clip_num', type=int, help="Number of clips.")
        parser.add_argument('--model_args.overlap_ratio', type=float, help="Overlap ratio for clips.")
        parser.add_argument('--backbone_args.args.dropout', type=float, help="Input dropout for backbone.")
        parser.add_argument('--neck_args.args.in_features', type=int, help="Input features for neck.")
        parser.add_argument('--neck_args.args.hidden_size', type=int, help="Input hidden_size features for neck.")
        parser.add_argument('--neck_args.args.out_features', type=int, help="Output features for neck.")
        parser.add_argument('--neck_args.args.dropout', type=float, help="Input dropout for neck_args.")
        parser.add_argument('--neck_args.type', type=str, help="type for neck.")

        parser.add_argument('--head_args.type', type=str, help="type for head.")
        parser.add_argument('--head_args.args.in_features', type=int, help="Input features for head.")
        parser.add_argument('--head_args.args.hidden_size', type=int, help="Hidden size for LSTM head.")
        parser.add_argument('--head_args.args.num_layers', type=int, help="Number of layers for LSTM head.")
        parser.add_argument('--head_args.args.dropout', type=float, help="Dropout for LSTM head.")
        parser.add_argument('--head_args.args.feature_mean', type=str, help="feature_mean for LSTM head.")
        parser.add_argument('--head_args.args.num_targets', type=int, help="Number of targets for head.")
        parser.add_argument('--head_args.args.t_min_dis', type=float, help="Minimum distance for targets loss.")
        # feeder
        parser.add_argument('--data_clip', type=str2bool, default=True, help='clip the data')
        #endregion yapf: enable

        return parser
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
