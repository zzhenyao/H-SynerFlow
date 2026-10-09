import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn

# torchlight
import torchlight
from torchlight.io import str2bool
from torchlight.io import DictAction
from torchlight.io import import_class

class IO():
    """
        IO Processor
    """

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        #self.load_weights()
        self.gpu()

    def load_arg(self, argv=None):
        parser = self.get_parser()

        # load arg form config file
        p = parser.parse_args(argv)
        if p.config is not None:
            # load config file
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)

            # update parser from config file
            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('Unknown Arguments: {}'.format(k))
                    assert k in key

            parser.set_defaults(**default_arg)

        self.arg = parser.parse_args(argv)

    def init_environment(self):
        self.io = torchlight.IO(
            self.arg.exp_dir,
            save_log= False,
            print_log=False)
        self.io.save_arg(self.arg)

        # gpu
        if self.arg.use_gpu:
            gpus = torchlight.visible_gpu(self.arg.device)
            torchlight.occupy_gpu(gpus)
            self.gpus = gpus
            self.dev = "cuda:0"
        else:
            self.dev = "cpu"

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))

    def load_weights(self):
        
        self.model.backbone = self.io.load_weights(self.model.backbone, 
                                        self.arg.backbone_args['args']['weights'],
                                        self.arg.backbone_args['args']['ignore_weights'])
    




    def gpu(self):
        # move modules to gpu
        self.model = self.model.to(self.dev)
        for name, value in vars(self).items():
            cls_name = str(value.__class__)
            if cls_name.find('torch.nn.modules') != -1:
                setattr(self, name, value.to(self.dev))

        # model parallel
        if self.arg.use_gpu and len(self.gpus) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)

    def start(self):
        return None

    @staticmethod
    def get_parser(add_help=False):


        parser = argparse.ArgumentParser( add_help=add_help, description='IO Processor')

        parser.add_argument('--exp_dir', default='./outputs/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')


        return parser
