import argparse
import os
from util import util
import torch
from .configuration import Configuration as conf

class BaseOptions:
    def __init__(self):
        #self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # data params
        cf = conf()
        self.dataroot             = cf.dataroot
        self.dataset_mode         = cf.dataset_mode
        self.ninput_edges         = cf.ninput_edges
        self.max_dataset_size     = cf.max_dataset_size
        # network params
        self.batch_size           = cf.batch_size
        self.arch                 = cf.arch
        self.resblocks            = cf.resblocks
        self.fc_n                 = cf.fc_n
        self.ncf                  = cf.ncf
        self.pool_res             = cf.pool_res
        self.norm                 = cf.norm
        self.num_groups           = cf.num_groups
        self.init_type            = cf.init_type
        self.init_gain            = cf.init_gain
        # general params
        self.num_threads          = cf.num_threads
        self.gpu_ids              = cf.gpu_ids
        self.name                 = cf.name
        self.checkpoints_dir      = cf.checkpoints_dir
        self.serial_batches       = cf.serial_batches
        self.seed                 = cf.seed
        # visualization params
        self.export_folder        = cf.export_folder
        #
        self.initialized          = True
        self.is_train             = cf.is_train



        
    def setup(self):
        if not self.initialized:
            self.initialize()
        #self.opt, unknown = self.parser.parse_known_args()
        #self.opt.is_train = self.is_train   # train or test

        str_ids = self.gpu_ids.split(',')
        self.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.gpu_ids.append(id)
        # set gpu ids
        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])

        #args = vars(self.opt)

        if self.seed is not None:
            import numpy as np
            import random
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        if self.export_folder:
            self.export_folder = os.path.join(self.checkpoints_dir, self.name, self.export_folder)
            util.mkdir(self.export_folder)

        if self.is_train:
            #print('------------ Options -------------')
            #for k, v in sorted(args.items()):
            #    print('%s: %s' % (str(k), str(v)))
            #print('-------------- End ----------------')

            # save to the disk
            expr_dir = os.path.join(self.checkpoints_dir, self.name)
            util.mkdir(expr_dir)

            '''
            ### this is mainly for storing the selected training options . 
            ### pls try to add this somehow
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
            '''
        #return self.opt
