from .base_options import BaseOptions
from .configuration import Configuration as conf
class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        cf = conf()
        self.print_freq                   = cf.print_freq
        self.save_latest_freq             = cf.save_latest_freq
        self.save_epoch_freq              = cf.save_epoch_freq
        self.run_test_freq                = cf.run_test_freq
        self.continue_train               = cf.continue_train
        self.epoch_count                  = cf.epoch_count
        self.phase                        = cf.phase
        self.which_epoch                  = cf.which_epoch
        self.niter                        = cf.niter
        self.niter_decay                  = cf.niter_decay
        self.beta1                        = cf.beta1
        self.lr                           = cf.lr
        self.lr_policy                    = cf.lr_policy
        self.lr_decay_iters               = cf.lr_decay_iters
        # data augmentation stuff
        self.num_aug                      = cf.num_aug
        self.scale_verts                  = cf.scale_verts
        self.slide_verts                  = cf.slide_verts
        self.flip_edges                   = cf.flip_edges
        # tensorboard visualization
        self.no_vis                       = cf.no_vis
        self.verbose_plot                 = cf.verbose_plot
        self.is_train                     = True
