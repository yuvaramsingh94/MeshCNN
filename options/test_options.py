from .base_options import BaseOptions
from .configuration import Configuration as conf

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        cf = conf()
        self.results_dir= cf.results_dir
        self.phase= cf.phase
        self.which_epoch= cf.which_epoch
        self.num_aug= cf.num_aug
        self.is_train = False