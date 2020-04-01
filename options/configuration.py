'''
python train.py --dataroot ./datasets/shrec_16 --name shrec16 --ncf 64 128 256 256 --pool_res 600 450 300 180 --norm group --resblocks 1
--flip_edges 0.2 --slide_verts 0.2 --num_aug 20 --niter_decay 100
'''
class Configuration:
    def __init__(self):
        self.dataroot                     = "./datasets/shrec_16"
        self.dataset_mode                 = "classification"
        self.ninput_edges                 = 750 # of input edges (will include dummy edges)
        self.max_dataset_size             = float("inf") #maximum number of samples per epoch
        self.batch_size                   = 16 #input batch size
        self.arch                         = 'mconvnet' #selects network to use
        self.resblocks                    = 1#0 # of res blocks
        self.fc_n                         = 100 # between fc and nclasses
        self.ncf                          = [64, 128, 256, 256]#[16, 32, 32] #conv filters
        self.pool_res                     = [600, 450, 300, 180]#[1140, 780, 580] #pooling res
        self.norm                         = 'group'#'batch' #instance normalization or batch normalization or group normalization
        self.num_groups                   = 16 # of groups for groupnorm
        self.init_type                    = 'normal' #network initialization [normal|xavier|kaiming|orthogonal]
        self.init_gain                    = 0.02#scaling factor for normal, xavier and orthogonal.
        self.num_threads                  = 3 # threads for loading data
        self.gpu_ids                      = '0'#'0' #gpu ids: e.g. 0  0,1,2, 0,2. use -1 for cpu
        self.name                         = "shrec16"#'debug' #name of the experiment. it decides where to store samples and models
        self.checkpoints_dir              = './checkpoints' #models are saved here
        self.serial_batches               = 'store_true' #if true, takes meshes in order, otherwise takes them randomly
        #### mainly for the random seed things see the base_options.py code 
        self.seed                         = None#if specified, uses seed 
        self.export_folder                = '' #exports intermediate collapses to this folder

        #######################################train####################################
        self.print_freq                   = 10 #frequency of showing training results on console
        self.save_latest_freq             = 250 #frequency of saving the latest results
        self.save_epoch_freq              = 1 #frequency of saving checkpoints at the end of epochs
        self.run_test_freq                = 1 #frequency of running test in training script
        self.continue_train               = False #continue training: load the latest model
        self.epoch_count                  = 1 # 'the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.phase                        = 'train' #train, val, test, etc
        self.which_epoch                  = 'latest' #'which epoch to load? set to latest to use latest cached model')
        self.niter                        = 100 ## of iter at starting learning rate
        self.niter_decay                  = 100 # of iter to linearly decay learning rate to zero')
        self.beta1                        = 0.9 # momentum term of adam
        self.lr                           = 0.0002 #initial learning rate for adam
        self.lr_policy                    = 'lambda' #'learning rate policy: lambda|step|plateau')
        self.lr_decay_iters               = 50 #'multiply by a gamma every lr_decay_iters iterations')
        self.num_aug                      = 10 # of augmentation files'
        self.scale_verts                  = False #non-uniformly scale the mesh e.g., in x, y or z
        self.slide_verts                  = 0.2 #'percent vertices which will be shifted along the mesh surface')
        self.flip_edges                   = 0.2 #percent of edges to randomly flip
        self.no_vis                       = False #will not use tensorboard
        self.verbose_plot                 = False #plots network weights, etc.

        #######################################TEST####################################
        self.results_dir                  = './results/' #'saves results here.'
        self.phase                        = "test" #train, val, test, etc
        self.which_epoch                  = 'latest' # 'which epoch to load? set to latest to use latest cached model')
        self.num_aug                      = 20 # of augmentation files
        self.is_train                     = True # is training req
