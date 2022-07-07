from pathlib import Path
from yacs.config import CfgNode as CN
import os
import logging


_C = CN()

# GPU and distributed setting
_C.world_size = 1
_C.rank = 0
_C.dist_url = 'tcp://localhost:10000'
_C.dist_backend = 'nccl'
_C.seed = None
_C.gpu = None
_C.evaluate = False
_C.multiprocessing_distributed = True
_C.workers = 1

# Dataset setting
_C.dataset = ''
_C.data_dir = ''
_C.num_classes = -1
_C.train_txt_path = ''
_C.val_txt_path = ''
_C.imb_factor = 1.0 # For CIFAR only

# General training setting
_C.log_dir = 'logs'
_C.model_dir = 'ckps'
_C.resume = ''
_C.mark = ''
_C.debug = False
_C.batch_size = 32
_C.lr = 0.02
_C.momentum = 0.9
_C.weight_decay = 5e-4
_C.print_freq = 100
_C.start_eval_epoch = -1
_C.warm_epochs = 10
_C.start_epoch = 1
_C.epochs = 400

# Network architecture setting
_C.arch = ''
_C.dim_feat = -1
_C.dim_con = -1

# Options for ResCom
_C.queue_size_per_cls = 1
_C.select_num_pos = 1
_C.select_num_neg = 1
_C.con_weight = 1.0
_C.balsfx_n = 0.0
_C.effective_num_beta = 0.0
_C.temperature = 0.1



def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    log_dir = Path("saved")  / (cfg.mark) / Path(cfg.log_dir)
    print('=> creating {}'.format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = '{}.txt'.format(cfg.mark)
    # final_log_file = log_dir / log_file 
    
    model_dir =  Path("saved") / (cfg.mark) / Path(cfg.model_dir)
    print('=> creating {}'.format(model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_dir = str(model_dir)


class NoOperation:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            pass

        return no_op


def get_logger(config, resume=False, is_rank0=True):
    if is_rank0:
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        # # StreamHandler
        # stream_handler = logging.StreamHandler(sys.stdout)
        # stream_handler.setLevel(level=logging.INFO)
        # logger.addHandler(stream_handler)

        # FileHandler
        if resume == False:
            mode = "w+" 
        else:
            mode = "a+"
        log_dir = Path("saved") / (config.mark) / Path(config.log_dir)
        log_name = config.mark + ".log"
        file_handler = logging.FileHandler(os.path.join(log_dir, log_name), mode=mode)
        file_handler.setLevel(level=logging.INFO)
        logger.addHandler(file_handler)
    else:
        logger = NoOperation()

    return logger