import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *

from models.gamm import GAMM_UNet
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")


def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join('./best.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('test', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()
    


    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config    
    model = GAMM_UNet()
    
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])


    print('#----------Preparing dataset----------#')
    test_loader_dict = {}
    for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        test_dataset = Polyp_datasets(path_Data = config.data_path, train = False, Test = True, test_dataset=dataset)
        test_loader = DataLoader(test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True, 
                                    num_workers=config.num_workers,
                                    drop_last=True)

        test_loader_dict[dataset] = test_loader

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1


    print('#----------Testing----------#')
    ckpt = load_checkpoint(resume_model, torch.device('cpu'))

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    try:
        model.load_state_dict(state_dict)
    except Exception:
        sd = state_dict
        model_keys = set(model.state_dict().keys())
        sd_keys = set(sd.keys())
        if any(k.startswith('module.') for k in sd_keys) and not any(k.startswith('module.') for k in model_keys):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        elif not any(k.startswith('module.') for k in sd_keys) and any(k.startswith('module.') for k in model_keys):
            sd = {'module.' + k: v for k, v in sd.items()}

        try:
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            if isinstance(ckpt, dict):
                raise RuntimeError(f'Failed to load checkpoint. Available top-level keys: {list(ckpt.keys())}') from e
            raise
    for name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
                test_loader_t = test_loader_dict[name]
                loss = test_one_epoch(
                        test_loader_t,
                        model,
                        criterion,
                        logger,
                        config,
                        test_data_name=name
                    )     



if __name__ == '__main__':
    config = setting_config
    main(config)