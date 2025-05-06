# I created this data loader by refering following great reseaches & github repos.
# VP: stochastic video generation https://github.com/edenton/svg
# MP: On human motion prediction using recurrent neural network https://github.com/wei-mao-2019/LearnTrajDep
#     Trajectron++ https://github.com/StanfordASL/Trajectron-plus-plus
#     Motion Indeterminacy Diffusion https://github.com/gutianpei/mid
# TP: Social GAN https://github.com/agrimgupta92/sgan
from pathlib import Path
import dill
from yacs.config import CfgNode
import torch 
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

import os
import sys
sys.path.append('../src')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data/TP')))
print("Current working directory:", os.getcwd())
#from TP.environment import Environment, Scene, Node, derivative_of

"""

FUNZIONANTE

def unified_loader(cfg: CfgNode, rand=True, split="train", batch_size=None, aug_scene=False, custom_pkl_path=None) -> DataLoader:
    # train, val, test
    if cfg.DATA.TASK == "TP":
        from .TP.trajectron_dataset import EnvironmentDataset, hypers

            
        if 'longer' in cfg.DATA.DATASET_NAME and split != "train":
            i = int(cfg.DATA.DATASET_NAME[-1])
            cfg.defrost()
            cfg.DATA.OBSERVE_LENGTH -= i
            cfg.DATA.DATASET_NAME = cfg.DATA.DATASET_NAME[:-8]
            cfg.freeze()
        
        if cfg.DATA.DATASET_NAME == 'sdd' and split != 'train':
            i = cfg.DATA.PREDICT_LENGTH - 12
            cfg.defrost()
            cfg.DATA.OBSERVE_LENGTH -= i
            cfg.freeze()
        

        if cfg.DATA.DATASET_NAME == "sdd" and split == "val":
            env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / 'processed_data' / f"{cfg.DATA.DATASET_NAME}_test.pkl"
        else:
            env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / 'processed_data' / f"{cfg.DATA.DATASET_NAME}_{split}.pkl"


    with open(env_path, 'rb') as f:
        env = dill.load(f, encoding='latin1')
        print(env_path, " + env: ", env)
    

    # Crea il dataset
    dataset = EnvironmentDataset(
        env,
        state=hypers[cfg.DATA.TP.STATE],
        pred_state=hypers[cfg.DATA.TP.PRED_STATE],
        node_freq_mult=hypers['scene_freq_mult_train'],
        scene_freq_mult=hypers['node_freq_mult_train'],
        hyperparams=hypers,
        min_history_timesteps=1 if cfg.DATA.TP.ACCEPT_NAN and split == 'train' else cfg.DATA.OBSERVE_LENGTH - 1,
        min_future_timesteps=cfg.DATA.PREDICT_LENGTH,
        augment=aug_scene
    )

    
    # Filtra i nodi di tipo 'PEDESTRIAN'
    for node_type_dataset in dataset:
        if node_type_dataset.node_type == 'PEDESTRIAN':
            dataset = node_type_dataset
            break

            
    if cfg.DATA.TASK == "TP":
        from .TP.preprocessing import dict_collate as seq_collate

    if batch_size is None:
        batch_size = cfg.DATA.BATCH_SIZE

    # Crea il DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=rand,
        num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=seq_collate,
        drop_last=True if split == 'train' else False,
        pin_memory=True
    )

    return loader

"""

def unified_loader(cfg: CfgNode, rand=True, split="train", batch_size=None, aug_scene=False, custom_pkl_path=None) -> DataLoader:
    # train, val, test
    if cfg.DATA.TASK == "TP":
        from .TP.trajectron_dataset import EnvironmentDataset, hypers

    if custom_pkl_path is not None:
        print(f"Loading custom PKL file: {custom_pkl_path}")
        with open(custom_pkl_path, 'rb') as f:
            env = dill.load(f, encoding='latin1')

        dataset = EnvironmentDataset(
            env,
            state=hypers[cfg.DATA.TP.STATE],
            pred_state=hypers[cfg.DATA.TP.PRED_STATE],
            node_freq_mult=hypers['scene_freq_mult_train'],
            scene_freq_mult=hypers['node_freq_mult_train'],
            hyperparams=hypers,
            min_history_timesteps=1 if cfg.DATA.TP.ACCEPT_NAN and split == 'train' else cfg.DATA.OBSERVE_LENGTH - 1,
            min_future_timesteps=cfg.DATA.PREDICT_LENGTH,
            augment=aug_scene
        )

        # Filtra solo i PEDESRTIAN
        for node_type_dataset in dataset:
            if node_type_dataset.node_type == 'PEDESTRIAN':
                dataset = node_type_dataset
                break

        if batch_size is None:
            batch_size = cfg.DATA.BATCH_SIZE

        from .TP.preprocessing import dict_collate as seq_collate

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=rand,
            num_workers=cfg.DATA.NUM_WORKERS,
            collate_fn=seq_collate,
            drop_last=True if split == 'train' else False,
            pin_memory=True
        )

        return loader
