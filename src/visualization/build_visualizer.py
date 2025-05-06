from yacs.config import CfgNode
from typing import Dict, List, Type

from visualization.TP_visualizer import TP_Visualizer, Visualizer

# Per i miei dati
def Build_Visualizer(cfg: CfgNode, env=None) -> Type[Visualizer]:
    if cfg.DATA.TASK == "TP":
        return TP_Visualizer(cfg, env=env)

# Per altri dataset
# def Build_Visualizer(cfg: CfgNode, env=None) -> Type[Visualizer]:
#     if cfg.DATA.TASK == "TP":
#         return TP_Visualizer(cfg, env=env)
