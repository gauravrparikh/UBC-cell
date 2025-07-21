import sys
# sys.path.append("/usr/xtmp/zg78/proto_rset/")
# sys.path.append("/usr/xtmp/zg78/proto_rset/rashomon_sets")
from protopnet.visualization import *
import torch
from rashomon_sets.protorset_factory import ProtoRSetFactory
import os
import numpy as np
from pathlib import Path
import random
from app import GOLD_STD_REMOVALS, prep_rset

if __name__ == '__main__':
    N_JOBS = 20
    slurm_id = int(sys.argv[1])

    SEED = 1234
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    ui_rset = prep_rset()
    train_loader = ui_rset.viz_dataloader
    val_loader = ui_rset.val_dataloader

    model_path = Path("/usr/xtmp/jcd97/proto-rset/wandb/live/artifacts/739s0fb8/54_last_only_0.4579.pth")
    model = torch.load(model_path)

    model.prune_duplicate_prototypes()

    for i in range(model.prototype_layer.num_prototypes):
        if i % N_JOBS == slurm_id:
            ui_rset.display_global_analysis_for_proto(i)
