from __future__ import annotations

import sys
from pathlib import Path

import os
import torch
import random

import numpy as np
from mpi4py import MPI

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.option import args_parser
from src.utils.trainer import Trainer


def seed_everything(seed=42):
    # 1. 파이썬 내장 랜덤 시드 고정
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Numpy 시드 고정
    np.random.seed(seed)
    
    # 3. PyTorch 시드 고정 (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 멀티 GPU 사용 시


def main(argv=None):
    args = args_parser(argv)
    comm = MPI.COMM_WORLD
    seed_everything(args.seed)

    trainer = Trainer(args)
    data_state = trainer.prepare_data(comm)
    result = trainer.train(data_state, comm, logger=None, args=args)

    if comm.Get_rank() == 0 and result is not None:
        for message in result.get("messages", []):
            print(message)

if __name__ == "__main__":
    main()
