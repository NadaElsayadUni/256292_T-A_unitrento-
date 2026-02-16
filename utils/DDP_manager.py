# MULTI DATA PARALLELIZATION
from torch.distributed import init_process_group, destroy_process_group
import os
import datetime
import torch

'''
    This class is used to manage the DDP setup and teardown
    for a given rank and world_size
'''
class DDP():
    def __init__(
        self,
        rank,
        world_size
    ):
        self.rank = rank
        self.world_size = world_size
        # Check for available device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = f"cuda:{rank}"
            torch.cuda.set_device(self.rank)
            backend = "nccl"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            backend = "gloo"  # MPS doesn't support distributed, but we use gloo for single process
        else:
            self.device = "cpu"
            backend = "gloo"
        self.ddp_setup(rank, world_size, backend)
        self.main()
        destroy_process_group()

    def main(self):
        pass

    def ddp_setup(self, rank, world_size, backend="nccl"):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12398"
        init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=5400))


