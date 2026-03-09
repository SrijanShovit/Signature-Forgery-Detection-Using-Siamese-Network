from pathlib import Path
from typing import Optional
from typing import Callable
from torch import Tensor
import lightning as L
from torch.utils.data import DataLoader, RandomSampler
import torch
import random
import numpy as np
from src.dataset import SiameseSignatureDataset, TripletSignatureDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

class SignatureDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_transformations: Optional[Callable[[Tensor], Tensor]] = None,
        val_transformations: Optional[Callable[[Tensor], Tensor]] = None,
        test_transformations: Optional[Callable[[Tensor], Tensor]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        positive_ratio: float = 0.5,
        image_size: int = 128,
        train_split: float = 0.8,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        samples_per_epoch: int = 10000,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.positive_ratio = positive_ratio
        self.image_size = image_size
        self.train_split = train_split
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.samples_per_epoch = samples_per_epoch
        self.train_transformations = train_transformations
        self.val_transformations = val_transformations
        self.test_transformations = test_transformations

    def setup(self, stage: Optional[str] = None):
        # ---------------- FIT (train + val) ----------------
        if stage == "fit" or stage is None:

            train_root = self.data_dir / "train"

            all_person_ids = sorted([
                p.name for p in train_root.iterdir()
                if p.is_dir() and not p.name.endswith("_forg")
            ])

            split_idx = int(len(all_person_ids) * self.train_split)
            train_ids = all_person_ids[:split_idx]
            val_ids = all_person_ids[split_idx:]

            self.train_dataset = SiameseSignatureDataset(
                root_dir=train_root,
                person_ids=train_ids,
                transform=self.train_transformations,
                positive_ratio=self.positive_ratio,
            )

            self.val_dataset = SiameseSignatureDataset(
                root_dir=train_root,
                person_ids=val_ids,
                transform=self.val_transformations,
                positive_ratio=self.positive_ratio,
            )

        # ---------------- TEST ----------------
        if stage == "test" or stage is None:

            test_root = self.data_dir / "test"

            test_ids = sorted([
                p.name for p in test_root.iterdir()
                if p.is_dir() and not p.name.endswith("_forg")
            ])

            self.test_dataset = SiameseSignatureDataset(
                root_dir=test_root,
                person_ids=test_ids,
                transform=self.test_transformations,
                positive_ratio=self.positive_ratio,
            )

    # ---------------- DATALOADERS ----------------

    def train_dataloader(self):

        sampler = RandomSampler(
            self.train_dataset,
            replacement=True,
            num_samples=self.samples_per_epoch,
        )

        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=g
        )

        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(self.train_dataset, **loader_kwargs) # type: ignore

    def val_dataloader(self):

        sampler = RandomSampler(
            self.val_dataset,
            replacement=True,
            num_samples=self.samples_per_epoch // 4,
        )

        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=g
        )

        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(self.val_dataset, **loader_kwargs) # type: ignore
    
    def test_dataloader(self):
        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=g
        )

        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(self.test_dataset, **loader_kwargs) # type: ignore



class TripletDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_transformations: Optional[Callable[[Tensor], Tensor]] = None,
        val_transformations: Optional[Callable[[Tensor], Tensor]] = None,
        test_transformations: Optional[Callable[[Tensor], Tensor]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 128,
        train_split: float = 0.8,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        samples_per_epoch: int = 10000,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_split = train_split
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.samples_per_epoch = samples_per_epoch
        self.train_transformations = train_transformations
        self.val_transformations = val_transformations
        self.test_transformations = test_transformations

    def setup(self, stage: Optional[str] = None):
        # ---------------- FIT (train + val) ----------------
        if stage == "fit" or stage is None:
            train_root = self.data_dir / "train"

            all_person_ids = sorted([
                p.name for p in train_root.iterdir() if p.is_dir() and not p.name.endswith("_forg")
            ])

            split_idx = int(len(all_person_ids) * self.train_split)
            train_ids = all_person_ids[:split_idx]
            val_ids = all_person_ids[split_idx:]

            self.train_dataset = TripletSignatureDataset(
                root_dir=train_root,
                person_ids=train_ids,
                transform=self.train_transformations,
            )

            self.val_dataset = TripletSignatureDataset(
                root_dir=train_root,
                person_ids=val_ids,
                transform=self.val_transformations,
            )

        # ---------------- TEST ----------------
        if stage == "test" or stage is None:
            test_root = self.data_dir / "test"

            test_ids = sorted([
                p.name for p in test_root.iterdir() if p.is_dir() and not p.name.endswith("_forg")
            ])

            self.test_dataset = TripletSignatureDataset(
                root_dir=test_root,
                person_ids=test_ids,
                transform=self.test_transformations,
            )

    # ---------------- DATALOADERS ----------------

    def train_dataloader(self):
        sampler = RandomSampler(
            self.train_dataset,
            replacement=True,
            num_samples=self.samples_per_epoch,
        )

        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=g
        )

        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(self.train_dataset, **loader_kwargs)  # type: ignore

    def val_dataloader(self):
        sampler = RandomSampler(
            self.val_dataset,
            replacement=True,
            num_samples=self.samples_per_epoch // 4,
        )

        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=g
        )

        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(self.val_dataset, **loader_kwargs)  # type: ignore

    def test_dataloader(self):
        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=g
        )

        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(self.test_dataset, **loader_kwargs)  # type: ignore