import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

class SiameseSignatureDataset(Dataset):
    """
    Siamese Dataset for Signature Verification.

    Generates:
        Positive pairs  -> (real, real) of same person
        Negative pairs  -> (real, forged) OR (real, real of different person)
    """
    def __init__(
        self,
        root_dir: str | Path,
        person_ids: list[str],
        transform=None,
        positive_ratio: float = 0.5,
        seed: int = 42
    ):
        self.root_dir = Path(root_dir)
        self.person_ids = person_ids
        self.transform = transform
        self.positive_ratio = positive_ratio
        self.rng = random.Random(seed)
        self.data_index = self._build_index()


    def _build_index(self):
        index = {}

        for folder in self.root_dir.iterdir():
            if not folder.is_dir():
                continue

            folder_name = folder.name

            # Check if forged folder
            if folder_name.endswith("_forg"):
                continue  # skip here, handle with real folder

            person_id = folder_name
            real_dir = folder
            forged_dir = self.root_dir / f"{person_id}_forg"

            real_images = list(real_dir.glob("*"))
            forged_images = list(forged_dir.glob("*")) if forged_dir.exists() else []

            if len(real_images) < 2:
                continue

            index[person_id] = {
                "real": real_images,
                "forged": forged_images
            }

        return index

    def __len__(self):
        # Arbitrary large size since pairs are generated dynamically
        return 10000
    
    def _load_image(self, path: Path):
        """
        Loads image as grayscale.
        Signature verification does not require RGB.
        """
        img = Image.open(path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img
    

    def __getitem__(self, idx):

        # Decide positive or negative
        is_positive = self.rng.random() < self.positive_ratio

        pid = self.rng.choice(list(self.data_index.keys()))
        person_data = self.data_index[pid]

        if is_positive:
            # ---------------------------------------
            # POSITIVE PAIR
            # Two real signatures from same person
            # Goal: embeddings should be CLOSE
            # ---------------------------------------
            img1_path, img2_path = self.rng.sample(person_data["real"], 2)
            label = 1.0
        else:
            # ---------------------------------------
            # NEGATIVE PAIR
            # Goal: embeddings should be FAR
            #
            # We mix two types:
            # 1) real vs forged (same identity)
            # 2) real vs real (different identities)
            # ---------------------------------------
            if self.rng.random() < 0.5 and len(person_data["forged"]) > 0:
                img1_path = self.rng.choice(person_data["real"])
                img2_path = self.rng.choice(person_data["forged"])
            else:
                pid2 = self.rng.choice(
                    [p for p in self.person_ids if p != pid]
                )
                img1_path = self.rng.choice(self.data_index[pid]["real"])
                img2_path = self.rng.choice(self.data_index[pid2]["real"])

            label = 0.0

        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)

        return img1, img2, torch.tensor(label, dtype=torch.float32)
    



class TripletSignatureDataset(Dataset):
    """
    Siamese-style Triplet Dataset for Signature Verification.

    Generates:
        Triplets (anchor, positive, negative) where:
            - anchor: real signature
            - positive: another real signature of the same person
            - negative: either a forged signature of same person or real signature of a different person
    """
    def __init__(
        self,
        root_dir: str | Path,
        person_ids: list[str],
        transform=None,
        seed: int = 42
    ):
        self.root_dir = Path(root_dir)
        self.person_ids = person_ids
        self.transform = transform
        self.rng = random.Random(seed)
        self.data_index = self._build_index()

    def _build_index(self):
        index = {}
        for folder in self.root_dir.iterdir():
            if not folder.is_dir():
                continue
            folder_name = folder.name
            if folder_name.endswith("_forg"):
                continue
            person_id = folder_name
            real_dir = folder
            forged_dir = self.root_dir / f"{person_id}_forg"

            real_images = list(real_dir.glob("*"))
            forged_images = list(forged_dir.glob("*")) if forged_dir.exists() else []

            if len(real_images) < 2:
                continue

            index[person_id] = {
                "real": real_images,
                "forged": forged_images
            }
        return index

    def __len__(self):
        # Arbitrary large size since triplets are generated dynamically
        return 10000

    def _load_image(self, path: Path):
        img = Image.open(path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img

    # def __getitem__(self, idx):
    #     # ---------------- Anchor & Positive ----------------
    #     pid = self.rng.choice(list(self.data_index.keys()))
    #     person_data = self.data_index[pid]
    #     anchor_path, positive_path = self.rng.sample(person_data["real"], 2)

    #     # ---------------- Negative ----------------
    #     if self.rng.random() < 0.5 and len(person_data["forged"]) > 0:
    #         negative_path = self.rng.choice(person_data["forged"])
    #     else:
    #         # Real signature from a different person
    #         pid2 = self.rng.choice([p for p in self.person_ids if p != pid])
    #         negative_path = self.rng.choice(self.data_index[pid2]["real"])

    #     anchor = self._load_image(anchor_path)
    #     positive = self._load_image(positive_path)
    #     negative = self._load_image(negative_path)

    #     return anchor, positive, negative

    def __getitem__(self, idx):
        # ---------------- Anchor & Positive ----------------
        pid = self.rng.choice(list(self.data_index.keys()))
        person_data = self.data_index[pid]
        
        # Ensure there are at least 2 real signatures
        anchor_path, positive_path = self.rng.sample(person_data["real"], 2)

        # ---------------- Negative ----------------
        negative_path = None
        has_forged = len(person_data.get("forged", [])) > 0

        if has_forged:
            # 80% chance to pick same-person forgery, 20% other-person real
            if self.rng.random() < 0.8:
                negative_path = self.rng.choice(person_data["forged"])
        
        if negative_path is None:
            # Pick a real signature from a different person
            other_pids = [p for p in self.person_ids if p != pid]
            pid2 = self.rng.choice(other_pids)
            negative_path = self.rng.choice(self.data_index[pid2]["real"])

        # ---------------- Load Images ----------------
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)

        return anchor, positive, negative