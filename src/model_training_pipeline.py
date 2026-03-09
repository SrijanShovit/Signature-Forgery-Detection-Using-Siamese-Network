import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, MeanMetric
import torchvision.models as models

class SiameseModel(L.LightningModule):
    def __init__(self, embedding_dim: int = 128, lr: float = 1e-3, margin: float = 1.0):
        super().__init__()

        self.margin = margin
        self.lr = lr

        # backbone
        self.backbone = self._build_backbone()

        # embedding head
        self.embedding_head = nn.Linear(512, embedding_dim)

        # train metrics
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

        # val metrics
        self.val_auc = BinaryAUROC()
        self.val_outputs: list = []

        # test containers
        self.test_outputs: list = []
        self.test_threshold: float|None = None


    # -------------------- Model Bakcbone --------------------
    def _build_backbone(self):
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # freeze backbone
        for param in resnet.parameters():
            param.requires_grad = False

        # Unfreeze last 2 blocks (layer3 and layer4)
        for param in resnet.layer3.parameters():
            param.requires_grad = True

        for param in resnet.layer4.parameters():
            param.requires_grad = True

        # Build backbone sequentially including trainable layers
        backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # frozen
            resnet.layer2,  # frozen
            resnet.layer3,  # trainable
            resnet.layer4,  # trainable
            resnet.avgpool
        )

        return backbone


    # -------------------- Forward --------------------
    def forward_once(self, x):
        # x shape: (B, 1, 128, 128)
        x = self.backbone(x)  # (B, 512, 1, 1)
        x = torch.flatten(x, 1) # Flatten spatial dims --> # (B, 512)
        x = self.embedding_head(x) # Linear(512 → embedding_dim) --> (B,Demb)
        x = F.normalize(x, p=2, dim=1)     # p = 2 means L2 normalize embeddings
        return x

    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2
    
    # -------------------- Loss --------------------
    def contrastive_loss(self, z1, z2, y):
        distances = F.pairwise_distance(z1, z2)
        loss = y * distances.pow(2) + (1 - y) * F.relu(self.margin - distances).pow(2)
        return loss.mean(), distances
    
    # -------------------- Training Step --------------------
    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        y = y.float()

        # Compute embeddings and distances
        z1, z2 = self(x1, x2)
        distances = F.pairwise_distance(z1, z2)

        # Select hard and easy pairs (balanced hard mining)
        z1_selected, z2_selected, y_selected = self._select_hard_easy_pairs(
            z1, z2, y, distances, batch_idx
        )

        # Compute contrastive loss and accuracy
        loss, distances_selected = self.contrastive_loss(z1_selected, z2_selected, y_selected)
        preds = (distances_selected < self.margin / 2).float()
        acc = self.train_acc(preds, y_selected)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss


    # -------------------- Helper: Hard + Easy Pair Selection --------------------
    def _select_hard_easy_pairs(self, z1, z2, y, distances, batch_idx):
        device = distances.device

        # Masks for positive/negative
        pos_mask = (y == 1)
        neg_mask = (y == 0)

        # ---------------- Top-k Hard Positives ----------------
        # Top-k Hard Positives
        if pos_mask.sum() > 0:
            pos_idx_local = torch.topk(distances[pos_mask], k=min(16, pos_mask.sum()))[1]
            pos_idx = torch.arange(len(y), device=device)[pos_mask][pos_idx_local]
        else:
            pos_idx = torch.tensor([], device=device, dtype=torch.long)

        # Top-k Hard Negatives
        if neg_mask.sum() > 0:
            neg_idx_local = torch.topk(-distances[neg_mask], k=min(16, neg_mask.sum()))[1]
            neg_idx = torch.arange(len(y), device=device)[neg_mask][neg_idx_local]
        else:
            neg_idx = torch.tensor([], device=device, dtype=torch.long)

        # Combine hard indices (no unique needed if disjoint)
        selected_idx = torch.cat([pos_idx, neg_idx])

        return z1[selected_idx], z2[selected_idx], y[selected_idx]
    
    
    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]

        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)

    
    # -------------------- Validation Step --------------------
    def on_validation_epoch_start(self):
        self.val_outputs = []

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        y = y.float()
        z1, z2 = self(x1, x2)
        loss, distances = self.contrastive_loss(z1, z2, y)
        preds = (distances < self.margin / 2).float()
        acc = self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True,on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True,on_step=False, on_epoch=True)

        self.val_outputs.append({
            "distances": distances.detach().cpu(),
            "labels": y.detach().cpu()
        })
    
    def on_validation_epoch_end(self):
        distances = torch.cat([x["distances"] for x in self.val_outputs])
        labels = torch.cat([x["labels"] for x in self.val_outputs])

        # smaller distance = more similar → invert for ROC
        scores = -distances

        auc = self.val_auc(scores, labels.int())

        self.log("val_auc", auc, prog_bar=True,on_step=False, on_epoch=True)

        self.val_outputs.clear()


    # -------------------- Predictions --------------------
    def calc_preds(self,x1,x2,threshold=None):
        z1, z2 = self(x1, x2)
        distances = F.pairwise_distance(z1, z2)
        # Use threshold if provided, otherwise fallback to margin/2
        t = threshold if threshold is not None else self.margin / 2
        preds = (distances < t).float()
        # detach --> ensures predictions/distances are not tracked by autograd, saving memory
        return preds.detach(), distances.detach()
    
    # -------------------- Test Step --------------------
    # -------------------- Batch Inference --------------------
    def on_test_epoch_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        x1, x2, y = batch
        y = y.float()
        preds, distances = self.calc_preds(x1, x2, threshold=self.test_threshold)

        self.test_outputs.append({
            "preds": preds.detach().cpu(),
            "labels": y.detach().cpu(),
            "distances": distances.detach().cpu()
        })

    # -------------------- Aggregated Results --------------------
    def on_test_epoch_end(self):
        all_preds = torch.cat([x["preds"] for x in self.test_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_outputs])
        all_distances = torch.cat([x["distances"] for x in self.test_outputs])

        print(f"Total samples: {len(all_labels)}")

        self.test_preds = all_preds
        self.test_labels = all_labels
        self.test_distances = all_distances

        # clear memory
        self.test_outputs.clear()
    

    # -------------------- Optimizer --------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }



class TripletSiameseModel(L.LightningModule):
    def __init__(self, embedding_dim: int = 128, lr: float = 1e-3, margin: float = 1.0,top_k: int = 16):
        super().__init__()

        self.margin = margin
        self.lr = lr
        self.top_k = top_k

        # backbone
        self.backbone = self._build_backbone()

        # embedding head
        self.embedding_head = nn.Linear(512, embedding_dim)

        # Metrics
        self.train_loss_metric = MeanMetric()
        self.val_loss_metric = MeanMetric()
        self.val_violation_metric = MeanMetric()

        # Test outputs
        self.test_outputs: list = []


    # -------------------- Model Bakcbone --------------------
    def _build_backbone(self):
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for param in resnet.parameters():
            param.requires_grad = False

        for param in resnet.layer3.parameters():
            param.requires_grad = True

        for param in resnet.layer4.parameters():
            param.requires_grad = True

        # Build backbone sequentially including trainable layers
        backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # frozen
            resnet.layer2,  # frozen
            resnet.layer3,  # trainable
            resnet.layer4,  # trainable
            resnet.avgpool
        )

        return backbone


    # -------------------- Forward --------------------
    def forward_once(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding_head(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, batch):
        anchor, positive, negative = batch
        z_a = self.forward_once(anchor)
        z_p = self.forward_once(positive)
        z_n = self.forward_once(negative)
        return z_a, z_p, z_n
    
    # -------------------- Triplet Loss --------------------
    def triplet_loss(self, anchor, positive, negative, margin=None):
        margin = margin if margin is not None else self.margin
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(pos_dist - neg_dist + margin)
        return losses.mean(), pos_dist, neg_dist
    

    # -------------------- Hard Mining --------------------
    def _select_hard_triplets(self, z_a, z_p, z_n):
        pos_dist = F.pairwise_distance(z_a, z_p, p=2)
        neg_dist = F.pairwise_distance(z_a, z_n, p=2)
        triplet_losses = F.relu(pos_dist - neg_dist + self.margin)

        if len(triplet_losses) > self.top_k:
            topk_idx = torch.topk(triplet_losses, k=self.top_k)[1]
            return z_a[topk_idx], z_p[topk_idx], z_n[topk_idx]
        else:
            return z_a, z_p, z_n

    
    # -------------------- Training Step --------------------
    def training_step(self, batch, batch_idx):
        z_a, z_p, z_n = self.forward(batch)

        z_a_sel, z_p_sel, z_n_sel = self._select_hard_triplets(z_a, z_p, z_n)
        loss, pos_dist, neg_dist = self.triplet_loss(z_a_sel, z_p_sel, z_n_sel)

        self.train_loss_metric.update(loss)
        self.log("train_loss", self.train_loss_metric, prog_bar=True, on_step=False, on_epoch=True)

        violations = (pos_dist - neg_dist + self.margin > 0).float().mean()
        self.log("train_triplet_violation", violations, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    
    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]

        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)
        self.train_loss_metric.reset()

    
    # -------------------- Validation Step --------------------
    def validation_step(self, batch, batch_idx):
        z_a, z_p, z_n = self.forward(batch)

        z_a_sel, z_p_sel, z_n_sel = self._select_hard_triplets(z_a, z_p, z_n)
        loss, pos_dist, neg_dist = self.triplet_loss(z_a_sel, z_p_sel, z_n_sel)

        self.val_loss_metric.update(loss)
        violations = (pos_dist - neg_dist + self.margin > 0).float().mean()
        self.val_violation_metric.update(violations)

        self.log("val_loss", self.val_loss_metric, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_triplet_violation", self.val_violation_metric, prog_bar=True, on_step=False, on_epoch=True)


    def on_validation_epoch_end(self):
        self.val_loss_metric.reset()
        self.val_violation_metric.reset()


    # -------------------- Test / Prediction --------------------
    def on_test_epoch_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        z_a, z_p, z_n = self.forward(batch)
        
        # Compute distances
        pos_dist = F.pairwise_distance(z_a, z_p)
        neg_dist = F.pairwise_distance(z_a, z_n)

        self.test_outputs.append({
            "pos_dist": pos_dist.detach().cpu(),
            "neg_dist": neg_dist.detach().cpu()
        })


    def on_test_epoch_end(self):
        pos_distances = torch.cat([x["pos_dist"] for x in self.test_outputs])
        neg_distances = torch.cat([x["neg_dist"] for x in self.test_outputs])

        self.test_outputs.clear()

        # Store for later evaluation
        self.test_pos_distances = pos_distances
        self.test_neg_distances = neg_distances

        print(f"Test samples: {len(pos_distances)}")
        print(f"Pos distance mean: {pos_distances.mean():.4f}, Neg distance mean: {neg_distances.mean():.4f}")


    # -------------------- Batch Inference (Optional) --------------------
    def predict_distances(self, anchor, positive, negative=None):
        """
        Returns distances:
            - If negative is None: returns only anchor-positive distance
            - Otherwise: returns anchor-positive and anchor-negative distances
        """
        z_a = self.forward_once(anchor)
        z_p = self.forward_once(positive)
        pos_dist = F.pairwise_distance(z_a, z_p)

        if negative is not None:
            z_n = self.forward_once(negative)
            neg_dist = F.pairwise_distance(z_a, z_n)
            return pos_dist.detach(), neg_dist.detach()
        return pos_dist.detach()
    

    # -------------------- Optimizer --------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}