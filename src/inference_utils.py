import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image


def find_best_threshold(pos_dist, neg_dist):
    pos = pos_dist.numpy()
    neg = neg_dist.numpy()

    thresholds = np.linspace(min(pos.min(), neg.min()),
                             max(pos.max(), neg.max()),
                             200)

    best_acc = 0
    best_t = 0

    for t in thresholds:
        pos_correct = (pos < t).sum()
        neg_correct = (neg >= t).sum()
        acc = (pos_correct + neg_correct) / (len(pos) + len(neg))

        if acc > best_acc:
            best_acc = acc
            best_t = t

    return best_t, best_acc




def evaluate_with_best_threshold(pos_dist, neg_dist):

    pos = pos_dist.cpu().numpy()
    neg = neg_dist.cpu().numpy()

    # find best threshold
    best_t, best_acc = find_best_threshold(pos_dist, neg_dist)

    # labels
    y_true = np.concatenate([
        np.ones(len(pos)),   # genuine
        np.zeros(len(neg))   # forged
    ])

    # distances
    distances = np.concatenate([pos, neg])

    # predictions
    y_pred = (distances < best_t).astype(int)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    print("Best threshold:", best_t)
    print("Best accuracy:", best_acc)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,digits=4))

    return cm

def plot_triplet_distance_distributions(pos_dist, neg_dist, bins=50):
    """
    pos_dist: tensor of anchor-positive distances
    neg_dist: tensor of anchor-negative distances
    """

    pos = pos_dist.cpu().numpy()
    neg = neg_dist.cpu().numpy()

    plt.figure()

    plt.hist(pos, bins=bins, alpha=0.6, density=True, label="Anchor-Positive")
    plt.hist(neg, bins=bins, alpha=0.6, density=True, label="Anchor-Negative")

    plt.xlabel("Embedding Distance")
    plt.ylabel("Density")
    plt.title("Triplet Distance Distribution")
    plt.legend()
    plt.grid(True)

    plt.show()


# ----------------------------
# GRAD CAM UTILS
# ----------------------------
# Wrapper created because the original model expects triplet inputs, but Grad-CAM requires a single-image forward pass.
# This wrapper exposes the single-image embedding (forward_once) as a normal forward pass.
class EmbeddingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_once(x)

# ----------------------------
# UTILS
# ----------------------------
def tensor_to_numpy(t):
    img = t.squeeze().detach().cpu().numpy()

    if img.shape[0] in [1,3]:
        img = np.transpose(img, (1,2,0))

    img = img - img.min()
    img = img / (img.max() + 1e-8)

    return img

# ----------------------------
# VISUALIZATION FUNCTION
# ----------------------------
def visualize_triplet(device, model, triplet, cam, idx):

    anchor, positive, negative = triplet

    # take first image from batch
    anchor = anchor[0:1].to(device)
    positive = positive[0:1].to(device)
    negative = negative[0:1].to(device)

    model.eval()

    with torch.no_grad():
        z_a = model.forward_once(anchor)
        z_p = model.forward_once(positive)
        z_n = model.forward_once(negative)

        dist_ap = torch.norm(z_a - z_p, dim=1).item()
        dist_an = torch.norm(z_a - z_n, dim=1).item()

    imgs = [anchor, positive, negative]
    titles = ["Anchor", "Positive", "Negative"]

    fig, axes = plt.subplots(1,3,figsize=(12,4))

    for i, img in enumerate(imgs):

        grayscale_cam = cam(input_tensor=img)[0]

        rgb = tensor_to_numpy(img)
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

        axes[i].imshow(overlay)
        axes[i].set_title(titles[i])
        axes[i].axis("off")

    plt.suptitle(f"Triplet {idx} | AP: {dist_ap:.3f} | AN: {dist_an:.3f}")
    plt.show()