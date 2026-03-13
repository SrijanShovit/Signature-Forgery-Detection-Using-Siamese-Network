import matplotlib.pyplot as plt
import torch
import numpy as np

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