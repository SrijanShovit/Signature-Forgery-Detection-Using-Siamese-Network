import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
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
    print(classification_report(y_true, y_pred))

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