import torch
import matplotlib.pyplot as plt

def sanity_check_loader(loader, split_name: str):
    batch = next(iter(loader))
    img1, img2, labels = batch

    print(f"\n===== {split_name.upper()} SANITY CHECK =====")
    print("img1 shape:", img1.shape)
    print("img2 shape:", img2.shape)
    print("labels shape:", labels.shape)
    print("dtype:", img1.dtype)
    print("unique labels:", torch.unique(labels))

    assert img1.shape == img2.shape, "Image pairs shape mismatch"
    assert img1.shape[1] in (1, 3), "Images should have C=1 or C=3"
    assert labels.ndim == 1, "Labels should be 1D"
    assert set(labels.tolist()).issubset({0.0, 1.0}), "Labels must be 0 or 1"

    print(f"{split_name} sanity passed.")

    # visualize samples
    for i in range(min(3, len(labels))):
        fig, ax = plt.subplots(1, 2)

        if img1.shape[1] == 1:  # grayscale
            ax[0].imshow(img1[i].squeeze().cpu(), cmap="gray")
            ax[1].imshow(img2[i].squeeze().cpu(), cmap="gray")
        else:  # RGB
            ax[0].imshow(img1[i].permute(1, 2, 0).cpu())
            ax[1].imshow(img2[i].permute(1, 2, 0).cpu())

        fig.suptitle(f"{split_name} | Label: {labels[i].item()}")
        plt.show()


def sanity_check_triplet_loader(loader, split_name: str):
    batch = next(iter(loader))
    anchor, positive, negative = batch

    print(f"\n===== {split_name.upper()} TRIPLET SANITY CHECK =====")
    print("anchor shape:", anchor.shape)
    print("positive shape:", positive.shape)
    print("negative shape:", negative.shape)
    print("dtype:", anchor.dtype)

    assert anchor.shape == positive.shape == negative.shape, "Triplet images must have same shape"
    assert anchor.shape[1] in (1, 3), "Images should have C=1 or C=3"

    print(f"{split_name} triplet sanity passed.")

    # visualize samples
    for i in range(min(3, anchor.shape[0])):
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))

        if anchor.shape[1] == 1:  # grayscale
            ax[0].imshow(anchor[i].squeeze().cpu(), cmap="gray")
            ax[1].imshow(positive[i].squeeze().cpu(), cmap="gray")
            ax[2].imshow(negative[i].squeeze().cpu(), cmap="gray")
        else:  # RGB
            ax[0].imshow(anchor[i].permute(1, 2, 0).cpu())
            ax[1].imshow(positive[i].permute(1, 2, 0).cpu())
            ax[2].imshow(negative[i].permute(1, 2, 0).cpu())

        ax[0].set_title("Anchor")
        ax[1].set_title("Positive")
        ax[2].set_title("Negative")

        fig.suptitle(f"{split_name} | Triplet Sample {i}")
        plt.show()