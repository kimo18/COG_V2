import numpy as np
import matplotlib.pyplot as plt
import wandb

# Simulate data: (B, P, T, K)
B, P, T, K = 2, 4, 6, 3
tensor = np.random.rand(B, P, T, K)
tensor /= tensor.sum(axis=-1, keepdims=True)

# Init wandb
wandb.init(project="annotated-heatmap")

# Log annotated heatmap for all batches and classes
all_images = []

for b in range(B):
    for k in range(K):
        data = tensor[b, :, :, k]  # (P, T)

        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap="viridis", aspect="auto")

        # Annotate each cell with its value
        for i in range(P):
            for j in range(T):
                value = data[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)

        ax.set_title(f"Batch {b} - Class {k}")
        ax.set_xlabel("T (Time)")
        ax.set_ylabel("P (Points)")
        fig.colorbar(im, ax=ax)
        all_images.append(wandb.Image(fig, caption=f"B{b} C{k}"))
        plt.close(fig)

wandb.log({"Annotated Heatmaps": all_images})
