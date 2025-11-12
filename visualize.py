import matplotlib.pyplot as plt
import numpy as np
import os

def get_data(folder: str, gpus: int, size: int) -> np.ndarray:
    arr = np.empty((gpus, gpus))

    for file_name in os.listdir(folder):
        parts = file_name.removesuffix(".out").split(sep='_')
        if parts[2] == str(size):
            with open(f"{folder}/{file_name}", "r") as file:
                min_time = file.readline().split(sep=' ')[1]
            arr[int(parts[0])][int(parts[1])] = min_time

    return arr

nproc = 12
fig, axes = plt.subplots(2, 2, figsize=(24, 24), constrained_layout=True)
folder = "data_aws_xccl"
lib = "xccl"
lib_title = "XCCL"

for i in range(4):
    ax = axes[i // 2][i % 2]
    nbytes = 2 ** (8 * (i + 1) - 2)
    arr = get_data(f"{folder}/{lib}", nproc, nbytes) 

    ax.imshow(arr, origin="lower")
    ax.set_xticks(np.arange(nproc))
    ax.set_yticks(np.arange(nproc))

    ax.set_xlabel("Source")
    ax.set_ylabel("Destination")
    ax.set_title(f"{nbytes * 4} bytes")

    # Loop over data dimensions and create text annotations.
    for i in range(nproc):
        for j in range(nproc):
            entry = arr[i, j]
            text = ax.text(j, i, f"{entry:.1f}",
                        ha="center", va="center", color="w", fontsize=6)

fig.suptitle(f"{lib_title} Latency")
fig.savefig(f"vis/{lib}.png", dpi=400)
