import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open("SPX.pkl", mode="rb") as file:
    df: pd.DataFrame = pkl.load(file)

cols = ("size", "source", "dest", "min", "med", "max", "avg")
sizes = (2 ** 18, 2 ** 22, 2 ** 26, 2 ** 30)
nproc = 4

col_names = { "ipc_get": "IPC Get", "ipc_put": "IPC Put", "ipc_get_blit": "IPC Get BLIT Kernel", "ipc_put_blit": "IPC Put BLIT Kernel", "xccl": "XCCL", "mpi": "MPI" }

for col in df.columns:
    df_loc = pd.DataFrame([[*row] for row in df[col].values], columns=cols)
    groups = df_loc.groupby(["size"])
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    for plot_i, size in enumerate(sizes):
        df_size = groups.get_group(size)
        # print(df_size)
        arr = np.empty((nproc, nproc), dtype=np.float64)
        nbytes = size * 4
        # df_size.apply(lambda row: arr[row["source"]][row["dest"]] = nbytes / row["min"] * 1e-9)
        for i, j, val in df_size[["source", "dest", "min"]].itertuples(index=False):
            arr[i][j] = nbytes / val * 1e-3
        
        ax = axes[plot_i // 2][plot_i % 2]

        ax.imshow(arr, origin="lower")
        ax.set_xticks(np.arange(nproc))
        ax.set_yticks(np.arange(nproc))

        ax.set_xlabel("Source")
        ax.set_ylabel("Destination")
        ax.set_title(f"{nbytes} bytes")

        # Loop over data dimensions and create text annotations.
        for i in range(nproc):
            for j in range(nproc):
                entry = arr[i, j]
                text = ax.text(j, i, f"{entry:.4f}",
                            ha="center", va="center", color="w", fontsize=6)
    
    fig.suptitle(f"{col_names[col]} Bandwidth in GB/s")
    fig.savefig(f"vis_SPX_bandwidth/{col}.png", dpi=400)
        
