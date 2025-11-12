import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

col_names = { "ipc_get": "IPC Get", "ipc_put": "IPC Put", "ipc_get_blit": "IPC Get BLIT Kernel", "ipc_put_blit": "IPC Put BLIT Kernel", "xccl": "XCCL", "mpi": "MPI", "ipc_get_no_sdma": "IPC Get SDMA Disabled",  "ipc_put_no_sdma": "IPC Put SDMA Disabled"}

def heatmap(ax, arr, nbytes, nproc, bandwidth=False):
    ax.imshow(arr, origin="lower", vmin=np.nanmin(arr), vmax=np.nanmax(arr))
    ax.set_xticks(np.arange(nproc))
    ax.set_yticks(np.arange(nproc))

    ax.set_xlabel("Source")
    ax.set_ylabel("Destination")
    ax.set_title(f"{nbytes} bytes")

    # Loop over data dimensions and create text annotations.
    for i in range(nproc):
        for j in range(nproc):
            if i == j:
                continue
            entry = arr[i, j]
            if bandwidth:
                entry_text = f"{entry:.4f}"
            else:
                entry_text = f"{entry:.1f}"
            text = ax.text(j, i, entry_text,
                        ha="center", va="center", color="w", fontsize=6)

def plot_p2p(data_folder, graph_folder, modality):
    cols = ("size", "source", "dest", "min", "med", "max", "avg")
    # modalities = [("SPX", 4, 8), ("TPX", 12, 16), ("CPX", 24, 24)]
    modalities = {"SPX": (4, 8), "TPX": (12, 6), "CPX": (24, 24)}
    sizes_bw = (2 ** 20, 2 ** 24, 2 ** 28, 2 ** 32)
    sizes_lat = (2 ** 8, 2 ** 12, 2 ** 16, 2 ** 20)
    nproc, figsize = modalities[modality]
    with open(f"{data_folder}/{modality}_data.pkl", mode="rb") as file:
        dfs = pkl.load(file)

    os.makedirs(os.path.join(graph_folder, f"{modality}_bandwidth"), exist_ok=True)
    os.makedirs(os.path.join(graph_folder, f"{modality}_latency"), exist_ok=True)

    for col, df_loc in dfs.items():
        groups = df_loc.groupby(["size"])

        fig_bw, axes_bw = plt.subplots(2, 2, figsize=(figsize, figsize), constrained_layout=True)
        for plot_i, size in enumerate(sizes_bw):
            df_size = groups.get_group(size)

            arr_bw = np.empty((nproc, nproc), dtype=np.float64)

            for i, j, val in df_size[["source", "dest", "min"]].itertuples(index=False):
                arr_bw[i][j] = size / val * 1e-3 if i != j else np.nan
            
            ax_bw = axes_bw[plot_i // 2][plot_i % 2]

            heatmap(ax_bw, arr_bw, size, nproc, True)

        fig_lat, axes_lat = plt.subplots(2, 2, figsize=(figsize, figsize), constrained_layout=True)
        for plot_i, size in enumerate(sizes_lat):
            df_size = groups.get_group(size)

            arr_lat = np.empty((nproc, nproc), dtype=np.float64)

            for i, j, val in df_size[["source", "dest", "min"]].itertuples(index=False):
                arr_lat[i][j] = val if i != j else np.nan
            ax_lat = axes_lat[plot_i // 2][plot_i % 2]
            heatmap(ax_lat, arr_lat, size, nproc, False)
        
        fig_bw.suptitle(f"{col_names[col]} Bandwidth in GB/s")
        fig_bw.savefig(f"{graph_folder}/{modality}_bandwidth/{col}.png", dpi=400)
        fig_lat.suptitle(f"{col_names[col]} Latency in µs")
        fig_lat.savefig(f"{graph_folder}/{modality}_latency/{col}.png", dpi=400)

        plt.close(fig_bw)
        plt.close(fig_lat)


def plot_p2p_alt(data_folder, graph_folder, modality):
    cols = ("size", "source", "dest", "min", "med", "max", "avg")
    # modalities = [("SPX", 4, [1, 2]), ("TPX", 12, [1, 2, 8]), ("CPX", 24, [1, 2, 14])]
    modalities = {"SPX": (6, [2]), "TPX": (12, [2, 8]), "CPX": (12, [2, 14])}
    sizes_bw = (2 ** 18, 2 ** 22, 2 ** 26, 2 ** 30)
    sizes_lat = (2 ** 6, 2 ** 10, 2 ** 14, 2 ** 18)
    width, dests = modalities[modality]
    with open(f"{data_folder}/{modality}_data.pkl", mode="rb") as file:
        dfs = pkl.load(file)

    os.makedirs(graph_folder, exist_ok=True)

    fig_lat, axes_lat = plt.subplots(1, len(dests), figsize=(width, 5), constrained_layout=True)
    fig_bw, axes_bw = plt.subplots(1, len(dests), figsize=(width, 5), constrained_layout=True)
    if len(dests) == 1:
        axes_lat = [axes_lat]
        axes_bw = [axes_bw]
    for col, df_loc in dfs.items():
        df_loc = df_loc.sort_values(by=["size"], ignore_index=True)
        groups = df_loc.groupby(["source", "dest"])
        for ax_lat, ax_bw, dest in zip(axes_lat, axes_bw, dests):
            data = groups.get_group((1, dest)).reset_index(drop=True)
            # print(col, dest, data)
            ax_lat.plot(data["size"], data["min"], label=col_names[col])
            ax_bw.plot(data["size"], data["size"] / data["min"] * 1e-3, label=col_names[col])

    for ax_lat, ax_bw, dest in zip(axes_lat, axes_bw, dests):
        ax_lat.set_xscale("log")
        ax_lat.set_yscale("log")
        ax_lat.set_xlabel("Size (bytes)")
        ax_lat.set_ylabel("Latency in µs")
        ax_lat.set_title(f"P2P from 1 to {dest}")
        ax_lat.legend()

        ax_bw.set_xscale("log")
        ax_bw.set_xlabel("Size (bytes)")
        ax_bw.set_ylabel("Bandwidth in GB/s")
        ax_bw.set_title(f"P2P from 1 to {dest}")
        ax_bw.legend()

    fig_lat.suptitle(f"Latency of P2P Connections")
    fig_lat.savefig(f"{graph_folder}/{modality}_latency.png", dpi=400)

    fig_bw.suptitle(f"Bandwidth of P2P Connections")
    fig_bw.savefig(f"{graph_folder}/{modality}_bandwidth.png", dpi=400)

    plt.close(fig_bw)
    plt.close(fig_lat)


def plot_alltoall(data_folder, graph_folder, modality):
    cols = ("size", "min", "med", "max", "avg")
    # modalities = [("SPX", 4), ("TPX", 12), ("CPX", 24)]
    modalities = {"SPX": 4, "TPX": 4, "CPX": 24}
    nproc = modalities[modality]
    with open(f"{data_folder}/{modality}_data.pkl", mode="rb") as file:
        dfs = pkl.load(file)

    os.makedirs(graph_folder, exist_ok=True)

    fig_lat, ax_lat = plt.subplots(constrained_layout=True)
    fig_bw, ax_bw = plt.subplots(constrained_layout=True)
    for col, df_loc in dfs.items():
        df_loc = df_loc.sort_values(by=["size"], ignore_index=True)
        ax_lat.plot(df_loc["size"], df_loc["min"], label=col_names[col])
        ax_bw.plot(df_loc["size"], df_loc["size"] / df_loc["min"] * 1e-3 * nproc * nproc, label=col_names[col])

    ax_lat.set_xscale("log")
    ax_lat.set_yscale("log")
    ax_lat.set_xlabel("Size (bytes)")
    ax_lat.set_ylabel("Latency in µs")
    ax_lat.legend()
    fig_lat.suptitle(f"Latency of All-to-All Collective")
    fig_lat.savefig(f"{graph_folder}/{modality}_latency.png", dpi=400)


    ax_bw.set_xscale("log")
    ax_bw.set_xlabel("Size (bytes)")
    ax_bw.set_ylabel("Bandwidth in GB/s")
    ax_bw.legend()

    fig_bw.suptitle(f"Bandwidth of All-to-All Collective")
    fig_bw.savefig(f"{graph_folder}/{modality}_bandwidth.png", dpi=400)

    plt.close(fig_bw)
    plt.close(fig_lat)


def plot_broadcast_gather(data_folder, graph_folder, modality, gather=False):
    cols = ("size", "root", "min", "med", "max", "avg")
    # modalities = [("SPX", 4), ("TPX", 12), ("CPX", 24)]

    modalities = {"SPX": 4, "TPX": 4, "CPX": 24}
    nproc = modalities[modality]

    title = "Gather" if gather else "Broadcast"

    with open(f"{data_folder}/{modality}_data.pkl", mode="rb") as file:
        dfs = pkl.load(file)

    os.makedirs(graph_folder, exist_ok=True)

    fig_lat, ax_lat = plt.subplots(constrained_layout=True)
    fig_bw, ax_bw = plt.subplots(constrained_layout=True)

    bins, ref_ind = max([(df_loc["size"].nunique(), name) for name, df_loc in dfs.items()])
    sizes = dfs[ref_ind]["size"].astype(np.uint64).unique()
    sizes.sort()
    groups = len(dfs)

    for i, (col, df_loc) in enumerate(dfs.items()):
        # need to create bin labels and scatter in that area
        df_loc["size"] = df_loc["size"].astype(np.uint64)
        df_loc = df_loc.sort_values(by=["size"], ignore_index=True)
        pos = np.searchsorted(sizes, df_loc["size"]) * (groups * 2 + 1) + i + groups / 2
        ax_lat.scatter(pos, df_loc["min"], label=col_names[col], s=15.0, marker="x", alpha=0.75, linewidths=1.0)
        ax_bw.scatter(pos, df_loc["size"] / df_loc["min"] * 1e-3 * nproc, label=col_names[col], s=15.0, marker="x", alpha=0.75, linewidths=1.0)

    for i in range(bins):
        ax_lat.axvline((i + 1) * (2 * groups + 1) - 1, color="gray", linestyle="--", linewidth=0.5)
        ax_bw.axvline((i + 1) * (2 * groups + 1) - 1, color="gray", linestyle="--", linewidth=0.5)
    ax_lat.set_xticks(np.arange(0, bins) * (groups * 2 + 1) + groups, sizes, rotation=-60.0, size="small")
    ax_bw.set_xticks(np.arange(0, bins) * (groups * 2 + 1) + groups, sizes, rotation=-60.0, size="small")

    ax_lat.set_yscale("log")
    ax_lat.set_xlabel("Size (bytes)")
    ax_lat.set_ylabel("Latency in µs")
    ax_lat.legend()
    fig_lat.suptitle(f"Latency of {title} Collective")
    fig_lat.savefig(f"{graph_folder}/{modality}_latency.png", dpi=400)


    # ax_bw.set_xscale("log")
    ax_bw.set_xlabel("Size (bytes)")
    ax_bw.set_ylabel("Bandwidth in GB/s")
    ax_bw.legend()

    fig_bw.suptitle(f"Bandwidth of {title} Collective")
    fig_bw.savefig(f"{graph_folder}/{modality}_bandwidth.png", dpi=400)

    plt.close(fig_bw)
    plt.close(fig_lat)

# plot_alltoall("/pscratch/agatram/alltoall-6.4.1", "vis_new/alltoall-6.4.1", "SPX")
plot_p2p("data/p2p", "vis/p2p", "SPX")
# plot_broadcast_gather("/pscratch/agatram/gather-6.4.1", "vis_new/gather-6.4.1", "SPX", True)
# plot_broadcast_gather("data/broadcast", "vis/broadcast", "SPX", False)
plot_p2p_alt("data/p2p", "vis/p2p", "SPX")