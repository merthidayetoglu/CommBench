import os
import pickle as pkl
import pandas as pd
import numpy as np

def parse_osu(file_path: str) -> pd.DataFrame:
    dtypes = {"size": np.int64, "avg": np.float64, "min": np.float64, "max": np.float64}
    with open(file_path, mode="r") as file:
        for _ in range(4):
            file.readline()
        
        data = [line.split() for line in file]
        df = pd.DataFrame(data, columns=("size", "avg", "min", "max", "iter")).drop(["iter"], axis=1)
        for col in df.columns:
            df[col] = df[col].astype(dtypes[col])

        df = df[np.log2(df["size"]) % 2 == 0].reset_index(drop=True)

    return df

def parse_p2p(folder: str, out_file: str) -> dict:
    dfs: dict[str, pd.DataFrame] = {}

    for comm in os.listdir(folder):
        data = []
        print(comm)
        for file_name in os.listdir(f"{folder}/{comm}"):
            parts = file_name.removesuffix(".out").split(sep='_')
            with open(f"{folder}/{comm}/{file_name}", "r") as file:
                times = [np.float64(line.split(sep=' ')[1]) for line in file]
            data.append([np.int64(parts[2]) * 4, np.int32(parts[0]), np.int32(parts[1]), *times])
        dfs[comm] = pd.DataFrame(data, columns=("size", "source", "dest", "min", "med", "max", "avg"))

    with open(out_file, mode="wb") as file:
        pkl.dump(dfs, file)

def parse_alltoall(folder: str, out_file: str) -> dict:
    dfs: dict[str, pd.DataFrame] = {}

    for comm in os.listdir(folder):
        if comm == "mpi":
            dfs[comm] = parse_osu(f"{folder}/{comm}/osu.out")
            continue
        data = []
        print(comm)
        for file_name in os.listdir(f"{folder}/{comm}"):
            size = np.int64(file_name.removesuffix(".out")) * 4
            with open(f"{folder}/{comm}/{file_name}", "r") as file:
                times = [np.float64(line.split(sep=' ')[1]) for line in file]
            data.append([size, *times])
        dfs[comm] = pd.DataFrame(data, columns=("size", "min", "med", "max", "avg"))

    with open(out_file, mode="wb") as file:
        pkl.dump(dfs, file)

def parse_gather_broadcast(folder: str, out_file: str) -> dict:
    dfs: dict[str, pd.DataFrame] = {}

    for comm in os.listdir(folder):
        if comm == "mpi":
            dfs[comm] = parse_osu(f"{folder}/{comm}/osu.out")
            continue
        print(comm)
        data = []
        for file_name in os.listdir(f"{folder}/{comm}"):
            parts = file_name.removesuffix(".out").split(sep='_')
            with open(f"{folder}/{comm}/{file_name}", "r") as file:
                times = [np.float64(line.split(sep=' ')[1]) for line in file] # min, med, max, avg
            data.append([np.int64(parts[1]) * 4, np.int32(parts[0]), *times])
        dfs[comm] = pd.DataFrame(data, columns=("size", "root", "min", "med", "max", "avg")) # size, root, min, med, max, avg

    # print(dfs)

    with open(out_file, mode="wb") as file:
        pkl.dump(dfs, file)

folder = "data/bidir"
partitions = ["SPX", "TPX", "CPX"]
 
for modality in partitions:
    parse_p2p(f"{folder}/{modality}", f"{folder}/{modality}_data.pkl")
# parse_osu("/pscratch/agatram/gather/SPX/mpi/osu.out")