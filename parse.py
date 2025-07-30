import os
import pickle as pkl
import pandas as pd
import numpy as np

folder = "data_CPX_all"
out_file = "CPX.pkl"
df = pd.DataFrame({});

for comm in os.listdir(folder):
    data = []
    for file_name in os.listdir(f"{folder}/{comm}"):
        parts = file_name.removesuffix(".out").split(sep='_')
        with open(f"{folder}/{comm}/{file_name}", "r") as file:
            times = [np.float64(line.split(sep=' ')[1]) for line in file] # min, med, max, avg
        data.append((np.int64(parts[2]), np.int32(parts[0]), np.int32(parts[1]), *times))
    df[comm] = data # size, source, dest, min, med, max, avg

print(df) # df.explode() ?

with open(out_file, mode="wb") as file:
    pkl.dump(df, file)