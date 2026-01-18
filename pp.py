import numpy as np
import pandas as pd


# import numpy file and count the rows
def count_embeddings(np_file: str) -> int:
    data = np.load(np_file, allow_pickle=True).item()
    return len(data)
print(count_embeddings("vector_search/sostanze_2008_145/old_embeddings_dict_2008_145.npy"))
