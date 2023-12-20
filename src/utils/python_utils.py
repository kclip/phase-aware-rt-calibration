from typing import Sized, Generator, Iterable, List, Union

import numpy as np
import pandas as pd


def batch(iterable: Sized, batch_size: int = None) -> Generator:
    if batch_size is None:
        yield iterable
    else:
        n = len(iterable)
        for ndx in range(0, n, batch_size):
            yield iterable[ndx:min(ndx + batch_size, n)]


def get_closest_value_index(
    lookup_array: np.ndarray,  # Shape [N_LOOKUP]
    queried_entries: Iterable  # Shape [N_ENTRIES]
) -> List[int]:  # Shape [N_ENTRIES] ; Closest value indexes
    lookup_array_ = np.asarray(lookup_array)
    indexes = []
    for entry in queried_entries:
        error = np.abs(lookup_array_ - entry)
        idx = np.argmin(error)
        indexes.append(int(idx))
    return indexes


def rad2deg(x: Union[float, np.ndarray, pd.Series, pd.DataFrame]) -> Union[float, np.ndarray, pd.Series, pd.DataFrame]:
    return x * (180 / np.pi)


def deg2rad(x: Union[float, np.ndarray, pd.Series, pd.DataFrame]) -> Union[float, np.ndarray, pd.Series, pd.DataFrame]:
    return x * (np.pi / 180)
