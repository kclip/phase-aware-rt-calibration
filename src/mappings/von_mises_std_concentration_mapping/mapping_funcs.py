import os
from typing import Union

import numpy as np
import pandas as pd

from settings import PROJECT_FOLDER
from src.utils.save_utils import SafeOpen
from src.utils.python_utils import get_closest_value_index, deg2rad


VON_MISES_STD_CONCENTRATION_MAPPING_FOLDER = os.path.join(
    PROJECT_FOLDER,
    "src/mappings/von_mises_std_concentration_mapping/"
)
VON_MISES_STD_CONCENTRATION_ARRAY_FILENAME = "mapping_array.npy"


def _map_concentration_std_von_mises(
    from_value: Union[float, np.ndarray, pd.Series],
    from_concentration_to_std: bool,
    to_value_round: int = None,
    map_infinite_concentration: bool = False,
    print_errors: bool = False
) -> Union[float, np.ndarray, pd.Series]:
    if from_concentration_to_std:
        from_idx = 0
        to_idx = 1
    else:
        from_idx = 1
        to_idx = 0

    if isinstance(from_value, float):
        queried_values = np.array([from_value], dtype=np.float32)
    elif isinstance(from_value, pd.Series):
        queried_values = from_value.to_numpy()
    else:  # np.ndarray
        queried_values = from_value

    with SafeOpen(
        VON_MISES_STD_CONCENTRATION_MAPPING_FOLDER,
        VON_MISES_STD_CONCENTRATION_ARRAY_FILENAME,
        "rb"
    ) as file:
        vm_std_concentration_array = np.load(file)

    indexes_map = get_closest_value_index(
        lookup_array=vm_std_concentration_array[:, from_idx],
        queried_entries=queried_values
    )
    to_value = vm_std_concentration_array[indexes_map, to_idx]

    if to_value_round is not None:
        to_value = np.round(to_value, to_value_round)

    if (not from_concentration_to_std) and map_infinite_concentration:
        to_value = to_value.astype("object")
        to_value[to_value == np.max(vm_std_concentration_array[:, to_idx])] = "_infinity"

    if print_errors:
        errors = np.abs(vm_std_concentration_array[indexes_map, from_idx] - queried_values)
        if len(errors) > 0:
            print(
                f"Mapping error: median {np.quantile(errors, 0.5):.3f} | "
                f"10% quantile {np.quantile(errors, 0.9):.3f} | "
                f"max {np.max(errors):.3f}"
            )

    if isinstance(from_value, float):
        return to_value[0]
    elif isinstance(from_value, pd.Series):
        return pd.Series(to_value, index=from_value.index)
    else:  # np.ndarray
        return to_value


def map_von_mises_concentration_to_std(
    concentrations: Union[float, np.ndarray, pd.Series],
    round_std: int = None,
    print_errors: bool = False
) -> Union[float, np.ndarray, pd.Series]:
    return _map_concentration_std_von_mises(
        from_value=concentrations,
        from_concentration_to_std=True,
        to_value_round=round_std,
        map_infinite_concentration=False,
        print_errors=print_errors
    )


def map_von_mises_std_to_concentration(
    std: Union[float, np.ndarray, pd.Series],
    std_in_degrees: bool = False,
    round_concentration: int = None,
    map_infinite_concentration: bool = False,
    print_errors: bool = False
) -> Union[float, np.ndarray, pd.Series]:
    return _map_concentration_std_von_mises(
        from_value=deg2rad(std) if std_in_degrees else std,
        from_concentration_to_std=False,
        to_value_round=round_concentration,
        map_infinite_concentration=map_infinite_concentration,
        print_errors=print_errors
    )
