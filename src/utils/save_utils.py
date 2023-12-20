import json
import os
from json import JSONEncoder
from typing import Union
from dataclasses import dataclass, asdict, fields
import numpy as np
import tensorflow as tf


# File Management
# ---------------

def create_folder_if_not_exists(folder: str, raise_if_exists: bool = False):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    else:
        if raise_if_exists:
            raise ValueError(f"Folder '{folder}' already exists !")


class SafeOpen(object):
    def __init__(
        self,
        folder,
        filename,
        open_mode,
        raise_if_folder_exists: bool = False,
        overwrite: bool = False,
    ):
        self.folder = folder
        self.filepath = os.path.join(folder, filename)
        self.open_mode = open_mode
        self.raise_if_folder_exists = raise_if_folder_exists
        self.overwrite = overwrite

    def __enter__(self):
        is_writing = self.open_mode in ("w", "wb")

        # Create folder if it does not exist when writing
        if is_writing and (not os.path.isdir(self.folder)):
            os.makedirs(self.folder)
        else:
            if self.raise_if_folder_exists:
                raise ValueError(f"Folder '{self.folder}' already exists !")

        # Check if file already exists when writing
        if is_writing and (not self.overwrite) and os.path.isfile(self.filepath):
            raise FileExistsError(
                f"File '{self.filepath}' already exists... Please enable overwriting if you wish to proceed"
            )

        # Open file
        self.file = open(self.filepath, self.open_mode)
        return self.file

    def __exit__(self, *args):
        self.file.close()


# JSON
# ----

def _ndarray_encoding(array: Union[np.ndarray, tf.Tensor, tf.Variable]) -> dict:
    # Handle tensorflow Tensors
    from_tensor = False
    if isinstance(array, (tf.Tensor, tf.Variable)):
        array = array.numpy()
        from_tensor = True

    # Handle complex arrays
    is_complex = False
    array_data = dict(__ndarray__=array.real.tolist())
    if np.iscomplexobj(array):
        is_complex = True
        array_data["__ndarray_imag__"] = array.imag.tolist()

    return dict(
        **array_data,
        dtype=str(array.real.dtype),  # Keep only real part dtype
        to_tensor=from_tensor,
        is_complex=is_complex
    )


class JSONCustomEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.complexfloating):
            return complex(obj)
        elif isinstance(obj, (np.ndarray, tf.Tensor, tf.Variable)):
            return _ndarray_encoding(obj)
        return JSONEncoder.default(self, obj)


def json_array_obj_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        array = np.array(dct['__ndarray__'], dtype=dct['dtype'])
        if dct['is_complex']:
            array_imag = np.array(dct['__ndarray_imag__'], dtype=dct['dtype'])
            array = array + (1j * array_imag)
        if dct['to_tensor']:
            array = tf.convert_to_tensor(array)
        return array
    return dct


# Store/Load Data
# ---------------

@dataclass()
class StorableDataclass(object):
    __filename__ = None

    @staticmethod
    def filename(filename):
        return f"{filename}.json"

    @staticmethod
    def arrays_filename(filename):
        return f"{filename}__arrays.npz"

    def store(self, directory, overwrite: bool = False):
        data_dict = asdict(self)
        # Store arrays in a separate (compressed) file
        arrays = {}
        for field in fields(self):
            if field.type in (np.ndarray, tf.Tensor, tf.Variable):
                is_tensor = field.type in (tf.Tensor, tf.Variable)
                arrays[field.name] = data_dict[field.name].numpy() if is_tensor else np.copy(data_dict[field.name])
                data_dict[field.name] = dict(
                    __file__=self.arrays_filename(self.__filename__),
                    __key__=field.name,
                    is_tensor=is_tensor
                )
        if len(arrays) > 0:
            with SafeOpen(directory, self.arrays_filename(self.__filename__), "wb", overwrite=overwrite) as file:
                np.savez_compressed(file, **arrays)
        # Store JSON with non-array data and refs to separate array file
        with SafeOpen(directory, self.filename(self.__filename__), "w", overwrite=overwrite) as file:
            json.dump(data_dict, file, cls=JSONCustomEncoder)

    @classmethod
    def load(cls, directory):
        # Load json data
        with SafeOpen(directory, cls.filename(cls.__filename__), "r") as file:
            json_data = json.load(file, object_hook=json_array_obj_hook)
        # Check if dataclass has an arrays file
        has_arrays_file = False
        for field_data in json_data.values():
            if isinstance(field_data, dict) and ("__file__" in field_data):
                has_arrays_file = True
                break
        # Load and format dataclass data (json and arrays)
        if has_arrays_file:
            data = dict()
            with SafeOpen(directory, cls.arrays_filename(cls.__filename__), "rb") as file:
                arrays_data = np.load(file)
                for k, v in json_data.items():
                    if isinstance(v, dict) and ("__file__" in v):
                        data[k] = arrays_data[k]
                        if v["is_tensor"]:
                            data[k] = tf.convert_to_tensor(data[k])
                    else:
                        data[k] = json_data[k]
        else:
            data = json_data
        # Return dataclass
        return cls(**data)
