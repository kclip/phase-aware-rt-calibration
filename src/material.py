from dataclasses import dataclass
from typing import Tuple
import numpy as np

from const import FREE_SPACE_PERMITTIVITY


@dataclass()
class Material(object):
    a: np.float32
    b: np.float32
    c: np.float32
    d: np.float32

    # Conductivity in S/m
    def conductivity(
        self,
        frequency: np.float32  # In Hz
    ) -> np.float32:
        frequency_ghz = frequency / np.float32(1e9)
        return self.c * np.power(frequency_ghz, self.d)

    # Real part of relative permittivity
    def real_part_permittivity(
        self,
        frequency: np.float32  # In 1/s
    ) -> np.float32:
        frequency_ghz = frequency / np.float32(1e9)
        return self.a * np.power(frequency_ghz, self.b)

    # Complex part of relative permittivity
    def complex_part_permittivity(
        self,
        frequency: np.float32  # In Hz
    ) -> np.float32:
        return self.conductivity(frequency) / (2 * np.pi * FREE_SPACE_PERMITTIVITY * frequency)

    # Complex relative permittivity
    def permittivity(
        self,
        frequency: np.float32  # In Hz
    ) -> np.complex64:
        return np.complex64(self.real_part_permittivity(frequency) + 1j * self.complex_part_permittivity(frequency))

    # Sionna RT custom radio material callback
    def material_callback(self, frequency: np.float32) -> Tuple[np.float32, np.float32]:
        return (
            self.real_part_permittivity(frequency),
            self.conductivity(frequency)
        )


# Pre-defined materials
CONCRETE = Material(a=np.float32(5.31), b=np.float32(0), c=np.float32(0.0326), d=np.float32(0.8095))
BRICK = Material(a=np.float32(3.91), b=np.float32(0), c=np.float32(0.0238), d=np.float32(0.16))
WOOD = Material(a=np.float32(1.99), b=np.float32(0), c=np.float32(0.0047), d=np.float32(1.0718))
GLASS = Material(a=np.float32(6.31), b=np.float32(0), c=np.float32(0.0036), d=np.float32(1.3394))
METAL = Material(a=np.float32(1), b=np.float32(0), c=np.float32(10**7), d=np.float32(0))
