
# Avoid unstable gradient at point (permittivity=1.0, conductivity=0.0)
_EPSILON_MATERIAL = 1e-7


def check_material(mat):
    """We need to make sure that material properties are always valid"""
    if mat.conductivity < 0:  # Non-negative conductivity
        mat.conductivity.assign(0 + _EPSILON_MATERIAL)
    if mat.relative_permittivity < 1:  # Relative permittivity not smaller than 1
        mat.relative_permittivity.assign(1 + _EPSILON_MATERIAL)
