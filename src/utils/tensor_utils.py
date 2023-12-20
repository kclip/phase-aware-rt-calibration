from typing import Tuple, List, Union
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


# Casting
# -------

def cast_to_complex(tensor: tf.Tensor) -> tf.Tensor:
    return tf.complex(tensor, tf.zeros(tensor.shape, dtype=tensor.dtype))


def cast_to_pure_imag(tensor: tf.Tensor) -> tf.Tensor:
    return tf.complex(tf.zeros(tensor.shape, dtype=tensor.dtype), tensor)


# Masked reductions
# -----------------

def reduce_masked_min(
    tensor: tf.Tensor,  # Numerical ; Shape [dim1, ..., dimN]
    mask: tf.Tensor,  # Boolean ; Shape [dim1, ..., dimN]
    axis: Union[int, List[int]]
) -> tf.Tensor:
    # Masked reduction
    min_values = tf.reduce_min(
        tf.where(mask, tensor, tensor.dtype.max),
        axis=axis
    )
    # Reduction should have at least one input value, otherwise return NaN
    min_mask = tf.reduce_any(mask, axis=axis)

    return tf.where(min_mask, min_values, np.nan)


def reduce_masked_max(
    tensor: tf.Tensor,  # Numerical ; Shape [dim1, ..., dimN]
    mask: tf.Tensor,  # Boolean ; Shape [dim1, ..., dimN]
    axis: Union[int, List[int]]
) -> tf.Tensor:
    # Masked reduction
    max_values = tf.reduce_max(
        tf.where(mask, tensor, tensor.dtype.min),
        axis=axis
    )
    # Reduction should have at least one input value, otherwise return NaN
    max_mask = tf.reduce_any(mask, axis=axis)

    return tf.where(max_mask, max_values, np.nan)


def reduce_masked_mean(
    tensor: tf.Tensor,  # Numerical ; Shape [dim1, ..., dimN]
    mask: tf.Tensor,  # Boolean ; Shape [dim1, ..., dimN]
    axis: int
) -> tf.Tensor:
    zero = tf.constant(0, dtype=tensor.dtype)
    sum_tensor = tf.reduce_sum(
        tf.where(mask, tensor, zero),
        axis=axis
    )
    count_mask = tf.reduce_sum(
        tf.cast(mask, dtype=tensor.dtype),
        axis=axis
    )
    return sum_tensor / count_mask  # Will be NaN for reductions without input values


# Tensor manipulation
# -------------------

def hermitian(tensor: tf.Tensor) -> tf.Tensor:
    """Conjugate transpose on the two last dimensions of a tensor"""
    n = len(tensor.shape)
    return tf.transpose(tensor, perm=[*range(n - 2), n - 1, n - 2], conjugate=True)


def tile_newdim(tensor: tf.Tensor, tile_dim: int, tile_n: int) -> tf.Tensor:
    """Repeat tensor across a newly generated dimension"""
    return tf.repeat(
        tf.expand_dims(tensor, axis=tile_dim),
        repeats=tile_n,
        axis=tile_dim
    )


def cartesian_product(
    a: tf.Tensor,  # Shape [dimA1, ..., dimAN]
    b: tf.Tensor  # Shape [dimB1, ..., dimBN]
) -> tf.Tensor:  # Shape [dimA1 * dimB1 * ... * dimAN * dimBN, 2]
    """
    Cartesian product of two tensors with same dimensions.
    Note: shape of input tensors is not preserved.
    """
    cartesian_prod = tf.stack(tf.meshgrid(a, b, indexing='ij'), axis=-1)
    return tf.reshape(cartesian_prod, [-1, 2])


def squared_norm(tensor: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """Squared 2-norm on the <axis> dimension of a tensor"""
    return tf.reduce_sum(
        tf.math.real(tf.math.conj(tensor) * tensor),
        axis=axis
    )


def count_non_null(tensor: tf.Tensor, axis: int) -> int:
    return tf.reduce_max(
        tf.math.count_nonzero(tensor, axis=axis)
    ).numpy()


def dot(
    x: tf.Tensor,  # Shape [dim1, ..., dimN]
    y: tf.Tensor,  # Shape [dim1, ..., dimN]
    axis: Union[int, List[int]] = -1
) -> tf.Tensor:  # Shape [dim1, ..., dim<axis - 1>, dim<axis + 1>, ..., dimN]
    """
    Sum reduce the product of two tensors along <axis>.
    Note: dimension broadcasting is applied to any given dimension of <x> and <y> with different multiplicity, as long
    as the multiplicity in one of these tensors is 1.
    """
    return tf.reduce_sum(x * y, axis=axis)


def reduce_mean_ignore_null(tensor: tf.Tensor, axis: Union[int, List[int]]) -> tf.Tensor:
    """Computes mean along <axis>, counting non-null elements only"""
    n_non_null = tf.math.count_nonzero(tensor, axis=axis)
    n_non_null = tf.cast(n_non_null, tensor.dtype)
    tensor_sum = tf.reduce_sum(tensor, axis=axis)
    return tensor_sum / n_non_null


# Random tensors
# --------------

def sample_circular_uniform_tensor(
    shape: Tuple[int], seed: float = None
) -> tf.Tensor:
    phases = tf.random.uniform(
        shape=shape,
        minval=0.0,
        maxval=2 * np.pi,
        dtype=tf.float32,
        seed=seed
    )
    return tf.math.exp(cast_to_pure_imag(phases))


def sample_circular_von_mises_tensor(
    shape: Tuple[int],
    mean: float,
    concentration: float,
    seed: float = None
) -> tf.Tensor:
    if concentration < 0:
        raise ValueError("Concentration parameter must be greater than 0")

    dist = tfp.distributions.VonMises(loc=mean, concentration=concentration)
    phases = dist.sample(shape, seed=seed)
    return tf.math.exp(cast_to_pure_imag(phases))


def sample_complex_standard_normal_tensor(
    shape: Tuple[int],
    std: float,
    seed: float = None
) -> tf.Tensor:
    return tf.complex(
        tf.random.normal(shape, mean=0.0, stddev=std / np.sqrt(2), dtype=tf.dtypes.float32, seed=seed),
        tf.random.normal(shape, mean=0.0, stddev=std / np.sqrt(2), dtype=tf.dtypes.float32, seed=seed)
    )


def sample_uniform_unitary_cartesian_coordinates_tensor(
    shape: Tuple[int],
    seed: float = None
) -> tf.Tensor:  # Shape [*shape, 3]
    azimuth = tf.random.uniform(
        shape,
        minval=0,
        maxval=2*np.pi,
        dtype=tf.dtypes.float32,
        seed=seed
    )
    z_coord = tf.random.uniform(
        shape,
        minval=-1,
        maxval=1,
        dtype=tf.dtypes.float32,
        seed=seed
    )
    norm_xy = tf.math.sqrt(1 - tf.math.pow(z_coord, 2))
    x_coord = norm_xy * tf.math.cos(azimuth)
    y_coord = norm_xy * tf.math.sin(azimuth)
    return tf.stack([x_coord, y_coord, z_coord], axis=-1)


# Bessel functions
# ----------------

def compute_log_bessel_i0(
    x: tf.Tensor,
    use_tfp: bool = False
) -> tf.Tensor:
    """Compute bessel ratio log(I_0(x)) in a numerically stable manner"""
    if use_tfp:
        return tfp.math.log_bessel_ive(0, x)
    else:
        return tf.math.log(tf.math.bessel_i0e(x)) + x


def compute_bessel_ratio(
    x: tf.Tensor,
    use_tfp: bool = False
) -> tf.Tensor:
    """Compute bessel ratio (I_1(x) / I_0(x)) in a numerically stable manner"""
    if use_tfp:
        return tfp.math.bessel_iv_ratio(1, x)
    else:
        return tf.math.bessel_i1e(x) / tf.math.bessel_i0e(x)


def compute_bessel_ratio_inverse(r: tf.float32) -> tf.float32:
    """
    Compute the approximate inverse R^-1(r) of the bessel ratio r = R(kappa) = I_1(kappa) / I_0(kappa) using Taylor's
    approximation of the first kind modified bessel functions I_0 and I_1.
    Expressions are taken from: https://dl.acm.org/doi/10.1145/355945.355949
    Results are correct up to 0.003 around kappa in [2.5 ; 3.1], bellow 1e-6 for kappa < 1 and kappa > 10.
    """
    if r < 0 or r >= 1:  # No solution
        return None
    elif r < 0.8:  # Taylor's expansion approximation for "small" r (r <= 3)
        return (
            (1 / (1 - tf.pow(r, 2))) *
            (
                (2 * r) -
                tf.pow(r, 3) -
                (tf.pow(r, 5) / 6) -
                (tf.pow(r, 7) / 24) +
                (tf.pow(r, 9) / 360) +
                (tf.pow(r, 11) * (53 / 2160))
            )
        )
    else:  # Taylor's expansion approximation for "large" r (r > 3)
        tmp_y = 2 / (1 - r)
        if r < 0.95:
            tmp_x = 2001.035224 + (4317.5526 * r) - (2326 * np.power(r, 2))
        else:
            tmp_x = 32 / (tmp_y - 131.5 + (120 * r))
        return 0.25 * (
            tmp_y + 1 +
            (
                3 / (
                    tmp_y - 5 -
                    (
                        12 / (
                            tmp_y - 10 - tmp_x
                        )
                    )
                )
            )
        )


# Geometry
# --------

def get_mean_angles(
    tensor_angles: tf.Tensor,  # in [rad] ; Shape [dim1, ..., dimN]
    mask: tf.Tensor = None,  # mask of values to count in <tensor_angles> ; Shape [dim1, ..., dimN]
    axis: Union[int, List[int]] = None
) -> tf.Tensor:  # Mean angles in [-pi, pi]
    if mask is not None:
        _mask = mask
    else:
        _mask = tf.cast(tf.ones(tensor_angles.shape), dtype=tf.bool)

    mean_cos = reduce_masked_mean(
        tensor=tf.math.cos(tensor_angles),
        mask=_mask,
        axis=axis
    )
    mean_sin = reduce_masked_mean(
        tensor=tf.math.sin(tensor_angles),
        mask=_mask,
        axis=axis
    )

    return tf.math.atan2(mean_sin, mean_cos)


def angles_to_unit_vec(
    elevation: tf.Tensor,  # Elevation angles [rad], any shape [dim1, ..., dimN]
    azimuth: tf.Tensor  # Azimuth angles [rad], any shape [dim1, ..., dimN]
) -> tf.Tensor:  # Shape [dim1, ..., dimN, 3]
    """From angular spherical coordinates to unitary vector in cartesian space"""
    return tf.stack([
        tf.sin(elevation) * tf.cos(azimuth),
        tf.sin(elevation) * tf.sin(azimuth),
        tf.cos(elevation)
    ], axis=-1)


def angle_in_minus_pi_plus_pi(angles: tf.Tensor) -> tf.Tensor:
    """Represent angles in the [-pi, pi] segment"""
    return tf.math.mod(angles + np.pi, 2*np.pi) - np.pi
