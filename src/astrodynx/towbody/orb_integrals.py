import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array


def ang_momentum(r: ArrayLike, v: ArrayLike) -> Array:
    r"""
    Returns the angular momentum vector of a two-body system.

    Parameters
    ----------
    r : (3,) ArrayLike
        Position vector of the object.
    v : (3,) ArrayLike
        Velocity vector of the object.

    Returns
    -------
    out : (3,) Array
        Angular momentum vector of the object.

    Notes
    -----
    The angular momentum vector is calculated using the cross product of the position and velocity vectors:
    $$
    \mathbf{h} = \mathbf{r} \times \mathbf{v}
    $$
    where $\mathbf{h}$ is the angular momentum vector, $\mathbf{r}$ is the position vector, $\mathbf{v}$ is the velocity vector.

    The result is a 3D vector representing the angular momentum of the object in the two-body system.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astrodynx.towbody.orb_integrals import ang_momentum
    >>> r = jnp.array([1.0, 0.0, 0.0])
    >>> v = jnp.array([0.0, 1.0, 0.0])
    >>> ang_momentum(r, v)
    Array([0., 0., 1.])
    """
    return jnp.cross(r, v)
