import jax.numpy as jnp
from jax import grad


    
def l1norm(x):
    """
    Computes l1 norm of a numpy array

    Args:
        x: numpy array

    Returns: 
        l1 norm of x
    """
    return jnp.linalg.norm(x, ord=1)


def l1grad(x):
    return grad(l1norm)(x)


def l1prox(x, t):
    """
    """
    return jnp.array([
        x_i - t if x_i > t \
            else 0 if -t < x < t \
                else x_i + t \
                    for x_i in x
    ])


def l2prox(x, t, y1, b):
    return (
        (y1 + 2*t*b) / (1 + 2*t)
    )


if __name__ == "__main__":
    x = jnp.array([-2.0, 1.0, 3.0])
    my_norm = l1norm(x)
    print(my_norm)


    my_gradient = l1grad(x)
    # 
    print(my_gradient)