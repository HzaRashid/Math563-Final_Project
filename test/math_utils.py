import numpy as np


def l1prox(t, y, b):
    """
    Computes the proximal operator of the t-scaled L1 distance t||y - b|| 
    where b is const. and y is free using the known soft-thresholding 
    formula (without translation -b) and the property of translation
    in 'Proximal Splitting Methods in Signal Processing', Combettes et al.
    

    Args:
        t (float): step-size
        y (ndarray): typically k*x for some convolution kernel k
        b (ndarray): blurred and noised image (typically k*x + n for some additive noise n)

    Returns:
        proximal operator of the L1 distance t||y - b|| (ndarray).
    """
    return np.array([
                y[i] + t if y[i] - b[i] < -t 
        else    y[i] - t if y[i] - b[i] > t 
        else    b[i]
        for i in range(len(y))
    ])


def l2prox(t, y, b):
    """
    Computes the proximal operator of the t-scaled squared L2 distance t||y - b||^2 
    where b is const. and y is free.
    https://odlgroup.github.io/odl/generated/odl.solvers.nonsmooth.proximal_operators.proximal_l2_squared.html#odl.solvers.nonsmooth.proximal_operators.proximal_l2_squared

    Args:
        t (float): step-size
        y (ndarray): typically k*x for some convolution kernel k
        b (ndarray): blurred and noised image (typically k*x + n for some additive noise n)

    Returns:
        proximal operator of the t-scaled squared L2 distance t||y - b||^2 (ndarray).
    
    """
    return (y + 2*t*b) / (1 + 2*t)


if __name__ == "__main__":
    y = np.array([-2.0, 1.0, 3.0])
    b = np.array([1.0, 5.0, 2.0])
    t = 0.1
    print(f"y := {y}")
    print(f"b := {b}")
    print(f"t := {t}")
    print()
    print(f"l1 prox: {l1prox(y=y, b=b, t=t)}")
    print(f"l2 prox: {l2prox(y=y, b=b, t=t)}")