import numpy as np


def l1prox(t, y, b):
    """
    Computes the proximal operator of the t-scaled L1 distance t||y - b|| 
    where b is const. and y is free using the known soft-thresholding 
    formula (without translation -b) and the property of translation
    in 'Proximal Splitting Methods in Signal Processing', Combettes et al.
    
    Args:
        t (float): positive step-size
        y (ndarray): typically k*x for some convolution kernel k and input x
        b (ndarray): blurred and noised image (typically k*x + n for some additive noise n)

    Returns:
        proximal operator of the L1 distance t||y - b|| (ndarray).
    """
    return np.array([
                y[i] + t    if y[i] - b[i] < -t 
        else    y[i] - t    if y[i] - b[i] > t 
        else    b[i]
        for i in range(len(y))
    ])


def l2prox(t, y, b):
    """
    Computes the proximal operator of the t-scaled squared L2 distance t||y - b||^2 
    where b is const. and y is free.
    https://odlgroup.github.io/odl/generated/odl.solvers.nonsmooth.proximal_operators.proximal_l2_squared.html#odl.solvers.nonsmooth.proximal_operators.proximal_l2_squared

    Args:
        t (float): positive step-size
        y (ndarray): typically k*x for some convolution kernel k and input x
        b (ndarray): blurred and noised image (typically k*x + n for some additive noise n)

    Returns:
        proximal operator of the t-scaled squared L2 distance t||y - b||^2 (ndarray).
    
    """
    return (y + 2*t*b) / (1 + 2*t)


def box_prox(t, x):
    """
    Computes the proximal operator t-scaled indicator of 
    {x, a finite dimenional vector: 0 <= x_i <= 1 for all dimenions i}.
    https://odlgroup.github.io/odl/generated/odl.solvers.nonsmooth.proximal_operators.proximal_box_constraint.html#odl.solvers.nonsmooth.proximal_operators.proximal_box_constraint

    Args:
        t (float): positive step-size
        x (ndarray): vector

    Returns:
        proximal operator of the t-scaled squared L2 distance t||y - b||^2 (ndarray).
    
    """
    return np.array([
                0       if x[i] < 0
        else    x[i]    if 0 <= x[i] <= 1
        else    1
        for i in range(len(x))
    ])


def iso_prox(t, g, w1, w2):
    """
    Computes the proximal operator of the tg-scaled iso-norm tg||(w1, w2)||.
    Note that it returns the resulting column vectors seperately.
    
    Args:
        t (float): positive step-size
        g (float): positive const. ("gamma")
        w1 (ndarray): vector
        w2 (ndarray): vector

    Returns:
        proximal operator of tg||(w1, w2)||  (ndarray).
    """
    tg = t * g
    return np.column_stack([
        (1 - tg / max(np.sqrt(w1[i]**2 + w2[i]**2), tg)) * np.array([w1[i], w2[i]])
        for i in range(len(w1))
    ])


if __name__ == "__main__":
    x = np.array([2.0, -0.05, 0.475])
    y = np.array([-2.0, 1.0, 3.0])
    b = np.array([1.0, 5.0, 2.0])
    t = 0.1

    print(f"x := {x}")
    print(f"y := {y}")
    print(f"b := {b}")
    print(f"t := {t}")
    print()
    print(f"l1 prox: {l1prox(y=y, b=b, t=t)}")
    print(f"l2 prox: {l2prox(y=y, b=b, t=t)}")
    print(f"box prox: {box_prox(t=t, x=x)}")
    y1 = np.array([0.0, 0.0, 5.0])
    y2 = np.array([0.0, 0.5, 0.0])
    g = 0.5

    print(f"iso prox: {iso_prox(t=t, g=g, w1=y1, w2=y2)}")