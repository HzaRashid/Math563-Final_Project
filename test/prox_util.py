import numpy as np


def l1prox(t, y, b):
    """
    Computes the proximal operator of the t-scaled L1 distance t||y - b|| 
    where b is const. and y is free using the known soft-thresholding 
    formula (without translation -b) and the property of translation
    in 'Proximal Splitting Methods in Signal Processing', Combettes et al.
    
    Args:
        t (float): positive step-size
        y (np.ndarray): typically k*x for some convolution kernel k and input x
        b (np.ndarray): blurred and noised image (typically k*x + n for some additive noise n)

    Returns:
        proximal operator of the L1 distance t||y - b|| (np.ndarray).
    """
    diff = y - b
    return  np.where(diff < -t, y + t,  # y_ij + t if y_ij - b_ij < -t 
            np.where(diff >  t, y - t,  # y_ij - t if y_ij - b_ij > t
                     b                  # b_ij, else
                    ))


def l2prox(t, y, b):
    """
    Computes the proximal operator of the t-scaled squared L2 distance t||y - b||^2 
    where b is const. and y is free.

    Args:
        t (float): positive step-size
        y (np.ndarray): typically k*x for some convolution kernel k and input x
        b (np.ndarray): blurred and noised image (typically k*x + n for some additive noise n)

    Returns:
        proximal operator of the t-scaled squared L2 distance t||y - b||^2 (np.ndarray).
    
    """
    return (y + 2*t*b) / (1 + 2*t)


def box_prox(t, x):
    """
    Computes the proximal operator t-scaled indicator of 
    {x, a finite dimenional vector: 0 <= x_i <= 1 for all dimenions i}.

    Args:
        t (float): positive step-size
        x (np.ndarray): vector

    Returns:
        proximal operator of the t-scaled squared L2 distance t||y - b||^2 (np.ndarray).
    
    """
    return  np.where(x < 0, 0,  # 0 if x_ij = 0
            np.where(x > 1, 1,  # 1 if x_ij > 1
                     x          # x_ij, else
                     ))       


def iso_prox(t, g, w1, w2):
    """
    Computes the proximal operator of the tg-scaled iso-norm tg||(w1, w2)||.
    Note that it returns the resulting column vectors seperately as rows.
    
    Args:
        t (float): positive step-size
        g (float): positive const. ("gamma")
        w1 (np.ndarray): vector
        w2 (np.ndarray): vector

    Returns:
        proximal operator of tg||(w1, w2)||  (np.ndarray).
    """
    tg = t * g
    # we can replace the condition with 
    # a maximum resulting in the desired 0 elements
    # according to the formula
    return (1 - tg / np.maximum(np.sqrt(w1**2 + w2**2), tg)) * np.array([w1, w2])


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
    isoprox = iso_prox(t=t, g=g, w1=y1, w2=y2)
    print(f"iso prox: {isoprox}")

    print(isoprox.shape)