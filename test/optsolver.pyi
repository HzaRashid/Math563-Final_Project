# type annotations for optsolver.py

class OptSolver:
    def __init__(self, k: int, b: int,
                 deblurring_objective: str = 'l1',
                 maxiter: int = 500,
                 relax: float = 0.5,
                 step_size: float = 0.1,
                 gamma: float = 0.1) -> None: ...


class DouglasRachfordPrimal(OptSolver):
    def __init__(self, k: int, b: int,
                 deblurring_objective: str = 'l1',
                 maxiter: int = 500,
                 relax: float = 0.5,
                 step_size: float = 0.1,
                 gamma: float = 0.1) -> None: ...
    

class DouglasRachfordDual(OptSolver):
    def __init__(self, k: int, b: int,
                 deblurring_objective: str = 'l1',
                 maxiter: int = 500,
                 relax: float = 0.5,
                 step_size: float = 0.1,
                 gamma: float = 0.1) -> None: ...
    

class ADMM(OptSolver):
    def __init__(self, k: int, b: int,
                 deblurring_objective: str = 'l1',
                 maxiter: int = 500,
                 relax: float = 0.5,
                 step_size: float = 0.1,
                 gamma: float = 0.1) -> None: ...


class ChambollePock(OptSolver):
    def __init__(self, k: int, b: int,
                 deblurring_objective: str = 'l1',
                 maxiter: int = 500,
                 relax: float = 0.5,
                 step_size: float = 0.1,
                 step_size2: float = 0.1,
                 gamma: float = 0.1) -> None: ...