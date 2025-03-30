

class DouglasRachfordPrimal():
    def __init__(self,
                 objective='l1',
                 maxiter=500,
                 relax=1.0,
                 step_size=0.1,
                 gamma=0.5
                 ):
        self.objective = objective
        self.maxiter = maxiter
        self.relax = relax
        self.step_size = step_size
        self.gamma = gamma

    def solve(self, kernel, blurred_image):
        pass