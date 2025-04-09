import prox_util as prox

# main algorithm logic templates
class SolverTemplate:
    """
    implements main algorithm logic
    using saved state (for faster runtime)
    """
    def __init__(self,
                 initial_iterates,
                 scaling,
                 maxiter,
                 util,
                 b,
                 err_ord,
                 step_size
                 ):
        self.scaling = scaling
        self.initial_iterates = initial_iterates
        self.util = util
        self.maxiter = maxiter
        self.b = b
        self.err_ord=err_ord
        self.step_size = step_size
        
    def douglasrachford_main(self, resolvent_A, resolvent_B, track_objective=False): 
        z1, z2 = self.initial_iterates
        eps = []
        for _ in range(self.maxiter):
            x, y = resolvent_A(z1, z2)
            u, v = resolvent_B(x, y, z1, z2)
            z1 = z1 + self.scaling * (u - x)
            z2 = z2 + self.scaling * (v - y)

            if track_objective:
                eps.append(self.util.get_objective(x=x, b=self.b, ord=self.err_ord))
        
        return self._douglasrachford_out(x, z1, eps)
    
    def _douglasrachford_out(self, x, z1, eps):
        return prox.box_prox(self.step_size, z1), [*eps[:-1], self.util.get_objective(x=x, b=self.b, ord=self.err_ord)]
    
    def admm_main(self, resolvent_A, resolvent_B, final_out, track_objective):
        trec = 1/self.step_size
        x, u, y, w, z = self.initial_iterates
        eps = []

        for _ in range(self.maxiter):
            w1t, z1t = trec*w, trec*z # w/t and z/t used multiple times, just compute once
            x, u = resolvent_A(x, y, u, w1t, z1t)
            Ax, y = resolvent_B(x, y, z1t)
            w = w + self.scaling * (x - u)
            z = z + self.scaling * (Ax - y)

            if track_objective:
                eps.append(self.util.get_objective(x, self.b, ord=self.err_ord))

        return final_out(u, y, trec*w, trec*z), eps
    
    def chambollepock_main(self, prox_g_conj, track_objective=False):
        x, y, z = self.initial_iterates
        eps = []

        for _ in range(self.maxiter):
            y = prox_g_conj(y, z)
            x_up = prox.box_prox(self.step_size, x - self.step_size * self.util.applyAT(y))
            z = 2 * x_up - x
            x = x_up

            if track_objective:
                eps.append(self.util.get_objective(x=x, b=self.b, ord=self.err_ord))

        return x, eps