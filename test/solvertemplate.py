import prox_util as prox
import periodic_mat_util as mat
# main algorithm logic templates
class SolverTemplate:
    """
    implements main algorithm logic
    using saved state (for faster runtime)
    """
    def __init__(self,
                #  initial_iterates,
                 scaling,
                 maxiter,
                #  b,
                 err_ord,
                 step_size,
                 util
                 ):
        self.scaling = scaling
        # self.initial_iterates = initial_iterates
        self.maxiter = maxiter
        # self.b = b
        self.err_ord=err_ord
        self.step_size = step_size
        self.util = util
        
    def douglasrachford_main(self, b, resolvent_A, resolvent_B,
                             get_obj, if_track=False): 
        z1, z2 = b, self.util.applyA(b)
        eps = []
        for _ in range(self.maxiter):
            x, y = resolvent_A(z1, z2, b)
            u, v = resolvent_B(x, y, z1, z2)
            z1 = z1 + self.scaling * (u - x)
            z2 = z2 + self.scaling * (v - y)

            if if_track:
                eps.append(get_obj(x,b,ord=self.err_ord))
        
        return self._douglasrachford_out(x, z1, b,eps, get_obj)
    
    def _douglasrachford_out(self, x, z1, b, eps, get_obj):
        return prox.box_prox(self.step_size, z1), [*eps[:-1], get_obj(x,b,ord=self.err_ord)]
    
    def admm_main(self, b, resolvent_A, resolvent_B, final_out,
                  get_obj, if_track=False):
        like_b = b.copy()
        like_b_triple_cat = mat.cat_mats([b.copy() for _ in range(3)])
        trec = 1/self.step_size
        x, u, y, w, z = like_b, like_b, like_b_triple_cat, like_b, like_b_triple_cat
        eps = []

        for _ in range(self.maxiter):
            w1t, z1t = trec*w, trec*z # w/t and z/t used multiple times, just compute once
            x, u = resolvent_A(x, y, u, w1t, z1t)
            Ax, y = resolvent_B(x, y, z1t, b)
            w = w + self.scaling * (x - u)
            z = z + self.scaling * (Ax - y)

            if if_track:
                eps.append(get_obj(x,b,ord=self.err_ord))

        return final_out(u, y, trec*w, trec*z), eps
    
    def chambollepock_main(self, b, prox_f,  prox_g_conj,
                           get_obj, if_track=False):
        x, y, z = b.copy(), self.util.applyA(b), b.copy()
        eps = []

        for _ in range(self.maxiter):
            y = prox_g_conj(y, z, b)
            x_up = prox_f(x, y)
            z = 2 * x_up - x
            x = x_up

            if if_track:
                eps.append(get_obj(x,b,ord=self.err_ord))

        return x, eps