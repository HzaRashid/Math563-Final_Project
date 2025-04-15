import numpy as np
import prox_util as prox
from optutil import OptUtil
import periodic_mat_util as mat
from numpy.typing import NDArray
from typing import Callable, Tuple, List

# types for resolvent methods taking 3, 4, and 5 arguments
Resolvent3Type = Callable[[NDArray, NDArray, NDArray], Tuple[NDArray, NDArray]]
Resolvent4Type = Callable[[NDArray, NDArray, NDArray, NDArray], Tuple[NDArray, NDArray]]
Resolvent5Type = Callable[[NDArray, NDArray, NDArray, NDArray, NDArray], Tuple[NDArray, NDArray]]
GetObjType = Callable[[NDArray, NDArray, int], float] # type for get_objective, see optsolver.py

# main algorithm logic templates
class SolverTemplate:
    """
    A template for iterative solvers used in image deblurring algorithms.

    This class implements core algorithm logic using stored states for
    better runtime. It provides common algorithmic templates for solving 
    deblurring problems using different splitting methods, such as the 
    Douglas-Rachford, ADMM, and Chambolle-Pock algorithms.
    """
    def __init__(self,
                 scaling: float,
                 maxiter: int,
                 err_ord: int,
                 step_size: float,
                 util: OptUtil
                 ):
        """
        Initialize the solver template with algorithm-specific parameters.

        Parameters:
            scaling (float): The relaxation/scaling factor used to update iterates.
            maxiter (int): The maximum number of iterations to run.
            err_ord (int): The order of the norm (e.g., 1 for L1, 2 for L2) used to 
                           compute the objective error.
            step_size (float): Step size parameter used for the proximal operators.
            util (OptUtil): Utility object that provides matrix transforms and other 
                            stateful operations needed in the algorithm.
        """
        self.scaling = scaling
        self.maxiter = maxiter
        self.err_ord=err_ord
        self.step_size = step_size
        self.util = util
        
    def douglasrachford_main(self, 
                             b: NDArray, 
                             resolvent_A: Resolvent3Type, 
                             resolvent_B: Resolvent4Type,
                             get_obj: GetObjType, 
                             if_track: bool = False,
                             stop_criterion: float = -1.0) -> Tuple[NDArray, List[float], int]:
        """
        Solve the deblurring problem using the Douglas-Rachford splitting algorithm.

        Parameters:
            b (NDArray): The blurred image.
            resolvent_A (Resolvant3Type): A function implementing the resolvent of the
                                          first operator; expected to accept (z1, z2, b)
                                          and return a tuple (x, y).
            resolvent_B (Resolvant4Type): A function implementing the resolvent of the 
                                          second operator; expected to accept (x, y, z1, z2)
                                          and return a tuple (u, v).
            get_obj (GetObjType): A function that computes the deblurring objective value.
                                  It should accept two NDArray inputs and an integer norm order.
            if_track (bool, optional): If True, tracks the objective error over iterations.
                                       Defaults to False.
            stop_criterion (float, optional): If larger than 0, stops the algorithm when 
                                              get_obj(...) < stop_criterion.

        Returns:
            Tuple[NDArray, List[float], int]: A tuple containing:
                - The final deblurred image (after applying the box proximal operator).
                - A list of objective error values tracked during the iterations.
                - The iteration at which early stopping was applied, if applicable, otherwise maxiter.
        """
        z1, z2 = b, self.util.applyA(b) # initial iterates
        self._tracking_init(stop_criterion) # objective tracking where applicable

        for iter in range(self.maxiter):
            x, y = resolvent_A(z1, z2, b)
            u, v = resolvent_B(x, y, z1, z2)
            z1 = z1 + self.scaling * (u - x)
            z2 = z2 + self.scaling * (v - y)

            if self._iter_check(x, b, iter, if_track, get_obj, stop_criterion):
                break
        
        return self._douglasrachford_out(x, z1, b, self.eps, get_obj, self.stop_iter)
    
    def _douglasrachford_out(self, 
                             x: NDArray, 
                             z1: NDArray, 
                             b: NDArray, 
                             eps: List[float], 
                             get_obj: GetObjType,
                             iter: int) -> Tuple[NDArray, List[float], int]:
        """
        Finalize and return the output of the Douglas-Rachford algorithm.

        Applies the box proximal operator to refine the final iterate and computes 
        the final objective error.

        Parameters:
            x (NDArray): The last computed iterate before projection.
            z1 (NDArray): The intermediate variable from the iteration.
            b (NDArray): The original blurred image.
            eps (List[float]): The list of objective errors collected over iterations.
            get_obj (GetObjType): The objective function used to compute error values.
            iter (int): Iteration at which algorithm 'solved' the problem.

        Returns:
            Tuple[NDArray, List[float], int]: A tuple containing:
                - The final projected solution after applying prox.box_prox.
                - The complete list of objective errors with the last error value computed.
                - The iteration at which early stopping was applied, if applicable, otherwise maxiter.
        """
        return prox.box_prox(self.step_size, z1), [*eps[:-1], get_obj(x,b,ord=self.err_ord)], iter
    
    def admm_main(self, 
                  b: NDArray, 
                  resolvent_A: Resolvent5Type,
                  resolvent_B: Resolvent4Type,
                  final_out: Callable[[NDArray, NDArray, NDArray, NDArray], NDArray],
                  get_obj: GetObjType,
                  if_track: bool = False,
                  stop_criterion: float = -1.0
                  ) -> Tuple[NDArray, List[float], int]:
        """
        Solve the deblurring problem using the ADMM (Alternating Direction Method of Multipliers) algorithm.

        Parameters:
            b (NDArray): The blurred image.
            resolvent_A (Resolvant5Type): A function implementing the resolvent of the first operator.
                                          Expected to take five NDArray arguments and return a tuple (x, u).
            resolvent_B (Resolvant4Type): A function implementing the resolvent of the second operator.
                                          Expected to take four NDArray arguments and return a tuple (Ax, y).
            final_out (Callable): A function to combine the intermediate variables into the final solution.
                                  Expected to take four NDArray arguments and return a single NDArray.
            get_obj (GetObjType): A function that computes the deblurring objective value.
            if_track (bool, optional): If True, tracks the objective error over iterations.
                                       Defaults to False.
            stop_criterion (float, optional): If larger than 0, stops the algorithm when 
                                              get_obj(...) < stop_criterion.

        Returns:
            Tuple[NDArray, List[float], int]: A tuple containing:
                - The final deblurred image obtained by composing the intermediate results.
                - A list of objective errors tracked over the iterations.
                - The iteration at which early stopping was applied, if applicable, otherwise maxiter.
        """
        # initial iterates
        like_b = b.copy()
        like_b_triple_cat = mat.cat_mats([b.copy() for _ in range(3)])
        trec = 1/self.step_size
        x, u, y, w, z = like_b, like_b, like_b_triple_cat, like_b, like_b_triple_cat
        # initialize early stopping states and objective tracking where applicable
        self._tracking_init(stop_criterion)

        for iter in range(self.maxiter):
            w1t, z1t = trec*w, trec*z # w/t and z/t used multiple times, just compute once
            x, u = resolvent_A(x, y, u, w1t, z1t)
            Ax, y = resolvent_B(x, y, z1t, b)
            w = w + self.scaling * (x - u)
            z = z + self.scaling * (Ax - y)

            if self._iter_check(x, b, iter, if_track, get_obj, stop_criterion):
                break

        return final_out(u, y, trec*w, trec*z), [*self.eps[:-1], get_obj(x,b,ord=self.err_ord)], self.stop_iter
    

    
    def chambollepock_main(self, 
                           b: NDArray, 
                           prox_f: Callable[[NDArray, NDArray], NDArray],  
                           prox_g_conj: Callable[[NDArray, NDArray, NDArray], NDArray],  
                           get_obj: Callable[[NDArray, NDArray, int], float], 
                           if_track: bool = False,
                           stop_criterion: float = -1.0
                           ) -> Tuple[NDArray, List[float], int]:
        """
        Solve the deblurring problem using the Chambolle-Pock primal-dual algorithm.

        Parameters:
            b (NDArray): The blurred image.
            prox_f (Callable): A proximal operator corresponding to the primal function f.
                               Expected to take an NDArray (x) and an NDArray (y) and return a refined NDArray.
            prox_g_conj (Callable): A proximal operator for the conjugate of function g. It should accept
                                    three NDArrays (y, z, b) and return an updated NDArray.
            get_obj (Callable): A function to compute the deblurring objective error. It should accept two NDArrays and an int.
            if_track (bool, optional): If True, tracks the objective error over the iterations.
                                       Defaults to False.
            stop_criterion (float, optional): If larger than 0, stops the algorithm when 
                                              get_obj(...) < stop_criterion.

        Returns:
            Tuple[NDArray, List[float], int]: A tuple containing:
                - The final deblurred image after iterations.
                - A list of tracked objective error values.
                - The iteration at which early stopping was applied, if applicable, otherwise maxiter.
        """
        x, y, z = b.copy(), self.util.applyA(b), b.copy()
        self._tracking_init(stop_criterion)

        for iter in range(self.maxiter):
            y = prox_g_conj(y, z, b)
            x_up = prox_f(x, y)
            z = 2 * x_up - x
            x = x_up

            if self._iter_check(x, b, iter, if_track, get_obj, stop_criterion):
                break

        return x, [*self.eps[:-1], get_obj(x,b,ord=self.err_ord)], self.stop_iter
    

    def _tracking_init(self, stop_criterion):
        """
        """
        self.check_stop = stop_criterion > 0 # whether early stopping is applicable
        self.stop_iter = self.maxiter # stores when early stopping is applied
        self.eps = [] # collects objective values when if_track==True, eps[i] = objective value at iteration i
        self.cur_eps = None # store current iteration's objective
    
    def _iter_check(self, x, b, iter, if_track, get_obj, stop_criterion):
        self.cur_eps = (get_obj(x, b, ord=self.err_ord) 
                        if self.check_stop or if_track 
                        else self.cur_eps)

        if self.check_stop and self.cur_eps < stop_criterion: # stop criterion met
            self.stop_iter = iter # iteration where criterion was met
            return True
        
        if if_track:
            self.eps.append(self.cur_eps)

        return False