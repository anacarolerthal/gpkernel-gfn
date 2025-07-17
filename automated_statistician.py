from utils import KernelFunction, evaluate_likelihood, generate_gp_data
import numpy as np


def greedy_statistician_search(X, Y, method='BIC', max_steps=10):
    """
    Greedily constructs a composite kernel by maximizing GP marginal likelihood using sum, product
    or replacement of base kernels.

    Parameters:
        X (np.ndarray): input data
        Y (np.ndarray): target data
        method (str): 'BIC' for Bayesian Information Criterion, 'LL' for Log Likelihood
        max_steps (int): maximum number of expansions

    Returns:
        KernelFunction: the best composite kernel found
    """
    base_kernels = [
        KernelFunction().rbf(),
        KernelFunction().linear(),
        KernelFunction().periodic(),
        KernelFunction().white_noise(),
        KernelFunction().constant(),
    ]
    _likelihood_cache = {} # cache

    def get_ll(kf):
        key = str(kf)
        if key in _likelihood_cache:
            return _likelihood_cache[key]
        ll = evaluate_likelihood(kf, X, Y, runtime=False)
        _likelihood_cache[key] = ll
        return ll
    
    def get_BIC(kf):
        """
        Calculate Bayesian Information Criterion (BIC) for a given kernel function.
        BIC = -2 * log_likelihood + k * log(n)
        where k is the number of parameters and n is the number of data points.
        """
        n = len(Y)
        k = kf.num_params() +1 
        ll = get_ll(kf)
        return -2 * ll + k * np.log(n)

    def expand_subtrees(k): 
        expansions = []

        # for each base kernel, consider all 3 ops
        for bk in base_kernels:
            if k.name != bk.name:
                expansions.append(bk)
                expansions.append(k.add(bk))
                expansions.append(k.multiply(bk))

        # if current kernel is a sum or product, expand its children (considers all subtrees)
        if k.name in ["Sum", "Product"]:
            for i, child in enumerate(k.children):
                subexpansions = expand_subtrees(child)

                for new_subtree in subexpansions:
                    new_children = k.children[:]
                    new_children[i] = new_subtree
                    new_kernel = KernelFunction(k.name, children=new_children)
                    expansions.append(new_kernel)
        return expansions

    current_kernel = None
    for step in range(max_steps):
        candidates = []

        if current_kernel is None:
            candidates = base_kernels
        else:

            candidates.extend(expand_subtrees(current_kernel))
            
        if method == 'BIC':
            best_bic = np.inf
            candidates = [c for c in candidates if c.num_params() > 0]
            scored = [(get_BIC(c), c) for c in candidates]
            scored = [x for x in scored if np.isfinite(x[0])]
            best_candidate_bic, best_candidate = min(scored, key=lambda x: x[0])
            print(f"[Step {step+1}] BIC: {best_candidate_bic:.2f} LL: {get_ll(best_candidate):.2f} | {best_candidate}")
            
            if best_candidate_bic < best_bic:
                best_bic = best_candidate_bic
                current_kernel = best_candidate
            else:
                print("No improvement found, stopping search.")
                break
        elif method == 'LL':
            best_ll = -np.inf
            scored = [(get_ll(c), c) for c in candidates]
            scored = [x for x in scored if np.isfinite(x[0])]
            best_candidate_ll, best_candidate = max(scored, key=lambda x: x[0])
            print(f"[Step {step+1}] LL: {best_candidate_ll:.2f} BIC: {get_BIC(best_candidate):.2f} | {best_candidate}")
            if best_candidate_ll > best_ll:
               best_ll = best_candidate_ll
               current_kernel = best_candidate
            else:
                print("No improvement found, stopping search.")
                break
        else:
            raise ValueError("Method must be either 'BIC' or 'LL'.")
    return current_kernel