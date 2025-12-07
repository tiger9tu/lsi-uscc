import functools
import time
import numpy as np
from mrh.exploratory.citools import fockspace, grad

# ------------------ Utilities ------------------
def timeit(func=None, *, label=None, _print_fn=print):
    if func is None:
        return lambda f: timeit(f, label=label, _print_fn=_print_fn)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0] if args else None
        verbose = getattr(self, "VERBOSE", 1) if self is not None else 1
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        t1 = time.perf_counter()
        if verbose >= 1:
            _print_fn(f"{label or func.__name__} took {(t1 - t0)*1000:.3f} ms")
        return out
    return wrapper

def print_list_matrix(obj, digits=3, _print_fn=print, _level=0):
    """
    print lists
    """
    indent = "  " * _level  

    try:
        arr = np.asarray(obj)
    except Exception:
        _print_fn(indent + repr(obj))
        return

    if arr.ndim == 0:
        _print_fn(indent + repr(obj))
        return

    if arr.ndim == 1:
        _print_fn(indent + "[")
        for el in obj:
            print_list_matrix(el, digits, _print_fn=_print_fn, _level=_level + 1)
        _print_fn(indent + "]")
        return

    if arr.ndim == 2:
        _print_fn(indent + "[")
        lines = str(np.round(arr, digits)).split("\n")
        for line in lines:
            _print_fn(indent + "  " + line)
        _print_fn(indent + "]")
        return

    _print_fn(indent + "[")
    for el in obj:
        print_list_matrix(el, digits, _print_fn=_print_fn, _level=_level + 1)
    _print_fn(indent + "]")

def cilas2f(lasci, norb_f, nelec_f):
    """Convert LAS CI (per-fragment) to full Fock-space CI."""
    
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f


# def get_sorted_excitations(las, epsilon=0.0, fraction=None, truncated = False, verbose=0):

#     # select excitations based on fraction if provided
#     # if there are multiple excitations with the same gradient magnitude at the cutoff,
#     # all such excitations will be included
#     if fraction is not None:
#         g_all, _, _, _ = grad.get_grad_exact(las, epsilon=0.0)

#         sortg = np.sort(np.abs(g_all))
#         if verbose > 2:
#             print("Total number of excitations:", len(g_all))
#         n = len(sortg)
#         k = int(np.floor(fraction * n))
#         k = min(max(k, 1), n)
#         epsilon = sortg[-k] - 1e-12  # add small buffer to include the k-th element

#     if truncated:
#         all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact_rdm12(las, epsilon)
#     else:
#         all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las, epsilon)
#     gredients = np.array(g_sel)[:,0]
#     sorted_indices = np.argsort(-np.abs(gredients))
#     a_idxs_selected = [a_idxs_selected[i] for i in sorted_indices]
#     i_idxs_selected = [i_idxs_selected[i] for i in sorted_indices]

#     return a_idxs_selected, i_idxs_selected

def get_sorted_excitations(a_idxs, i_idxs, g, epsilon=0.0, fraction = None, verbose=0):
    # select excitations based on fraction if provided
    # if there are multiple excitations with the same gradient magnitude at the cutoff,
    # all such excitations will be included
    if fraction is not None:
        sortg = np.sort(np.abs(g))
        if verbose > 2:
            print("Total number of excitations:", len(g))
        n = len(sortg)
        k = int(np.floor(fraction * n))
        k = min(max(k, 1), n)
        epsilon = sortg[-k] - 1e-12  # add small buffer to include the k-th element

    gredients = np.array(g)
    selected_indices = [idx for idx, grad in enumerate(gredients) if abs(grad) > epsilon]
    selected_gradients = gredients[selected_indices]
    sorted_indices = np.argsort(-np.abs(selected_gradients))

    a_idxs_selected = [a_idxs[selected_indices[i]] for i in sorted_indices]
    i_idxs_selected = [i_idxs[selected_indices[i]] for i in sorted_indices]
    g_selected = [gredients[selected_indices[i]] for i in sorted_indices]

    return a_idxs_selected, i_idxs_selected, g_selected


def lassi_rdm2_to_lasscf(rdm2s_lassi):
    """
    Convert LASSI-style spin-resolved 2-RDM to LAS/LASSCF-style 3-block spin RDM.

    Parameters
    ----------
    rdm2s_lassi : np.ndarray
        LASSI 2-RDM in spin-resolved form.
        - Single-state case: shape (2, ncas, ncas, 2, ncas, ncas)
        - Multi-state case : shape (nroots, 2, ncas, ncas, 2, ncas, ncas)

        Spin index convention assumed:
            0 -> alpha
            1 -> beta

    Returns
    -------
    casdm2s : np.ndarray
        LAS/LASSCF-style 2-RDM with compressed spin blocks (aa, ab, bb).

        - If input is single-state:
            shape (3, ncas, ncas, ncas, ncas)
        - If input is multi-state:
            shape (nroots, 3, ncas, ncas, ncas, ncas)

        Spin block convention:
            0 -> aa
            1 -> ab (symmetric ab = ba)
            2 -> bb
    """
    rdm2s_lassi = np.asarray(rdm2s_lassi)

    if rdm2s_lassi.ndim == 6:
        # Single state: (2, ncas, ncas, 2, ncas, ncas)
        G = rdm2s_lassi
        _, ncas, _, _, _, _ = G.shape

        casdm2 = np.zeros((3, ncas, ncas, ncas, ncas), dtype=G.dtype)

        # aa block: <a_pα† a_rα† a_sα a_qα>
        casdm2[0] = G[0, :, :, 0, :, :]

        # ab block: mixed-spin. Use symmetrized combination of αβ and βα.
        ab_alpha_beta = G[0, :, :, 1, :, :]  # αβ
        ab_beta_alpha = G[1, :, :, 0, :, :]  # βα
        casdm2[1] = 0.5 * (ab_alpha_beta + ab_beta_alpha)

        # bb block: <a_pβ† a_rβ† a_sβ a_qβ>
        casdm2[2] = G[1, :, :, 1, :, :]

        return casdm2

    elif rdm2s_lassi.ndim == 7:
        # Multi-root: (nroots, 2, ncas, ncas, 2, ncas, ncas)
        nroots, _, ncas, _, _, _, _ = rdm2s_lassi.shape
        casdm2s = np.zeros((nroots, 3, ncas, ncas, ncas, ncas),
                           dtype=rdm2s_lassi.dtype)

        for I in range(nroots):
            G = rdm2s_lassi[I]  # shape (2, ncas, ncas, 2, ncas, ncas)

            # aa
            casdm2s[I, 0] = G[0, :, :, 0, :, :]

            # ab (symmetrized)
            ab_alpha_beta = G[0, :, :, 1, :, :]
            ab_beta_alpha = G[1, :, :, 0, :, :]
            casdm2s[I, 1] = 0.5 * (ab_alpha_beta + ab_beta_alpha)

            # bb
            casdm2s[I, 2] = G[1, :, :, 1, :, :]

        return casdm2s

    else:
        raise ValueError(
            "rdm2s_lassi must have shape (2,ncas,ncas,2,ncas,ncas) "
            "or (nroots,2,ncas,ncas,2,ncas,ncas)."
        )
