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

def print_matrix(obj, digits=3, _print_fn=print):
    try:
        arr = np.asarray(obj)
    except Exception:
        _print_fn(repr(obj))
        return

    if arr.ndim == 0:
        _print_fn(format(arr.item(), f'.{digits}g'))
        return
    if arr.ndim > 2:
        _print_fn(repr(obj))
        return

    def fmt(x):
        if isinstance(x, (str, bytes)):
            return str(x)
        try:
            return format(x, f'.{digits}g')
        except Exception:
            return str(x)

    if arr.ndim == 1:
        strs = [fmt(x) for x in arr]
        # Use individual widths (no vertical alignment needed for single row),
        # but keep a single space between columns
        row = '[ ' + ' '.join(s for s in strs) + ' ]'
        _print_fn(row)
        return

    # 2D case: compute string repr and column widths for alignment
    rows_str = [[fmt(x) for x in row] for row in arr]
    ncols = arr.shape[1]
    col_widths = [max(len(rows_str[r][c]) for r in range(arr.shape[0])) for c in range(ncols)]

    padded_rows = []
    for r in range(arr.shape[0]):
        padded = [rows_str[r][c].rjust(col_widths[c]) for c in range(ncols)]
        padded_rows.append('[ ' + ' '.join(padded) + ' ]')

    _print_fn('[\n ' + '\n '.join(padded_rows) + '\n]')


def print_list_matrix(obj, digits=3, _print_fn = print):
    try:
        arr = np.asarray(obj)
    except Exception:
        _print_fn(repr(obj))
        return

    # If it's a 2D array-like, delegate to print_matrix
    if arr.ndim <= 2:
        print_matrix(obj, digits, _print_fn=_print_fn)
        return

    # If 1D iterable, recurse on elements
    else:
        for el in obj:
            print_list_matrix(el, digits, _print_fn=_print_fn)
        return

def cilas2f(lasci, norb_f, nelec_f):
    """Convert LAS CI (per-fragment) to full Fock-space CI."""
    
    ci_f = []
    for i, ci in enumerate(lasci):
        ci_f.append(fockspace.hilbert2fock(ci, norb_f[i], nelec_f[i])[0])
    return ci_f


def get_sorted_excitations(las, epsilon=0.0, fraction=None, verbose=0):

    # select excitations based on fraction if provided
    # if there are multiple excitations with the same gradient magnitude at the cutoff,
    # all such excitations will be included
    if fraction is not None:
        g_all, _, _, _ = grad.get_grad_exact(las, epsilon=0.0)

        sortg = np.sort(np.abs(g_all))
        if verbose > 2:
            print("Total number of excitations:", len(g_all))
        n = len(sortg)
        k = int(np.floor(fraction * n))
        k = min(max(k, 1), n)
        epsilon = sortg[-k] - 1e-12  # add small buffer to include the k-th element

    all_g, g_sel, a_idxs_selected, i_idxs_selected = grad.get_grad_exact(las, epsilon)
    gredients = np.array(g_sel)[:,0]
    sorted_indices = np.argsort(-np.abs(gredients))
    a_idxs_selected = [a_idxs_selected[i] for i in sorted_indices]
    i_idxs_selected = [i_idxs_selected[i] for i in sorted_indices]

    return a_idxs_selected, i_idxs_selected
        

