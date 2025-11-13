import time
import functools
import mrh
def timeit(func=None, *, label=None, logger=print):
    """Decorator to time a function. Use as @timeit or @timeit(label='name')."""
    if func is None:
        return lambda f: timeit(f, label=label, logger=logger)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        logger(f"{label or func.__name__} took {(t1 - t0) * 1000:.3f} ms")
        return result

    return wrapper

@timeit
def giid(N):
    # example workload
    s = 0
    for i in range(N):
        s += i
    return s

if __name__ == "__main__":
    timeit(label="giid execution")(giid)(1000000)