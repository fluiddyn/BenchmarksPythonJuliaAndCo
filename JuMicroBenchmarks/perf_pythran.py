import numpy as np

from transonic import jit, wait_for_all_extensions

from perf_py import (
    print_perf,
    fib,
    qsort_kernel,
    matrix_statistics_rand,
    matrix_multiply_randomgen,
    matrix_statistics_ones,
    matrix_statistics_randomgen,
    matrix_multiply_rand,
    matrix_multiply_ones,
    bench_random,
    bench_random_randomgen,
    pisum,
    pisum_vec,
    parse_int,
    parse_int_rand,
    parse_int_randomgen,
    printfd,
    # abs2,
    # mandel,
    mandelperf,
    mandelperf2,
    broadcast,
    broadcast_inplace,
)

args_jit = dict(native=True, xsimd=True)

fib = jit(fib, **args_jit)

qsort_kernel = jit(qsort_kernel, **args_jit)

matrix_statistics_rand = jit(matrix_statistics_rand, **args_jit)
matrix_statistics_ones = jit(matrix_statistics_ones, **args_jit)

matrix_multiply_rand = jit(matrix_multiply_rand, **args_jit)
matrix_multiply_ones = jit(matrix_multiply_ones, **args_jit)
broadcast = jit(broadcast, **args_jit)
broadcast_inplace = jit(broadcast_inplace, **args_jit)
bench_random = jit(bench_random, **args_jit)

pisum = jit(pisum, **args_jit)
pisum_vec = jit(pisum_vec, **args_jit)

mandelperf = jit(mandelperf, **args_jit)

mandelperf2 = jit(mandelperf2, **args_jit)

parse_int = jit(parse_int, **args_jit)
parse_int_rand = jit(parse_int_rand, **args_jit)

# Pythran does not support format and f-strings :-(
# printfd = jit(printfd)


def warmup():
    fib(2)
    parse_int(np.ones(2, dtype=np.uint32))
    parse_int_rand(2)
    mandelperf()
    mandelperf2()
    lst = np.random.random(50)
    qsort_kernel(lst, 0, len(lst) - 1)
    pisum()
    pisum_vec()
    matrix_statistics_rand(10)
    matrix_statistics_ones(10)
    matrix_multiply_rand(10)
    matrix_multiply_ones(10)
    a = np.ones((10, 10))
    broadcast(a)
    broadcast_inplace(a)
    bench_random(10)
    printfd(2)


if __name__ == "__main__":

    import sys
    from time import time

    if "warmup" in sys.argv:
        warmup()
        print("wait_for_all_extensions")
        wait_for_all_extensions()
        sys.exit()

    mintrials = 10

    assert fib(20) == 6765
    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        f = fib(20)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("recursion_fibonacci", tmin)

    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        n = parse_int_rand(1000)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("parse_integers_rand", tmin)

    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        n = parse_int_randomgen(1000)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("parse_integers_randomgen", tmin)

    tmin = float("inf")
    for i in range(mintrials):
        numbers = np.random.randint(0, 2 ** 32 - 1, 1000).astype(np.uint32)
        assert numbers.size == 1000
        t = time()
        n = parse_int(numbers)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("parse_integers", tmin)

    assert sum(mandelperf()) == 14791
    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        mandelperf()
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("mandelbrot", tmin)

    assert sum(mandelperf2()) == 14791
    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        mandelperf2()
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("mandelbrot2", tmin)

    tmin = float("inf")
    for i in range(mintrials):
        lst = np.random.random(5000)
        t = time()
        qsort_kernel(lst, 0, len(lst) - 1)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("recursion_quicksort", tmin)

    assert abs(pisum() - 1.644_834_071_848_065) < 1e-6
    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        pisum()
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("pisum", tmin)

    assert abs(pisum_vec() - 1.644_834_071_848_065) < 1e-6
    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        pisum_vec()
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("pisum_vec", tmin)

    (s1, s2) = matrix_statistics_rand(1000)
    assert s1 > 0.5 and s1 < 1.0
    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        matrix_statistics_rand(1000)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("matrix_statistics_rand_numpy", tmin)

    (s1, s2) = matrix_statistics_randomgen(1000)
    assert s1 > 0.5 and s1 < 1.0
    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        matrix_statistics_randomgen(1000)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("matrix_statistics_rand", tmin)

    (s1, s2) = matrix_statistics_ones(1000)
    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        matrix_statistics_ones(1000)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("matrix_statistics_ones", tmin)

    mintrials = 10

    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        C = matrix_multiply_rand(1000)
        assert C[0, 0] >= 0
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("matrix_multiply_rand_numpy", tmin)

    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        C = matrix_multiply_ones(1000)
        assert C[0, 0] >= 0
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("matrix_multiply_ones", tmin)

    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        C = matrix_multiply_randomgen(1000)
        assert C[0, 0] >= 0
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("matrix_multiply_rand", tmin)

    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        C = bench_random(1000)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("random_numpy", tmin)

    mintrials = 100
    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        C = bench_random_randomgen(1000)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("random", tmin)

    tmin = float("inf")
    a = np.ones((1000, 1000))
    for i in range(mintrials):
        t = time()
        broadcast(a)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("broadcast", tmin)

    mintrials = 10
    tmin = float("inf")
    for i in range(mintrials):
        a = np.ones((1000, 1000))
        t = time()
        broadcast_inplace(a)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("broadcast_inplace", tmin)

    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        printfd(100_000)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("print_to_file", tmin)
