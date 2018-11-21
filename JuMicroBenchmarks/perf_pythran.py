from time import time

import numpy as np

# pythran import numpy as np

from fluidpythran import cachedjit, used_by_cachedjit

from perf import (
    print_perf,
    fib,
    qsort_kernel,
    matrix_statistics,
    matrix_statistics_ones,
    matrix_multiply,
    matrix_multiply_ones,
    pisum,
    pisum_vec,
    parse_int,
    printfd,
    abs2,
    mandel,
    mandelperf,
)

from fib import fib as fib_pythran


fib = cachedjit(fib)
qsort_kernel = cachedjit(qsort_kernel)

matrix_statistics = cachedjit(matrix_statistics)
matrix_statistics_ones = cachedjit(matrix_statistics_ones)

matrix_multiply = cachedjit(matrix_multiply)
matrix_multiply_ones = cachedjit(matrix_multiply_ones)

pisum = cachedjit(pisum)
pisum_vec = cachedjit(pisum_vec)

used_by_cachedjit("mandelperf")(abs2)
used_by_cachedjit("mandelperf")(mandel)
mandelperf = cachedjit(mandelperf)

parse_int = cachedjit(parse_int)

# Pythran does not support format and f-strings
# printfd = cachedjit(printfd)


if __name__ == "__main__":

    mintrials = 10

    n = 20
    assert fib(n) == 6765
    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        f = fib(n)
        t = time() - t
        if t < tmin:
            tmin = t
    print("fib with fluidpythran:")
    print_perf("recursion_fibonacci", tmin)

    assert fib_pythran(n) == 6765
    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        f = fib_pythran(n)
        t = time() - t
        if t < tmin:
            tmin = t
    print("Now fib without fluidpythran:")
    print_perf("recursion_fibonacci_pythran", tmin)

    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        n = parse_int(1000)
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

    (s1, s2) = matrix_statistics(1000)
    assert s1 > 0.5 and s1 < 1.0
    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        matrix_statistics(1000)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("matrix_statistics", tmin)

    (s1, s2) = matrix_statistics_ones(1000)
    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        matrix_statistics_ones(1000)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("matrix_statistics_ones", tmin)

    tmin = float("inf")
    for i in range(mintrials):
        t = time()
        C = matrix_multiply(1000)
        assert C[0, 0] >= 0
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("matrix_multiply", tmin)

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
        printfd(100_000)
        t = time() - t
        if t < tmin:
            tmin = t
    print_perf("print_to_file", tmin)
