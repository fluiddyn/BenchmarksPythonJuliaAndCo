
import time
import random

import numpy as np

# pythran import numpy as np

from fluidpythran import cachedjit, used_by_cachedjit

from perf import print_perf

## fibonacci ##


@cachedjit
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


## quicksort ##


@cachedjit
def qsort_kernel(a, lo, hi):
    i = lo
    j = hi
    while i < hi:
        pivot = a[(lo + hi) // 2]
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        if lo < j:
            qsort_kernel(a, lo, j)
        lo = i
        j = hi
    return a


## randmatstat ##


@cachedjit
def randmatstat(t):
    n = 5
    randn = np.random.randn
    v = np.zeros(t)
    w = np.zeros(t)
    for i in range(t):
        a = randn(n, n)
        b = randn(n, n)
        c = randn(n, n)
        d = randn(n, n)
        P = np.concatenate((a, b, c, d), axis=1)
        Q = np.concatenate(
            (np.concatenate((a, b), axis=1), np.concatenate((c, d), axis=1)),
            axis=0,
        )
        v[i] = np.trace(np.linalg.matrix_power(P.T @ P, 4))
        w[i] = np.trace(np.linalg.matrix_power(Q.T @ Q, 4))
    return (np.std(v) / np.mean(v), np.std(w) / np.mean(w))


## randmatmul ##


# @cachedjit
def randmatmul(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    return A @ B


## mandelbrot ##

@used_by_cachedjit("mandelperf")
def abs2(z):
    return z.real * z.real + z.imag * z.imag

@used_by_cachedjit("mandelperf")
def mandel(z):
    maxiter = 80
    c = z
    for n in range(maxiter):
        if abs2(z) > 4:
            return n
        z = z * z + c
    return maxiter

@cachedjit
def mandelperf():
    r1 = -2. + 0.1 * np.arange(26)
    r2 = -1. + 0.1 * np.arange(21)
    return [mandel(complex(r, i)) for r in r1 for i in r2]


@cachedjit
def pisum():
    sum = 0.0
    for j in range(1, 501):
        sum = 0.0
        for k in range(1, 10001):
            sum += 1.0 / (k * k)
    return sum


#### Is this single threaded?
# def pisumvec():
#     return numpy.sum(1./(numpy.arange(1,10000)**2))

@cachedjit
def parse_int(t):
    for i in range(1, t):
        n = np.random.randint(0, 2 ** 32 - 1)
        s = hex(n)
        # s = string(n, base = 16)
        # if s[-1] == "L":
        #     s = s[0:-1]
        m = int(s, 16)
        assert m == n
    return n


def printfd(t):
    with open("/dev/null", "w") as file:
        for i in range(1, t):
            file.write(f"{i:d} {i+1:d}\n")


## run tests ##


if __name__ == "__main__":

    mintrials = 5

    assert fib(20) == 6765
    tmin = float("inf")
    for i in range(mintrials):
        t = time.time()
        f = fib(20)
        t = time.time() - t
        if t < tmin:
            tmin = t
    print_perf("recursion_fibonacci", tmin)

    tmin = float("inf")
    for i in range(mintrials):
        t = time.time()
        n = parse_int(1000)
        t = time.time() - t
        if t < tmin:
            tmin = t
    print_perf("parse_integers", tmin)

    assert sum(mandelperf()) == 14791
    tmin = float("inf")
    for i in range(mintrials):
        t = time.time()
        mandelperf()
        t = time.time() - t
        if t < tmin:
            tmin = t
    print_perf("userfunc_mandelbrot", tmin)

    tmin = float("inf")
    for i in range(mintrials):
        lst = [random.random() for i in range(1, 5000)]
        t = time.time()
        qsort_kernel(lst, 0, len(lst) - 1)
        t = time.time() - t
        if t < tmin:
            tmin = t
    print_perf("recursion_quicksort", tmin)

    assert abs(pisum() - 1.644834071848065) < 1e-6
    tmin = float("inf")
    for i in range(mintrials):
        t = time.time()
        pisum()
        t = time.time() - t
        if t < tmin:
            tmin = t
    print_perf("iteration_pi_sum", tmin)

    # assert abs(pisumvec()-1.644834071848065) < 1e-6
    # tmin = float('inf')
    # for i in range(mintrials):
    #     t = time.time()
    #     pisumvec()
    #     t = time.time()-t
    #     if t < tmin: tmin = t
    # print_perf ("pi_sum_vec", tmin)

    (s1, s2) = randmatstat(1000)
    assert s1 > 0.5 and s1 < 1.0
    tmin = float("inf")
    for i in range(mintrials):
        t = time.time()
        randmatstat(1000)
        t = time.time() - t
        if t < tmin:
            tmin = t
    print_perf("matrix_statistics", tmin)

    tmin = float("inf")
    for i in range(mintrials):
        t = time.time()
        C = randmatmul(1000)
        assert C[0, 0] >= 0
        t = time.time() - t
        if t < tmin:
            tmin = t
    print_perf("matrix_multiply", tmin)

    tmin = float("inf")
    for i in range(mintrials):
        t = time.time()
        printfd(100000)
        t = time.time() - t
        if t < tmin:
            tmin = t
    print_perf("print_to_file", tmin)
