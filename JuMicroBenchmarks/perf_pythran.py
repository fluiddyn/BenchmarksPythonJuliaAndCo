from time import time

import numpy as np

from transonic import jit, wait_for_all_extensions

from randomgen import RandomGenerator

rnd = RandomGenerator()


@jit
def fib(n: int):
    """fibonacci"""
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


@jit
def qsort_kernel(a, lo, hi):
    """quicksort"""
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


@jit
def matrix_statistics_rand(t):
    n = 5
    randn = np.random.randn
    matrix_power = np.linalg.matrix_power
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
        v[i] = np.trace(matrix_power(P.T @ P, 4))
        w[i] = np.trace(matrix_power(Q.T @ Q, 4))
    return (np.std(v) / np.mean(v), np.std(w) / np.mean(w))


def matrix_statistics_randomgen(t):
    n = 5
    randn = rnd.randn
    matrix_power = np.linalg.matrix_power
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
        v[i] = np.trace(matrix_power(P.T @ P, 4))
        w[i] = np.trace(matrix_power(Q.T @ Q, 4))
    return (np.std(v) / np.mean(v), np.std(w) / np.mean(w))


@jit
def matrix_statistics_ones(t):
    n = 5
    matrix_power = np.linalg.matrix_power
    v = np.zeros(t)
    w = np.zeros(t)
    for i in range(t):
        a = b = c = d = np.ones((n, n))
        P = np.concatenate((a, b, c, d), axis=1)
        Q = np.concatenate(
            (np.concatenate((a, b), axis=1), np.concatenate((c, d), axis=1)),
            axis=0,
        )
        v[i] = np.trace(matrix_power(P.T @ P, 4))
        w[i] = np.trace(matrix_power(Q.T @ Q, 4))
    return (np.std(v) / np.mean(v), np.std(w) / np.mean(w))


@jit
def matrix_multiply_rand(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    return A @ B


@jit
def matrix_multiply_ones(n):
    A = np.ones((n, n))
    B = np.ones((n, n))
    return A @ B


def matrix_multiply_randomgen(n):
    A = rnd.rand(n, n)
    B = rnd.rand(n, n)
    return A @ B


@jit
def bench_random(n: int):
    return np.random.rand(n, n)


def bench_random_randomgen(n: int):
    return rnd.rand(n, n)


@jit
def broadcast(a):
    return 10 * (2 * a ** 2 + 4 * a ** 3) + 2 / a


@jit
def broadcast_inplace(a):
    a[:] = 10 * (2 * a ** 2 + 4 * a ** 3) + 2 / a


## mandelbrot ##


def abs2(z):
    return z.real * z.real + z.imag * z.imag


def mandel(z):
    maxiter = 80
    c = z
    for n in range(maxiter):
        if abs2(z) > 4:
            return n
        z = z * z + c
    return maxiter


@jit
def mandelperf():
    r1 = [-2.0 + 0.1 * i for i in range(26)]
    r2 = [-1.0 + 0.1 * i for i in range(21)]
    return [mandel(complex(r, i)) for r in r1 for i in r2]


@jit
def mandelperf2():
    r1 = -2.0 + 0.1 * np.arange(26)
    r2 = -1.0 + 0.1 * np.arange(21)
    result = np.empty(r1.size * r2.size)
    ind = 0
    for r in r1:
        for i in r2:
            result[ind] = mandel(complex(r, i))
            ind += 1

    return result


@jit
def pisum():
    sum = 0.0
    n = 500
    for j in range(n):
        for k in range(1, 10001):
            sum += 1.0 / (k * k)
    return sum / n


@jit
def pisum_vec():
    n = 500
    s = 0.0
    a = np.arange(1, 10001)
    for j in range(500):
        s += np.sum(1.0 / (a ** 2))
    return s / n


@jit
def parse_int_rand(t):
    numbers = np.random.randint(0, 2 ** 32 - 1, t)
    for n in numbers:
        m = int(hex(n), 16)
        assert m == n
    return n


@jit
def parse_int(numbers):
    for n in numbers:
        m = int(hex(n), 16)
        assert m == n
    return n


def parse_int_randomgen(t):
    numbers = rnd.random_uintegers(t, bits=32)
    for n in numbers:
        m = int(hex(n), 16)
        assert m == n
    return n


def printfd(t):
    with open("/dev/null", "w") as file:
        for i in range(t):
            file.write("{:d} {:d}\n".format(i, i + 1))


def print_perf(name, time):
    print(f"transonic, {name:30s} {time * 1000:8.3f} ms", flush=True)


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
        numbers = np.random.randint(0, 2 ** 32 - 1, 1000)
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
