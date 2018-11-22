# Julia micro benchmarks

See the [Julia web page](https://julialang.org/benchmarks/).

with few modifications...

For the Pythran benchmark, we use FluidPythran. Install with:

```
pip install pythran fluidpythran
```

Since the Pythran cached JIT can be slow. It is better to warm it up before the benchmark... Run

```
make cachedjit
```

Then, to run the benchmark:

```
make print_table
```

You should get something like this:

```
                         | python/julia | pythran/julia | pythran/python |
pisum                    |      17.64   |      1.00     |      0.06      |
pisum_vec                |       1.09   |      0.94     |      0.86      |
recursion_fibonacci      |      67.87   |      1.07     |      0.02      |
recursion_quicksort      |      55.03   |      1.12     |      0.02      |
mandelbrot               |      82.72   |      1.04     |      0.01      |
parse_integers           |      11.55   |      3.38     |      0.29      |
matrix_statistics        |       6.33   |      3.15     |      0.50      |
matrix_statistics_ones   |       6.05   |      1.58     |      0.26      |
matrix_multiply          |       1.97   |     20.00     |     10.16      |
matrix_multiply_ones     |       1.01   |     10.67     |     10.52      |
random                   |       7.49   |     67.07     |      8.95      |
```

Pythran and Julia have usually very similar performance.

We see that the random generation (involved in matrix_statistics and
matrix_multiply) is very fast in Julia. Julia and Numpy do not use the same
random generator... See https://github.com/serge-sans-paille/pythran/issues/759

There is a real performance issue for Pythran for matrix multiplication (here,
the shape is (1000, 1000))! It's nearly 10 times slower than with Numpy!

The solution was to change the blas library used by Pythran. With `blas=mkl`,
i.e. a file `~/.pythranrc` containing

```
[pythran]
complex_hook=True

[compiler]
blas=mkl
CXX = clang++-6.0
CC = clang-6.0
```

I got

```
                         | python/julia | pythran/julia | pythran/python |
pisum                    |      17.70   |      1.00     |      0.06      |
pisum_vec                |       1.10   |      0.94     |      0.85      |
recursion_fibonacci      |      72.45   |      1.04     |      0.01      |
recursion_quicksort      |      54.96   |      1.12     |      0.02      |
mandelbrot               |      84.07   |      1.06     |      0.01      |
parse_integers           |      12.55   |      3.69     |      0.29      |
matrix_statistics        |       6.71   |      3.74     |      0.56      |
matrix_statistics_ones   |       6.40   |      2.30     |      0.36      |
matrix_multiply          |       1.97   |     10.86     |      5.52      |
matrix_multiply_ones     |       1.01   |      0.95     |      0.94      |
random                   |       7.48   |     66.91     |      8.95      |
```

With `blas=openblas`:

```
                         | python/julia | pythran/julia | pythran/python |
pisum                    |      17.70   |      1.00     |      0.06      |
pisum_vec                |       1.11   |      0.94     |      0.84      |
recursion_fibonacci      |      63.77   |      1.00     |      0.02      |
recursion_quicksort      |      55.49   |      1.14     |      0.02      |
mandelbrot               |      83.58   |      1.04     |      0.01      |
parse_integers           |      13.64   |      3.40     |      0.25      |
matrix_statistics        |       6.58   |      2.60     |      0.39      |
matrix_statistics_ones   |       6.24   |      0.94     |      0.15      |
matrix_multiply          |       1.96   |     11.43     |      5.82      |
matrix_multiply_ones     |       1.01   |      0.99     |      0.98      |
random                   |       7.48   |     67.45     |      9.02      |
```

There is a real performance issue for Pythran for random generation (Pythran
~10 times slower than Numpy) which slow down many benchmarks (parse_integers,
matrix_statistics, matrix_multiply, random).
