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
pisum                    |      17.65   |      1.00     |      0.06      |
pisum_vec                |       1.09   |      0.93     |      0.86      |
recursion_fibonacci      |      78.68   |      1.24     |      0.02      |
recursion_quicksort      |      54.94   |      1.11     |      0.02      |
mandelbrot               |      84.93   |      0.94     |      0.01      |
matrix_statistics_ones   |       5.87   |      1.60     |      0.27      |
matrix_multiply_ones     |       1.01   |     10.60     |     10.50      |
broadcast                |      15.21   |      0.53     |      0.03      |
broadcast_inplace        |      15.08   |      0.51     |      0.03      |
random                   |       7.50   |     67.12     |      8.95      |
parse_integers           |      11.56   |      3.30     |      0.29      |
matrix_statistics        |       6.11   |      3.15     |      0.52      |
matrix_multiply          |       1.96   |     19.90     |     10.16      |
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
pisum                    |      17.65   |      1.00     |      0.06      |
pisum_vec                |       1.09   |      0.93     |      0.86      |
recursion_fibonacci      |      78.68   |      1.26     |      0.02      |
recursion_quicksort      |      54.94   |      1.13     |      0.02      |
mandelbrot               |      84.93   |      0.94     |      0.01      |
matrix_statistics_ones   |       5.87   |      2.29     |      0.39      |
matrix_multiply_ones     |       1.01   |      1.03     |      1.02      |
broadcast                |      15.21   |      0.53     |      0.03      |
broadcast_inplace        |      15.08   |      0.51     |      0.03      |
random                   |       7.50   |     67.11     |      8.94      |
parse_integers           |      11.56   |      3.31     |      0.29      |
matrix_statistics        |       6.11   |      3.79     |      0.62      |
matrix_multiply          |       1.96   |     10.70     |      5.46      |
```

With `blas=openblas`:

```
                         | python/julia | pythran/julia | pythran/python |
pisum                    |      17.67   |      1.00     |      0.06      |
pisum_vec                |       1.08   |      0.94     |      0.86      |
recursion_fibonacci      |      71.31   |      1.14     |      0.02      |
recursion_quicksort      |      55.27   |      1.10     |      0.02      |
mandelbrot               |      82.54   |      0.93     |      0.01      |
matrix_statistics_ones   |       5.99   |      0.95     |      0.16      |
matrix_multiply_ones     |       1.01   |      0.99     |      0.98      |
broadcast                |      15.32   |      0.53     |      0.03      |
broadcast_inplace        |      15.18   |      0.52     |      0.03      |
random                   |       7.50   |     67.06     |      8.94      |
parse_integers           |      11.90   |      3.36     |      0.28      |
matrix_statistics        |       6.06   |      2.57     |      0.42      |
matrix_multiply          |       1.97   |     11.37     |      5.78      |
```

There is a real performance issue for Pythran for random generation (Pythran
~10 times slower than Numpy) which slow down many benchmarks (parse_integers,
matrix_statistics, matrix_multiply, random).
