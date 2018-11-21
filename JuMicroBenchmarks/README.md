# Julia micro benchmarks

See the [Julia web page](https://julialang.org/benchmarks/).

with few modifications...

For the Pythran benchmark, we use FluidPythran. Install with:

```
pip install fluidpythran
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
recursion_fibonacci      |      76.51   |      1.23     |      0.02      |
parse_integers           |      12.31   |      3.42     |      0.28      |
userfunc_mandelbrot      |      82.65   |      1.06     |      0.01      |
recursion_quicksort      |      53.98   |      1.13     |      0.02      |
pisum                    |      17.64   |      1.00     |      0.06      |
pisum_vec                |       1.12   |      0.93     |      0.84      |
matrix_statistics        |       6.41   |      3.14     |      0.49      |
matrix_statistics_ones   |       6.12   |      1.58     |      0.26      |
matrix_multiply          |       1.88   |     18.43     |      9.80      |
matrix_multiply_ones     |       1.01   |      9.84     |      9.76      |
```

Pythran and Julia have usually very similar performance.

We see that the random generation (involved in matrix_statistics and
matrix_multiply) is very fast in Julia. Julia and Numpy do not use the same
random generator...

There is a real performance issue for Pythran for matrix multiplication (here,
the shape is (1000, 1000))! It's nearly 10 times slower than with Numpy!
