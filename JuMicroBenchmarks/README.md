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
pisum                    |      17.64   |      1.00     |      0.06      |
pisum_vec                |       1.10   |      0.94     |      0.85      |
recursion_fibonacci      |      66.64   |      1.04     |      0.02      |
recursion_quicksort      |      54.94   |      1.11     |      0.02      |
mandelbrot               |      83.79   |      1.06     |      0.01      |
parse_integers           |      12.17   |      3.33     |      0.27      |
matrix_statistics        |       6.38   |      3.16     |      0.50      |
matrix_statistics_ones   |       6.19   |      1.58     |      0.26      |
matrix_multiply          |       1.96   |     19.95     |     10.17      |
matrix_multiply_ones     |       1.02   |     10.61     |     10.41      |

```

Pythran and Julia have usually very similar performance.

We see that the random generation (involved in matrix_statistics and
matrix_multiply) is very fast in Julia. Julia and Numpy do not use the same
random generator... See https://github.com/serge-sans-paille/pythran/issues/759

There is a real performance issue for Pythran for matrix multiplication (here,
the shape is (1000, 1000))! It's nearly 10 times slower than with Numpy!
