# Modified Julia micro benchmarks

See the [Julia web page](https://julialang.org/benchmarks/).

Install the dependencies with

```
pip install -r requirements.txt
```

Then, to run the benchmark:

```
make print_table
```

I got

```
                         | python/julia | pythran/julia | python/pythran |
pisum                    |      17.89   |      1.00     |     17.98      |
pisum_vec                |       1.12   |      0.94     |      1.20      |
recursion_fibonacci      |      62.12   |      1.00     |     62.12      |
recursion_quicksort      |      54.98   |      1.12     |     49.03      |
mandelbrot               |      87.42   |      0.93     |     94.05      |
matrix_statistics_ones   |       6.41   |      0.91     |      7.02      |
matrix_multiply_ones     |       1.01   |      0.99     |      1.02      |
broadcast                |      15.02   |      0.55     |     27.10      |
broadcast_inplace        |      15.03   |      0.51     |     29.27      |
parse_integers_rand      |       3.55   |      3.42     |      1.04      |
random                   |       2.75   |      4.44     |      0.62      |
matrix_statistics_rand   |       6.59   |      1.31     |      5.05      |
matrix_multiply_rand     |       1.27   |      1.50     |      0.85      |
```

Julia and Pythran have very similar performances.

Python with Numpy is very good for simple matrix multiplication and pisum_vec.
For other microbenchmarks Python is slow or very slow.

Pythran is slower than Julia for random number generation and for the
benchmark called parse_integers (`int(hex(n), 16)`), which may not be not so
important for scientific computing.

For the 2 last benchmarks, Pythran is slightly slower than Julia because of
slower random generation. For pure `matrix_statistics`, Pythran is actually
slightly faster than Julia.

We see that the random generation (also involved in matrix_statistics and
matrix_multiply) is very fast in Julia. Julia, Pythran and Numpy do not use the
same random generators...

Julia micro-benchmarks are not very well written because they do not really
measure "only one micro-task", but [random generation + the micro-task]. It
should be better to get a random generation benchmark and to avoid random
generation in other benchmarks...

For Python (without Pythran), we use the package randomgen to get faster
random generation that with numpy.random.

On the other hand, Pythran is nearly twice as fast as Julia for complex
broadcasting operations (here `10 * (2*a**2 + 4*a**3) + 2 / a`). Being able to
be very fast for such operations with such readable (and dimension independent)
syntax is very interesting for scientific computing.

### Three concluding remarks

1. Pythran performs a 3-step compilation (high level Python -> optimized
   Python, optimized Python -> C++ and finally C++ compilation). We don't see the
   benefice of the high level Python -> Python optimization with such very simple
   cases...

2. Julia is overall very fast, with a fast JIT. It's impressive.

3. For (data) scientists, the bad performance of Python for computational tasks
   (such as fibonacci, quicksort, mandelbrot, or even broadcast) is not so much a
   problem as long as we have good tools (for example Pythran) to speed-up these
   parts. Then, the overall program will not be slow.

### Two scientists using open-source

- *Pure Python is much slower than Julia!*

- *Yes, but Python-Numpy code can be boosted to get very good performance...*

- *Ok, but Julia has multiple dispatch and funky syntaxes!*

- *Python also has cool language features and libraries! Anyway, I need to be good at Python for doing so many things.*

So Python, C++ and Julia (and other languages) are complementary for scientific computing.

### Notes

Pythran is configured with `~/.pythranrc`:

```
[pythran]
complex_hook=True

[compiler]
blas=openblas
CXX = clang++-6.0
CC = clang-6.0
```
