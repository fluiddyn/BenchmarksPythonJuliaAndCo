# Numerical benchmarks for Julia, Python.

We try to avoid trivial and non-sens benchmarks (for people doing numerics!)
like fibonacci, sorting and so on.

We put ourself in the shoes of a typical Matlab or Matlab-like programmer,
writing quite short but numerically intensive programs.

Are Python and Julia easy to use and efficient? We compare them with a C++
optimized implementation (and sometimes with a Fortran one).

The benchmark(s):

* Gaussian:  Gaussian elimination with partial pivoting.

* FeStiff: compute the stiffness matrix, for the Poisson equation,
  discretized with P2 finite elements on triangles.

* Weno: a classical solver for hyperbolic equations, in
  dimension 1, with application to Burghers equation and to Convection.

* :new: Sparse: building a sparse matrix and doing a sparse matrix x vector product.

* MicroBenchmarks: very simple benchmarks to show the importance
  of different programing styles.

We will add other numerical significant benchmarks in the (near) future.

### Dependencies:

#### What you need to install:

* python3
* pip (pip3)
* g++ (and/or clang++)
* gfortran
* lapack
* openblas
* cmake
* gnuplot

You can install them using your distribution tool (apt...).

* julia

:exclamation: Julia :exclamation: since 2018-10-08 programs need at least
version Version 1.1 (stable version in 2018-10); note that all programs needed
adaptation when moving to this version, and will not run with former ones.

Note also that the version packaged with Ubuntu 18-04 is older. Install the
stable version from [here](https://julialang.org). Note also that Julia is
evolving, and it is possible that the codes need some adaptation to run with
later versions of the language.

You also need:

* pythran
* scipy
* Numpy
* numba

to install them,  you can just do:

```bash
pip install pythran
```

and so on...

You can also install them from [conda](https://conda.io/docs/).

## TODO list

* Add a small static website (Sphinx? Pelican?) to present
  the results (Readthedocs?)

* Improve the quality of the figures (using Pandas +
  Matplotlib?)

* Automate the execution of the benchmarks and the creation of
  the website (Gitlab CI?).

* Add something based on the [Julia micro
  benchmarks](https://julialang.org/benchmarks/)

* Maybe add Fortran if a Fortran developper is willing to
  work on this...

* Maybe add Xtensor if a xtensor developper is willing to
  work on this...