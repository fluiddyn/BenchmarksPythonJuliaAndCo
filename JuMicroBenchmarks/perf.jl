# This file was formerly a part of Julia. License is MIT: https://julialang.org/license

using Compat

import Compat.LinearAlgebra
import Compat.Test
import Compat.Printf
import Compat.Statistics
import Compat.Sys

include("./perfutil.jl")

## slow pi series ##

function pisum()
    sum = 0.0
    n = 500
    for j = 1:n
        for k = 1:10000
            sum += 1.0/(k*k)
        end
    end
    sum/n
end

@compat Test.@test abs(pisum()-1.644834071848065) < 1e-10
@timeit pisum() "pisum" "Summation of a power series"



## slow pi series, vectorized ##

function pisum_vec()
    s = 0.0
    a = (1:10000.)
    n = 500
    for j = 1:n
        s += sum(1 ./ (a.^2))
    end
    s / n
end

@compat Test.@test abs(pisum_vec()-1.644834071848065) < 1e-10
@timeit pisum_vec() "pisum_vec" "Summation of a power series"


## recursive fib ##

fib(n) = n < 2 ? n : fib(n-1) + fib(n-2)

@compat Test.@test fib(20) == 6765
@timeit fib(20) "recursion_fibonacci" "Recursive fibonacci"

## parse integer ##

function parseintperf_rand(t)
    local n, m
    for i=1:t
        n = rand(UInt32)
        @static if VERSION >= v"0.7.0-DEV.4446"
            s = string(n, base = 16)
            m = UInt32(parse(Int64, s, base = 16))
        else
            s = hex(n)
            m = UInt32(parse(Int64, s, 16))
        end
        @assert m == n
    end
    return n
end

@timeit parseintperf_rand(1000) "parse_integers_rand" "Integer parsing"

function parseintperf(numbers)
    local n, m
    for n in numbers
        @static if VERSION >= v"0.7.0-DEV.4446"
            s = string(n, base = 16)
            m = UInt32(parse(Int64, s, base = 16))
        else
            s = hex(n)
            m = UInt32(parse(Int64, s, 16))
        end
        @assert m == n
    end
    return numbers[end]
end

numbers = rand(UInt32, 1000)

@timeit parseintperf(numbers) "parse_integers" "Integer parsing"

## array constructors ##

@compat Test.@test all(fill(1.,200,200) .== 1)

## matmul and transpose ##

A = fill(1.,200,200)
@compat Test.@test all(A*A' .== 200)
# @timeit A*A' "AtA" "description"

## mandelbrot set: complex arithmetic and comprehensions ##

function myabs2(z)
    return real(z)*real(z) + imag(z)*imag(z)
end

function mandel(z)
    c = z
    maxiter = 80
    for n = 1:maxiter
        if myabs2(z) > 4
            return n-1
        end
        z = z^2 + c
    end
    return maxiter
end

mandelperf() = [ mandel(complex(r,i)) for i=-1.:.1:1., r=-2.0:.1:0.5 ]
@compat Test.@test sum(mandelperf()) == 14791
@timeit mandelperf() "mandelbrot" "Calculation of mandelbrot set"

## numeric vector sort ##

function qsort!(a,lo,hi)
    i, j = lo, hi
    while i < hi
        pivot = a[(lo+hi)>>>1]
        while i <= j
            while a[i] < pivot; i += 1; end
            while a[j] > pivot; j -= 1; end
            if i <= j
                a[i], a[j] = a[j], a[i]
                i, j = i+1, j-1
            end
        end
        if lo < j; qsort!(a,lo,j); end
        lo, j = i, hi
    end
    return a
end

sortperf(n) = qsort!(rand(n), 1, n)
@compat Test.@test issorted(sortperf(5000))
@timeit sortperf(5000) "recursion_quicksort" "Sorting of random numbers using quicksort"

## random matrix statistics ##

function randmatstat(t)
    n = 5
    v = zeros(t)
    w = zeros(t)
    for i=1:t
        a = randn(n,n)
        b = randn(n,n)
        c = randn(n,n)
        d = randn(n,n)
        P = [a b c d]
        Q = [a b; c d]
        @static if VERSION >= v"0.7.0"
            v[i] = LinearAlgebra.tr((P'*P)^4)
            w[i] = LinearAlgebra.tr((Q'*Q)^4)
        else
            v[i] = trace((P'*P)^4)
            w[i] = trace((Q'*Q)^4)
        end
    end
    @compat return (Statistics.std(v)/Statistics.mean(v), Statistics.std(w)/Statistics.mean(w))
end

(s1, s2) = randmatstat(1000)
@compat Test.@test 0.5 < s1 < 1.0 && 0.5 < s2 < 1.0
@timeit randmatstat(1000) "matrix_statistics_rand" "Statistics on a random matrix"


function randmatstat_ones(t)
    n = 5
    v = zeros(t)
    w = zeros(t)
    for i=1:t
        a = ones(n,n)
        b = ones(n,n)
        c = ones(n,n)
        d = ones(n,n)
        P = [a b c d]
        Q = [a b; c d]
        @static if VERSION >= v"0.7.0"
            v[i] = LinearAlgebra.tr((P'*P)^4)
            w[i] = LinearAlgebra.tr((Q'*Q)^4)
        else
            v[i] = trace((P'*P)^4)
            w[i] = trace((Q'*Q)^4)
        end
    end
    @compat return (Statistics.std(v)/Statistics.mean(v), Statistics.std(w)/Statistics.mean(w))
end

@timeit randmatstat_ones(1000) "matrix_statistics_ones" "Statistics on a random matrix"

## largish random number gen & matmul ##

@timeit rand(1000,1000)*rand(1000,1000) "matrix_multiply_rand" "Multiplication of random matrices"

@timeit ones(1000,1000)*ones(1000,1000) "matrix_multiply_ones" "Multiplication of ones matrices"

@timeit rand(1000,1000) "random" "random generation"

function my_multi_broadcast(a)
    @. 10 * (2*a^2 + 4*a^3) + 2 / a
end

a = ones(1000,1000)
@timeit my_multi_broadcast(a) "broadcast" "broadcast"

function my_multi_broadcast_inplace(a)
    @. a = 10 * (2*a^2 + 4*a^3) + 2 / a
end

a = ones(1000,1000)
@timeit my_multi_broadcast_inplace(a) "broadcast_inplace" "broadcast"


## printfd ##

@compat if Sys.isunix()
    function printfd(n)
        open("/dev/null", "w") do io
            for i = 1:n
                @compat Printf.@printf(io, "%d %d\n", i, i + 1)
            end
        end
    end

    printfd(1)
    @timeit printfd(100000) "print_to_file" "Printing to a file descriptor"
end

#maxrss("micro")
