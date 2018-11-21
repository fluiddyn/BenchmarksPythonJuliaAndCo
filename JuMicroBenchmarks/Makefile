
cachedjit: perf.py perf_pythran.py .dummy
	python perf_pythran.py

print_table: all
	python print_table.py

all: output_pythran.txt output_python.txt output_julia.txt

output_julia.txt: perf.jl
	julia perf.jl | tee output_julia.txt

output_pythran.txt: .dummy perf_pythran.py perf.py
	python perf_pythran.py | tee output_pythran.txt

output_python.txt: perf.py
	python perf.py | tee output_python.txt

.dummy: fib.py
	pythran -march=native -DUSE_XSIMD fib.py
	touch .dummy

clean:
	rm -f output_*.txt *.so .dummy