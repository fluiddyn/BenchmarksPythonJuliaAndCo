
cachedjit: perf_py.py perf_pythran.py .dummy
	python perf_pythran.py
	# for an "ahead-of-time function" (random without XSIMD)
	fluidpythran perf_pythran.py -pf "-march=native "

print_table: all
	python print_table.py

all: output_pythran.txt output_python.txt output_julia.txt

output_julia.txt: perf.jl
	julia perf.jl | tee output_julia.txt

output_pythran.txt: .dummy perf_pythran.py perf_py.py
	python perf_pythran.py | tee output_pythran.txt

output_python.txt: perf_py.py
	python perf_py.py | tee output_python.txt

.dummy: fib.py
	pythran -march=native -DUSE_XSIMD fib.py
	touch .dummy

clean:
	rm -f output_*.txt

clean_output_pythran:
	rm -f output_pythran.txt

cleanall: clean
	rm -f *.so .dummy
	fluidpythran -cc perf_pythran.py -f
