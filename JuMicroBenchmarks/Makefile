
print_table: all
	python print_table.py

all: output_pythran.txt output_python.txt output_julia.txt

output_julia.txt: perf.jl
	julia perf.jl | tee output_julia.txt

output_pythran.txt: perf_pythran.py perf_py.py
	python perf_pythran.py warmup && python perf_pythran.py | tee output_pythran.txt

output_python.txt: perf_py.py
	python perf_py.py | tee output_python.txt

clean:
	rm -f output_*.txt

clean_output_pythran:
	rm -f output_pythran.txt

cleancache:
	fluidpythran -cc perf_pythran.py -f

cleanall: clean cleancache
