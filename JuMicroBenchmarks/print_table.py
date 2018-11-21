names = [
    "recursion_fibonacci",
    "parse_integers",
    "userfunc_mandelbrot",
    "recursion_quicksort",
    "pisum",
    "pisum_vec",
    "matrix_statistics",
    "matrix_statistics_ones",
    "matrix_multiply",
    "matrix_multiply_ones",
]

tools = ["julia", "python", "pythran"]

tool = tools[0]


def get_times(tool):

    with open("output_" + tool + ".txt") as file:
        txt = file.read()

    times = {}

    for line in txt.split("\n"):
        words = line.split()
        try:
            key = words[1]
        except IndexError:
            continue
        if key in names:
            times[key] = float(words[2])

    return times


times_tools = {}

for tool in tools:
    times_tools[tool] = get_times(tool)

times_ju = times_tools["julia"]
times_python = times_tools["python"]
times_pythran = times_tools["pythran"]


print(f"{'':25s}| python/julia | pythran/julia | pythran/python |")

for name in names:
    t_ju = times_ju[name]
    t_python = times_python[name]
    t_pythran = times_pythran[name]

    print(
        f"{name:25s}|      {t_python/t_ju:5.2f}   |     {t_pythran/t_ju:5.2f}     |     {t_pythran/t_python:5.2f}      |"
    )
