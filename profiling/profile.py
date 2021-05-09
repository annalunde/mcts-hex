import pstats
from pstats import SortKey

path = r"output_file.txt"
p = pstats.Stats(path).strip_dirs()

p.sort_stats(SortKey.CUMULATIVE).print_stats(100)

# p.sort_stats(SortKey.TIME, SortKey.CUMULATIVE).print_stats(0.5, 'init')

# Run file as 'python profiling/profile.py > profiling/formatted_output.txt' for large outputs
