
import sys
import numpy as np

if len(sys.argv) < 4:
    print("Usage:", sys.argv[0], "<filename with file list> <output file> <dir list>")
    sys.exit(1)

filename_file, output_file = sys.argv[1:3]
dir_list = sys.argv[3:]

filenames = [k.strip() for k in open(filename_file).readlines()]

stats = []
for i_file in range(len(filenames)):
    filename = filenames[i_file]
    stats += [[]]
    for batch_size in [2, 5, 10, 50]:
        stats[-1] += [[]]
        for directory in dir_list:
            batch_output_dir = directory + "/" + filename + "/" + str(batch_size)
            stats[-1] += [[]]
            with open(batch_output_dir + "/stats.txt") as f:
                for line in f:
                    line = [k.strip() for k in line.strip().split("\t")]
                    if len(line) != 5:
                        continue
                    stats[-1][-1][-1] += [[float(k) for k in line[:4]] + [float(k) for k in line[5][1:-1].split()]]
                    assert len(top_idx_stats[-1]) == 4

stats = np.array(stats, dtype=np.float32)
np.save(output_file, stats)
