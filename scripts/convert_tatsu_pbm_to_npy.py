
import sys
import os
import numpy as np

if len(sys.argv) < 5:
    print("Usage:", sys.argv[0], "<data dir> <output dir> <file containing filename> <num files to choose>", file=sys.stderr)
    sys.exit(1)

data_dir, output_dir, filename, num_files = sys.argv[1:]
num_files = int(num_files)

filenames = []
with open(filename) as f:
    for line in f:
        if len(filenames) >= num_files:
            break
        filenames += [line.strip()]


base_to_idx = {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3}

for filename in filenames:
    print('doing file', filename)
    filepath = data_dir + "/" + filename
    dirpath = output_dir + "/" + filename
    try:
        os.mkdir(dirpath)
    except OSError as e:
        pass
    with open(filepath) as f:
        next(f)
        seq_arr = []
        score_arr = []
        for line in f:
            line = [k.strip() for k in line.strip().split()]
            seq = [base_to_idx[base] for base in line[0]]
            seq_arr += [np.eye(4)[seq].flatten()]
            score_arr += [line[3]]
        seq_arr = np.array(seq_arr)
        score_arr = np.array(score_arr)
        np.save(dirpath + '/inputs.npy', seq_arr)
        np.save(dirpath + '/labels.npy', score_arr)
