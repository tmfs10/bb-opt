
import sys
import os
import numpy as np
import torch


def get_data_mves(exp_folder, suffix, map_loc="cpu"):
    num_samples = 10
    stats_mves = {}
    batches = [2, 5, 10, 20]
    for i_sample in range(1, num_samples+1):
        dirpath = exp_folder + "/output_" + suffix + str(i_sample)
        if not os.path.exists(dirpath):
            continue
        print("reading sample", i_sample)

        for filename in os.listdir(dirpath):
            filepath = dirpath + "/" + filename
            if not os.path.isdir(filepath):
                continue

            if filename not in stats_mves:
                stats_mves[filename] = [ [] for i in range(len(batches))]
            for i_batch in range(len(batches)):
                batch_size = batches[i_batch]
                batch_dir = filepath + "/" + str(batch_size)
                if not os.path.exists(batch_dir) or not os.path.exists(batch_dir + "/19.pth"):
                    continue

                stats_mves[filename][i_batch] += [ [] ]
                with open(batch_dir + "/stats.txt") as f:
                    #print("reading", batch_dir)
                    i = -1
                    for line in f:
                        line = [k.strip() for k in line.strip().split("\t")]

                        if len(line) != 5:
                            continue

                        i += 1
                        assert i < 20
                        best_value = torch.load(batch_dir + "/" + str(i) + ".pth", map_location=map_loc)['ack_labels'].max().item()

                        assert line[4][0] == "["
                        assert line[4][-1] == "]"
                        top_idx_frac = [float(k) for k in line[4][1:-1].split(",")]
                        assert len(top_idx_frac) == 4

                        stats_mves[filename][i_batch][-1] += [ [float(k) for k in line[:4]] + [best_value] + top_idx_frac ]
                stats_mves[filename][i_batch][-1] = np.array(stats_mves[filename][i_batch][-1], dtype=np.float32) # (num_acks, 9)

    for filename in stats_mves:
        for i_batch in range(len(stats_mves[filename])):
            stats_mves[filename][i_batch] = np.array(stats_mves[filename][i_batch], dtype=np.float32)
    return stats_mves

def get_data_ucb(exp_folder, suffix, map_loc="cpu"):
    num_samples = 10
    stats_mves = {}
    batches = [2, 5, 10, 20]
    ucb_coeffs = [0.0, 0.5, 1.0, 2.0, 3.0]
    stats_ucb = [ {}  for i in range(len(ucb_coeffs))]
    for i_ucb in range(len(ucb_coeffs)):
        ucb = ucb_coeffs[i_ucb]
        print("reading", i_ucb)
        num_samples_read = 0
        for i_sample in range(1, num_samples+1):
            dirpath = exp_folder + "/output_ucb_" + suffix + str(i_sample) + "_" + str(ucb)
            if not os.path.exists(dirpath):
                continue

            num_samples_read += 1

            for filename in os.listdir(dirpath):
                filepath = dirpath + "/" + filename
                if not os.path.isdir(filepath):
                    continue
                #print("reading", filepath)

                if filename not in stats_ucb[i_ucb]:
                    stats_ucb[i_ucb][filename] = [ [] for i in range(len(batches))]
                for i_batch in range(len(batches)):
                    batch_size = batches[i_batch]
                    batch_dir = filepath + "/" + str(batch_size)
                    if not os.path.exists(batch_dir) or not os.path.exists(batch_dir + "/19.pth"):
                        continue

                    stats_ucb[i_ucb][filename][i_batch] += [ [] ]
                    with open(batch_dir + "/stats.txt") as f:
                        i = -1
                        for line in f:
                            line = [k.strip() for k in line.strip().split("\t")]

                            if line[0].startswith("best_ei_10"):
                                continue
                            i += 1
                            assert i < 20
                            best_value = torch.load(batch_dir + "/" + str(i) + ".pth", map_location=map_loc)['ack_labels'].max().item()

                            #assert line[4][0] == "[", str(line[4])
                            #assert line[4][-1] == "]", str(line[4])

                            top_idx_frac = line[4].split("[")[-1][:-1].split(",")
                            top_idx_frac = [float(k) for k in top_idx_frac]
                            assert len(top_idx_frac) == 4, str(top_idx_frac) + "\t" + batch_dir

                            stats_ucb[i_ucb][filename][i_batch][-1] += [ [float(k) for k in line[:4]] + [best_value] + top_idx_frac ]
                    stats_ucb[i_ucb][filename][i_batch][-1] = np.array(stats_ucb[i_ucb][filename][i_batch][-1], dtype=np.float32)
        print("read", num_samples_read, "samples")

        for filename in stats_ucb[i_ucb]:
            for i_batch in range(len(stats_ucb[i_ucb][filename])):
                stats_ucb[i_ucb][filename][i_batch] = np.array(stats_ucb[i_ucb][filename][i_batch], dtype=np.float32)

    return stats_ucb
