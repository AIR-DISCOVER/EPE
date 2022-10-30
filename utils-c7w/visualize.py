import os
import random

import imageio

ROOT_DIR = "/DATA_EDS/gaoha/datasets/Carla/NightDepth/saved/night-depth2DarkZurich/"
dataset_csv = os.path.join(ROOT_DIR, "filtered_matches.csv")
assert os.path.exists(dataset_csv)

# Open dataset_csv
with open(dataset_csv, 'r') as f:
    lines = f.readlines()[1:]

    # Randomly shuffle the lines
    random.shuffle(lines)

    # 'src_path', 'src_r0', 'src_r1', 'src_c0', 'src_c1', 'dst_path', 'dst_r0', 'dst_r1', 'dst_c0', 'dst_c1'
    for k in range(50):

        # First, generate a random hash value
        hash_val = ''.join([random.choice('0123456789abcdef') for _ in range(6)])

        tmp = lines[k].strip().split(',')

        src_r0 = int(tmp[1])
        src_r1 = int(tmp[2])
        src_c0 = int(tmp[3])
        src_c1 = int(tmp[4])
        dst_r0 = int(tmp[6])
        dst_r1 = int(tmp[7])
        dst_c0 = int(tmp[8])
        dst_c1 = int(tmp[9])

        # Open src_path as img
        img = imageio.imread(tmp[0])
        img2 = imageio.imread(tmp[5])
        sample_dir = os.path.join(ROOT_DIR, "sample")
        os.makedirs(sample_dir, exist_ok=True)
        imageio.imwrite(f'{sample_dir}/{hash_val}_Carla.png', img[src_r0:src_r1, src_c0:src_c1, :])
        imageio.imwrite(f'{sample_dir}/{hash_val}_CityScapes.png', img2[dst_r0:dst_r1, dst_c0:dst_c1, :])
