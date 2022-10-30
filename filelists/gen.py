import os
import sys


def generate(root, filepath):
    with open(filepath, 'w') as f:
        namelist = [i for i in sorted(os.listdir(f"{root}/mask_v/"), key=lambda i: int(i[:-4]))]
        for i in namelist:
            print(f"{root}/rgb_v/{i},{root}/mask_v/{i},{root}/rgb_v/{i},{root}/mask_v/{i}",file=f)

if __name__ == '__main__':
    generate(sys.argv[1], sys.argv[2])
