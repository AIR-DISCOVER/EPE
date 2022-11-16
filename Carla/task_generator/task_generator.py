"""
    Given `file_fake.txt`, `file_real.txt`, (Optional) `file_fake_test.txt`,
    Generate a directory holding stats in `./epe/stats/<fake>2<real>` and a `<fake>2<real>.yaml` in `./config`
    Created by c7w on 01/07/22.
"""
import subprocess

epe_root = "."
task_root = "./saved_tasks"

import os, sys

sys.path.append(os.path.join(epe_root, "code"))
from argparse import ArgumentParser

src_path_dict = {
    "carla-gbuffer": "data/file_lists/fake_datasets/CarlaDsWithGBuffer/file.txt",
    "carla-gbuffer-fog": "data/file_lists/fake_datasets/CarlaDsNight/fog.txt",
    "carla-gbuffer-night": "data/file_lists/fake_datasets/CarlaDsNight/night.txt",
    "carla-gbuffer-night-light": "data/file_lists/fake_datasets/CarlaDsNightLightOn/night-light.txt",
    "carla-gbuffer-rain": "data/file_lists/fake_datasets/CarlaDsNight/rain.txt",
    "carla-night-town03": "data/file_lists/fake_datasets/NightDepthLarge/town03.txt",
    "carla-night-town10": "data/file_lists/fake_datasets/NightDepthLarge/town10.txt",
    "night-depth": "data/file_lists/fake_datasets/NightDepth/src/file.txt",
    "nd2":  "data/file_lists/fake_datasets/NightDepth/src2/file.txt",
    "Day0923": "data/file_lists/fake_datasets/Day0923/file.txt",
}

dst_path_dict = {
    "cityscapes": "data/file_lists/real_datasets/cityscapes.txt",
    "acdc-night": "data/file_lists/real_datasets/ACDC/night.txt",
    "acdc-snow": "data/file_lists/real_datasets/ACDC/snow.txt",
    "acdc-rain": "data/file_lists/real_datasets/ACDC/rain.txt",
    "acdc-fog": "data/file_lists/real_datasets/ACDC/fog.txt",
    "night-depth-deeplab": "data/file_lists/real_datasets/NightDepth/file_DeepLab.txt",
    "night-depth-pspnet": "data/file_lists/real_datasets/NightDepth/file_PSPNet.txt",
    "night-depth-refinenet": "data/file_lists/real_datasets/NightDepth/file_RefineNet.txt",
    "seq1": "data/file_lists/real_datasets/nusc/file1.txt",
    "seq2": "data/file_lists/real_datasets/nusc/file2.txt",
    "seq3": "data/file_lists/real_datasets/nusc/file3.txt",
    "seq4": "data/file_lists/real_datasets/nusc/file4.txt",
    "seq5": "data/file_lists/real_datasets/nusc/file5.txt",
    "seq6": "data/file_lists/real_datasets/nusc/file6.txt",
    "DarkZurich": "data/file_lists/real_datasets/DarkZurich/file.txt",
    "NightCity": "data/file_lists/real_datasets/NightCity/file.txt",
    "NC-Nagoya": "data/file_lists/real_datasets/NightCity/cities/Nagoya_1025.txt",
    "bdd100k": "data/file_lists/real_datasets/bdd100k/file.txt",
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Source type.")
    parser.add_argument("--dst", type=str, required=True, help="Destination type.")
    args = parser.parse_args()

    src, dst = args.src, args.dst
    assert src in src_path_dict.keys() and dst in dst_path_dict.keys(), "KeyError: You must specify a right type name!"

    src_path, dst_path = src_path_dict[src], dst_path_dict[dst]

    # Makedir
    task_name = f"{src}2{dst}"
    task_dir = os.path.join(task_root, task_name)
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(os.path.join(task_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(task_dir, "checkpoints"), exist_ok=True)

    print("=== Task Generator for EPE ===")
    print(f"==> Source: {src}")
    print(f"==> Target: {dst}")
    print("[1/6] Generating crops for source dataset...")
    if not os.path.exists(os.path.join(task_dir, "crop_source.npz")):
        subprocess.run(f"python3 {epe_root}/epe/matching/feature_based/collect_crops.py" +
                       f" source {src_path} --out_dir {task_dir} ", shell=True)
    with open(src_path, 'r') as src_file:
        text_list = src_file.read().strip().split("\n")[0:4000:40]
        with open(os.path.join(task_dir, "test.txt"), 'w+') as g:
            g.write("\n".join(text_list))
    src_path_test = os.path.join(task_dir, "test.txt")

    print("[2/6] Generating crops for target dataset...")
    if not os.path.exists(os.path.join(task_dir, "crop_target.npz")):
        subprocess.run(f"python3 {epe_root}/epe/matching/feature_based/collect_crops.py" +
                       f" target {dst_path} --out_dir {task_dir} ", shell=True)

    print("[3/6] Finding matches between source and target dataset...")
    subprocess.run(' '.join(["python3", f"{epe_root}/epe/matching/feature_based/find_knn.py",
                             f"{task_dir}/crop_source.npz", f"{task_dir}/crop_target.npz",
                             f"{task_dir}/matches.npz"]), shell=True)

    print("[4/6] Filtering matches...")
    subprocess.run(' '.join(["python3", f"{epe_root}/epe/matching/filter.py",
                             f"{task_dir}/matches.npz",
                             f"{task_dir}/crop_source.csv", f"{task_dir}/crop_target.csv",
                             "0.6", f"{task_dir}/filtered_matches.csv"]), shell=True)
    filtered_matches_csv = os.path.join(task_dir, "filtered_matches.csv")

    print("[5/6] Calculating patch weights...")
    subprocess.run(' '.join(["python3", f"{epe_root}/epe/matching/compute_weights.py",
                             f"{task_dir}/filtered_matches.csv", "1080", "1920",
                             f"{task_dir}/weights.npz"]), shell=True)
    weights_npz = os.path.join(task_dir, "weights.npz")

    # Generate task yaml config file
    print("[6/6] Generating task configuration yaml and bash script files...")
    script_path = os.path.realpath(__file__)
    task_yaml = os.path.abspath(os.path.join(script_path, os.pardir, f'train_carla2cs_template.yaml'))
    with open(task_yaml, 'r') as file:
        text = file.read()
        text = text.replace("%%task_dir%%", task_dir) \
            .replace("%%task_name%%", task_name) \
            .replace("%%real_path%%", dst_path) \
            .replace("%%src_path%%", src_path) \
            .replace("%%src_path_test%%", src_path_test) \
            .replace("%%filtered_matches_csv%%", filtered_matches_csv) \
            .replace("%%weights_npz%%", weights_npz)

        with open(os.path.join(epe_root, "code", "config", f"{task_name}.yaml"), 'w+') as g:
            g.write(text)
    task_yaml = os.path.join(epe_root, "code", "config", f"{task_name}.yaml")

    # Generate bash scripts
    with open(os.path.abspath(os.path.join(script_path, os.pardir, 'bash_template.sh')), 'r') as file:
        text = file.read()
        text = text.replace("%%task_dir%%", task_dir) \
            .replace("%%task_yaml%%", task_yaml)

        with open(os.path.join(epe_root, "code", "bash_scripts", f"{task_name}.sh"), 'w+') as g:
            g.write(text)

    print("Done! Please check the following files:")
    print(f'''+ {os.path.join(epe_root, "code", "bash_scripts", f"{task_name}.sh")}''')
    print(f'''+ {os.path.join(epe_root, "code", "config", f"{task_name}.yaml")}''')