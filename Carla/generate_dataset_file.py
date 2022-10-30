import glob
import itertools
import os
import argparse
from multiprocessing import Process

import IPython
import imageio

##############################################################
# Constants
##############################################################

SERIES_FRAME_CNT = 800
SKIP_FRONT = 20

# CityScape Labels
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from PIL import Image

class ACDC:
    def __init__(self):
        self.label2id = {
            0: 1,  # road,
            1: 1,  # sidewalk,
            2: 10,  # building,
            3: 10,  # wall
            4: 10,  # fence
            5: 6,  # pole
            6: 7,  # traffic light
            7: 8, # traffic sign,
            8: 4,  # vegetation
            9: 3,  # terrain
            10: 0,  # sky
            11: 5,  # person
            12: 5,  # person
            13: 2,  # car
            14: 2,  # truck,
            15: 2,  # bus,
            16: 2,  # train
            17: 2,  # motor cycle
            18: 2,  # bicycle
            # if name in ['unlabeled', 'water', 'other', 'dynamic']: return 11
            # if name in ['building', 'fence', 'bridge']: return 10
            # if name in ['pole']: return 6
            # if name in ['wall', 'rail_track', 'guard_rail', 'road_line']: return 10
            # if name in ['person']: return 5
            # if name in ['sky']: return 0
            # if name in ['road', 'static', 'sidewalk', 'ground']: return 1
            # if name in ['car']: return 2
            # if name in ['vegetation']: return 4
            # if name in ['traffic_light']: return 7
            # if name in ['traffic_sign']: return 8
            # if name in ['terrain']: return 3
        }

    def generate_dataset_file(self, args):
        acdc_type = args.acdc_dir
        src_path, dst_path = args.src_path, args.dst_path

        result_images = []
        result_labels = []

        if acdc_type == "ref":
            assert False

        else:
            img_path = os.path.join(src_path, "images",  acdc_type)
            gt_path = os.path.join(src_path, "gt",  acdc_type)

            dir_set = ["train", "val"]

            for mode in dir_set:
                img_path_inner = os.path.join(img_path, mode)
                for scene_idx in os.listdir(img_path_inner):
                    img_list_path = os.path.join(img_path_inner, scene_idx)
                    for img_name in os.listdir(img_list_path):
                        result_image_path = os.path.join(img_list_path, img_name)
                        result_gt_path = os.path.join(gt_path, mode, scene_idx,
                                                   img_name[:len("GOPR0475_frame_000041")] + "_gt_labelIds.png")
                        result_images.append(os.path.abspath(result_image_path))
                        result_labels.append(os.path.abspath(result_gt_path))

        assert len(result_images) == len(result_labels)
        with open(dst_path, "w+") as file:
            for x, y in zip(result_images, result_labels):
                file.write(f"{x},{y}\n")






class CityScape:
    def __init__(self):
        self.name2color = {
            'unlabeled': (0, 0, 0),
            'road': (128, 64, 128),
            'sidewalk': (244, 35, 232),
            'building': (70, 70, 70),
            'wall': (102, 102, 156),
            'fence': (190, 153, 153),
            'pole': (153, 153, 153),
            'traffic_light': (250, 170, 30),
            'traffic_sign': (220, 220, 0),
            'vegetation': (107, 142, 35),
            'terrain': (152, 251, 152),
            'sky': (70, 130, 180),
            'person': (220, 20, 60),
            'rider': (255, 0, 0),
            'car': (0, 0, 142),
            'truck': (0, 0, 70),
            'bus': (0, 60, 100),
            'train': (0, 80, 100),
            'motorcycle': (0, 0, 230),
            'bicycle': (119, 11, 32),
            'rail_track': (230, 150, 140),
            'ground': (81, 0, 81),
            'guard_rail': (180, 165, 180),
            'bridge': (150, 100, 100),
        }

        self.color2name = {v: k for k, v in self.name2color.items()}
        CityScape.color2id = { k: Carla.name2id(v) for k, v in Carla.color2name.items() }

    def generate_dataset_file(self, args):
        pass

# Carla Labels
class Carla:
    def __init__(self):
        Carla.name2color = {
            'unlabeled': (0, 0, 0), ###
            'building': (70, 70, 70), ###
            'fence': (100, 40, 40), ###
            'other': (55, 90, 80),
            'person': (220, 20, 60), ###
            'pole': (153, 153, 153), ###
            'road_line': (157, 234, 50),
            'road': (128, 64, 128), ###
            'sidewalk': (244, 35, 232), ###
            'vegetation': (107, 142, 35), ###
            'car': (0, 0, 142), ###
            'wall': (102, 102, 156), ###
            'traffic_sign': (220, 220, 0), ###
            'sky': (70, 130, 180), ###
            'ground': (81, 0, 81), ###
            'bridge': (150, 100, 100), ###
            'rail_track': (230, 150, 140), ###
            'guard_rail': (180, 165, 180), ###
            'traffic_light': (250, 170, 30), ###
            'static': (110, 190, 160), ###
            'dynamic': (170, 120, 50),
            'water': (45, 60, 150), ###
            'terrain': (145, 170, 100), ###
            'anomaly': (255,255,255), ###
            'rider': (255, 0, 0), ###
            'truck': (0, 0, 70), ###
            'bus': (0, 60, 100), ###
            'train': (0, 80, 100), ###
            'motorcycle': (0, 0, 230), ###
            'bicycle': (119, 11, 32), ###
        }

        Carla.color2name = {v: k for k, v in Carla.name2color.items()}
        Carla.color2id = { k: Carla.name2id(v) for k, v in Carla.color2name.items() }
        print(Carla.color2id)

    @classmethod
    def name2id(cls, name):
        if name in ['unlabeled', 'water', 'other', 'dynamic', 'anomaly']: return 11
        if name in ['building', 'fence', 'bridge']: return 10
        if name in ['pole']: return 6
        if name in ['wall', 'rail_track', 'guard_rail', 'road_line']: return 10
        if name in ['person', 'rider']: return 5
        if name in ['sky']: return 0
        if name in ['road', 'static', 'sidewalk', 'ground']: return 1
        if name in ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']: return 2
        if name in ['vegetation']: return 4
        if name in ['traffic_light']: return 7
        if name in ['traffic_sign']: return 8
        if name in ['terrain']: return 3




    def generate_dataset_file(self, args):

        f = open(args.dst_path, 'w+', encoding='utf-8')

        # Test if src_path exists
        if not os.path.exists(args.src_path):
            raise Exception("Source path does not exist.")

        # Iterate through every directory in the src_path
        print( [os.listdir(args.src_path)[0],])
        for dir_name in os.listdir(args.src_path):

            # TODO: change this for generating other datasets!
            # Now generate foggy days only
            if not "Town10" in dir_name:
                continue

            print(f"Processing {dir_name}...")
            
            iid = sorted([int(x) for x in os.listdir(os.path.join(args.src_path, dir_name))])[-1]

            # G Buffers

            # Prepare filenames
            g_buffer_list = ["SceneDepth", "SceneColor", "SceneGBufferA", "SceneGBufferB", "SceneGBufferC", "SceneGBufferD"]
             
            
            # Iterate through all the pngs in tmp_path
            for frame_idx in tqdm(range(SKIP_FRONT, SERIES_FRAME_CNT+1), total=SERIES_FRAME_CNT-SKIP_FRONT+1):
                
                flag = False
                for ii in range(6):
                    g_buffer_file_name = f"{frame_idx}-{g_buffer_list[ii]}.png"
                    base_path = os.path.join(args.src_path, dir_name, str(iid), "gbuffer_v")
                    if not os.path.exists(os.path.join(base_path, g_buffer_file_name)):
                        flag = True
                        break
                
                file_name = f"{frame_idx}.png"
                if not os.path.exists(Path(os.path.join(args.src_path, dir_name, str(iid), 'rgb_v', file_name)).resolve().__str__()):
                    flag = True
                    
                if not os.path.exists(Path(os.path.join(args.src_path, dir_name, str(iid), 'mask_v', file_name)).resolve().__str__()):
                    flag = True
                
                if flag:
                    tqdm.write(f"{dir_name} {frame_idx} not found")
                    continue
                            
                
                file_name = str(frame_idx) + ".png"

                if file_name.endswith(".png"):
                    # img_path
                    f.write(Path(os.path.join(args.src_path, dir_name, str(iid), 'rgb_v', file_name)).resolve().__str__())
                    f.write(',')

                    # robust_label_path
                    f.write(Path(os.path.join(args.src_path, dir_name, str(iid), 'mask_v', file_name)).resolve().__str__())
                    f.write(',')

                    # gbuffer_path
                    # TODO: Make G-Buffer here!
                    # g_buffer_dir = Path(args.dst_path, '..', 'g_buffer', dir_name, file_name + '.npz').resolve()
                    g_buffer_dir = Path(os.path.join(args.src_path, dir_name, str(iid), 'rgb_v', file_name))

                    # if not os.path.exists(g_buffer_dir):

                    #     # print(f"{g_buffer_dir} does not exist, calculating...")
                    #     #
                    #     # Create the directory if it doesn't exist
                    #     os.makedirs(os.path.dirname(g_buffer_dir), exist_ok=True)


                    #     data = {}

                    #     # Load RGB image
                    #     data['img'] = np.array(imageio.imread(os.path.join(args.src_path, dir_name, str(iid), 'rgb_v', file_name)))

                    #     # Zip the gbuffers to a npz file
                    #     scene_depth_filename = g_buffer_files[0][frame_idx-1]
                    #     scene_color_filename = g_buffer_files[1][frame_idx-1]
                    #     scene_gbufferA_filename = g_buffer_files[2][frame_idx-1]
                    #     scene_gbufferB_filename = g_buffer_files[3][frame_idx-1]
                    #     scene_gbufferC_filename = g_buffer_files[4][frame_idx-1]
                    #     scene_gbufferD_filename = g_buffer_files[5][frame_idx-1]

                    #     # Read from pngs
                    #     base_path = os.path.join(args.src_path, dir_name, str(iid), "gbuffer_v")

                    #     img_scene_depth = Image.open(os.path.join(base_path, scene_depth_filename))
                    #     data['depth'] = np.array(img_scene_depth)[:,:,0].astype(np.float32) / 255.0

                    #     img_albedo = Image.open(os.path.join(base_path, scene_color_filename))
                    #     data['albedo'] = np.array(img_albedo)[:, :, 0:3].astype(np.float32) / 255.0

                    #     img_normal = Image.open(os.path.join(base_path, scene_gbufferA_filename))
                    #     data['normal'] = np.array(img_normal)[:, :, 0:3].astype(np.float32) / 255.0

                    #     img_gbufferB = Image.open(os.path.join(base_path, scene_gbufferB_filename))
                    #     data['gbufferB'] = np.array(img_gbufferB)[:, :, 0:3].astype(np.float32) / 255.0

                    #     img_gbufferC = Image.open(os.path.join(base_path, scene_gbufferC_filename))
                    #     data['gbufferC'] = np.array(img_gbufferC)[:, :, 0:3].astype(np.float32) / 255.0

                    #     img_gbufferD = Image.open(os.path.join(base_path, scene_gbufferD_filename))
                    #     data['gbufferD'] = np.array(img_gbufferD)[:, :, 0:3].astype(np.float32) / 255.0

                    #     data['gbuffers'] = np.concatenate(
                    #         [
                    #             data['depth'][:, :, np.newaxis],
                    #             data['albedo'],
                    #             data['normal'],
                    #             data['gbufferB'],
                    #             data['gbufferC'],
                    #             data['gbufferD']
                    #         ], axis=2
                    #     )

                    #     del data['depth']
                    #     del data['albedo']
                    #     del data['normal']
                    #     del data['gbufferB']
                    #     del data['gbufferC']
                    #     del data['gbufferD']


                    #     # Ground truth label map
                    #     gtlabel_map = np.array(imageio.imread(os.path.join(args.src_path, dir_name, str(iid), 'mask_v', file_name)))
                    #     # shader_map = np.zeros((gtlabel_map.shape[0], gtlabel_map.shape[1], 12))

                    #     mask = np.zeros(shape=(gtlabel_map.shape[0], gtlabel_map.shape[1], 12))

                    #     for idx, color in enumerate(Carla.color2id.keys()):
                    #         mask[:, :, Carla.color2id[tuple(color)] ] += (gtlabel_map == color).all(axis=2)

                    #     data['shader'] = mask

                    #     np.savez(g_buffer_dir, **data)

                    f.write(g_buffer_dir.resolve().__str__())
                    f.write(',')

                    # gt_label_path
                    f.write(Path(os.path.join(args.src_path, dir_name, str(iid), 'mask_v', file_name)).resolve().__str__())
                    f.write('\n')

            # break # Comment this line to process all the directories
        f.close()

        # Then, Iterate through all g_buffers and calculate the means and stds

        # sum = np.zeros(shape=(720, 1280, 3), dtype=np.float32)
        # tmp_std = np.zeros(shape=(720, 1280, 3), dtype=np.float32)
        #
        # g_buffer_dir = Path(args.dst_path, '..', 'g_buffer').resolve()
        #
        # for filename in glob.iglob(os.path.join(g_buffer_dir, '**/*.npz'), recursive=True):
        #     data = np.load(filename)
        #     sum += data['gbuffers'][0]
        #     print(filename)
        #
        # mean = sum / len(list(glob.iglob(os.path.join(g_buffer_dir, '**/*.npz'), recursive=True)))
        #
        # for filename in glob.iglob(os.path.join(g_buffer_dir, '**/*.npz'), recursive=True):
        #     data = np.load(filename)
        #     tmp_std += (data['gbuffers'][0] - mean) ** 2
        #     print(filename)
        #
        # std = np.sqrt(tmp_std / len(list(glob.iglob(os.path.join(g_buffer_dir, '**/*.npz'), recursive=True))))

        # Save the mean and std
        data = {}
        data['g_m'] = 0.0
        data['g_s'] = 1.0
        # Mkdir if not exists
        os.makedirs(Path(__file__).parent / 'stats', exist_ok=True)

        np.savez(Path(__file__).parent / 'stats/carla_stats.npz', **data)



if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(description="Generate dataset file for matching.")
    parser.add_argument("--type", type=str, required=True, choices=["cityscape", "carla", "ACDC"], help="Dataset type.")
    parser.add_argument("--src_path", type=str, required=True, help="Path to source dataset.")
    parser.add_argument("--dst_path", type=str, required=True, help="Path to target file.")
    parser.add_argument("--acdc_dir", type=str, required=False, default="fog", choices=["fog", "night", "rain", "snow", "ref"])
    args = parser.parse_args()

    # Generate dataset file
    if args.type == "cityscape":
        dataset = CityScape()

    elif args.type == "carla":
        dataset = Carla()

    elif args.type == "ACDC":
        dataset = ACDC()

    else:
        raise Exception("Unknown dataset type.")

    dataset.generate_dataset_file(args)
