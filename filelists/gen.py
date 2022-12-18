import os
import sys
import glob

def generate_cityscapes(root, save_path):
    f = open(save_path, 'w')
    for image in glob.glob(os.path.join(root, 'leftImg8bit', '**/*.png'), recursive=True):
        if 'test' in image or 'val' in image:
            continue
        image_rel_path = os.path.relpath(image, os.path.abspath(os.path.join(root, 'leftImg8bit')))
        label_rel_path = image_rel_path[:-15] + 'gtFine_trainLabelIds.png'
        image_path = os.path.abspath(os.path.join(root, 'leftImg8bit', image_rel_path))
        label_path = os.path.abspath(os.path.join(root, 'gtFine', label_rel_path))
        print(f'{image_path},{label_path}', file=f)
    f.close()

def generate_carla(root, save_path):
    f = open(save_path, 'w')
    for cate in os.listdir(root):
        times = [int(i) for i in os.listdir(os.path.join(root, cate))]
        if len(times) == 0:
            print(f'empty dir in {cate}')
            continue

        max_time = str(len(times))
        label_names = [i for i in sorted(os.listdir(os.path.join(root, cate, max_time, 'mask_v')), key=lambda i: int(i[:-4]))]
        for label in label_names:
            image_name = os.path.abspath(os.path.join(root, cate, max_time, 'rgb_v', label))
            label_name = os.path.abspath(os.path.join(root, cate, max_time, 'mask_v', label))
            gbuffer_names = [os.path.abspath(os.path.join(root, cate, max_time, 'gbuffer_v', label[:-4] + f'-{name}.png')) for name in ["GBufferA", "GBufferB", "GBufferC", "GBufferD", "SceneColor", "SceneDepth", "SSAO", "Velocity"]]
            flag = False
            for gbuffer_path in gbuffer_names:
                if not os.path.exists(gbuffer_path):
                    flag = True
                    break
            if flag:
                continue
            print(f'{image_name},{label_name},{image_name},{label_name}', file=f)
    f.close()

if __name__ == '__main__':
    '''
    python gen.py TYPE DATASET_ROOT FILELIST_SAVE_PATH
    TYPE: carla, cityscapes
    '''
    if sys.argv[1] == 'carla':
        generate_carla(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'cityscapes':
        generate_cityscapes(sys.argv[2], sys.argv[3])
    else:
        raise NotImplementedError
