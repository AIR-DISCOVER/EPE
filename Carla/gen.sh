### Pre-process data
# Generate file.txt for Carla
python3 ../Carla/generate_dataset_file.py --src_path /home/gaoha/epe/CarlaSampleDataset/ --dst_path /home/gaoha/epe/CarlaSampleDataset/file.txt --type carla

# Generate Crops for CityScapes dataset, save crops to out_dir/crop_cityscapes.npz
python3 ../epe/matching/feature_based/collect_crops.py cityscapes /home/gaoha/epe/Carla/cityscapes.txt --out_dir /home/gaoha/epe/Carla/

# Generate Crops for Carla
python3 ../epe/matching/feature_based/collect_crops.py carla /home/gaoha/epe/CarlaSampleDataset/file.txt --out_dir /home/gaoha/epe/Carla/

# Find matching
# Need: conda install -c pytorch faiss-gpu
python3 ../epe/matching/feature_based/find_knn.py /home/gaoha/epe/Carla/crop_carla.npz /home/gaoha/epe/Carla/crop_cityscapes.npz /home/gaoha/epe/Carla/matches.npz

# Filter
python3 ../epe/matching/filter.py /home/gaoha/epe/Carla/matches.npz /home/gaoha/epe/Carla/crop_carla.csv /home/gaoha/epe/Carla/crop_cityscapes.csv 1.0 /home/gaoha/epe/Carla/filtered_matches.csv

# Calc weights
python3 ../epe/matching/compute_weights.py /home/gaoha/epe/Carla/filtered_matches.csv 720 1280 /home/gaoha/epe/Carla/weights.npz

### Training
# Start Training
CUDA_VISIBLE_DEVICES=3 python3 /home/gaoha/epe/code/epe/EPEExperiment.py --log_dir /home/gaoha/epe/code/Carla/logs train /home/gaoha/epe/code/config/train_carla2cs.yaml

CUDA_VISIBLE_DEVICES=6 python3 /home/gaoha/epe/code/epe/EPEExperiment.py --log_dir /home/gaoha/epe/code/Carla/logs train /home/gaoha/epe/code/config/train_carla2cs_ie2.yaml

# Evaluate
CUDA_VISIBLE_DEVICES=7 python3 /home/gaoha/epe/code/epe/EPEExperiment.py --log_dir /home/gaoha/epe/code/Carla/logs TEST /home/gaoha/epe/code/config/test_carla2cs.yaml

# Evaluate
CUDA_VISIBLE_DEVICES=6 python3 /home/gaoha/epe/code/epe/EPEExperiment.py --log_dir /home/gaoha/epe/code/Carla/logs TEST /home/gaoha/epe/code/config/test_carla2kitti.yaml

### Visualize
# Visualize samples
CUDA_VISIBLE_DEVICES=3 python3 /home/gaoha/epe/code/utils-c7w/visualize.py