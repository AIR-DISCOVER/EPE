### Pre-process data
# Generate file.txt for Carla
python3 Carla/generate_dataset_file.py --src_path data/file_lists/fake_dataset/Carla/ --dst_path data/file_lists/fake_dataset/Carla/file.txt --type carla

# Generate Crops for CityScapes dataset, save crops to out_dir/crop_cityscapes.npz
python3 epe/matching/feature_based/collect_crops.py cityscapes data/file_lists/real_dataset/CityScapes.txt --out_dir carla2cs-new

# Generate Crops for Carla
python3 epe/matching/feature_based/collect_crops.py carla data/file_lists/fake_dataset/Carla/file.txt --out_dir carla2cs-new

# Find matching
# Need: conda install -c pytorch faiss-gpu
python3 epe/matching/feature_based/find_knn.py carla2cs-new/crop_carla.npz carla2cs-new/crop_cityscapes.npz carla2cs-new/matches.npz

# Filter
python3 epe/matching/filter.py carla2cs-new/matches.npz carla2cs-new/crop_carla.csv carla2cs-new/crop_cityscapes.csv 1.0 carla2cs-new/filtered_matches.csv

# Calc weights
python3 epe/matching/compute_weights.py carla2cs-new/filtered_matches.csv 720 1280 carla2cs-new/weights.npz

# ### Training
# # Start Training
# CUDA_VISIBLE_DEVICES=3 python3 epe/EPEExperiment.py --log_dir Carla/logs train config/train_carla2cs.yaml

# CUDA_VISIBLE_DEVICES=6 python3 epe/EPEExperiment.py --log_dir Carla/logs train config/train_carla2cs_ie2.yaml

# # Evaluate
# CUDA_VISIBLE_DEVICES=7 python3 epe/EPEExperiment.py --log_dir Carla/logs TEST config/test_carla2cs.yaml

# # Evaluate
# CUDA_VISIBLE_DEVICES=6 python3 epe/EPEExperiment.py --log_dir Carla/logs TEST config/test_carla2kitti.yaml

# ### Visualize
# # Visualize samples
# CUDA_VISIBLE_DEVICES=3 python3 utils-c7w/visualize.py