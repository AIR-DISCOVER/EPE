## Conda Env
# For V100
    conda create -n epe python=3.8
    conda activate epe
    conda install scikit-image
    conda install imageio
    pip install tqdm
    pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    pip install Ipython kornia lpips
# For 3090
    Please replcae "+cu102" in "For V100" by "+cu111"

# Dataset
    File example：
    real_dataset(CityScapes): data/file_lists/real_dataset/CityScapes.txt
    fake_dataset: data/file_lists/fake_dataset/Carla/file.txt

# Processing
    sh Carla/gen.sh
    Generates matching and file list for a new dataset

# Config 
    config/toy.yaml
    Please adjust real_dataset and fake_dataset in config while when changing dataset.

# Run
    sh bash_scripts/toy.sh train