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
    replcae "+cu102" by "cu111"

# Config 
    config/tpy.yaml
    replace all of the absolute path by relative path

# Data
    real_dataset(CityScapes): data/real_dataset/CityScapes.txt
    fake_dataset: data/fake_dataset/file.txt

