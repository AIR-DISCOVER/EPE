# Usage: bash <task_name>.sh train/TEST
export CUDA_VISIBLE_DEVICES=2
python3 /home/gaoha/epe/code/epe/EPEExperiment.py\
        --log_dir /DATA_EDS/gaoha/datasets/Carla/NightLab/saved/nd22NC-Nagoya/logs\
         $1 /home/gaoha/epe/code/config/nd22NC-Nagoya.yaml
