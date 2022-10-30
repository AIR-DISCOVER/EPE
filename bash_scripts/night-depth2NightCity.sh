# Usage: bash <task_name>.sh train/TEST
export CUDA_VISIBLE_DEVICES=7
python3 /home/gaoha/epe/code/epe/EPEExperiment.py\
        --log_dir /DATA_EDS/gaoha/datasets/Carla/NightDepth/saved/night-depth2NightCity/logs\
         $1 /home/gaoha/epe/code/config/night-depth2NightCity.yaml
