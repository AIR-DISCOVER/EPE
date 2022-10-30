# Usage: bash <task_name>.sh train/TEST
export CUDA_VISIBLE_DEVICES=6
python3 /home/gaoha/epe/code/epe/EPEExperiment.py\
        --log_dir /DATA_EDS/gaoha/datasets/Carla/saved_tasks/Day09232bdd100k/logs\
         $1 /home/gaoha/epe/code/config/Day09232bdd100k.yaml
