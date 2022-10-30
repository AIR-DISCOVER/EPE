# Usage: bash <task_name>.sh train/TEST
export CUDA_VISIBLE_DEVICES=2
python3 /home/gaoha/epe/code/epe/EPEExperiment.py\
        --log_dir /DATA_EDS/gaoha/datasets/Carla/saved_tasks/Day09232acdc-rain/logs\
         $1 /home/gaoha/epe/code/config/Day09232acdc-rain.yaml
