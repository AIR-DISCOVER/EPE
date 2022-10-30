# Usage: bash <task_name>.sh train/TEST
export CUDA_VISIBLE_DEVICES=3
python3 /home/gaoha/epe/code/epe/EPEExperiment.py\
        --log_dir /home/gaoha/epe/saved_tasks/carla-night-town102seq3/logs\
         $1 /home/gaoha/epe/code/config/carla-night-town102seq3.yaml
