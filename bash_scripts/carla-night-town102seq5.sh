# Usage: bash <task_name>.sh train/TEST
export CUDA_VISIBLE_DEVICES=0
python3 /home/gaoha/epe/code/epe/EPEExperiment.py\
        --log_dir /home/gaoha/epe/saved_tasks/carla-night-town102seq5/logs\
         $1 /home/gaoha/epe/code/config/carla-night-town102seq5.yaml
