# Usage: bash <task_name>.sh train/TEST
export CUDA_VISIBLE_DEVICES=3
python3 /home/gaoha/epe/code/epe/EPEExperiment.py\
        --log_dir /home/gaoha/epe/saved_tasks/carla-gbuffer-night2acdc-night/logs\
         $1 /home/gaoha/epe/code/config/carla-gbuffer-night2acdc-night.yaml
