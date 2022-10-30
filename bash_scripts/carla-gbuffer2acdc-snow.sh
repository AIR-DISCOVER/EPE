# Usage: bash <task_name>.sh train/TEST
export CUDA_VISIBLE_DEVICES=0
python3 /home/gaoha/epe/code/epe/EPEExperiment.py\
        --log_dir /home/gaoha/epe/saved_tasks/carla-gbuffer2acdc-snow/logs\
         $1 /home/gaoha/epe/code/config/carla-gbuffer2acdc-snow.yaml
         --uuid Day0923