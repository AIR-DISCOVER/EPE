# Usage: bash <task_name>.sh train/TEST
export CUDA_VISIBLE_DEVICES=2
python3 /home/gaoha/epe/code/epe/EPEExperiment.py\
        --log_dir /home/gaoha/epe/saved_tasks/carla-gbuffer-fog2acdc-fog/logs\
         $1 /home/gaoha/epe/code/config/carla-gbuffer-fog2acdc-fog.yaml
