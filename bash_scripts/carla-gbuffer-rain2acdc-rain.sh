# Usage: bash <task_name>.sh train/TEST
export CUDA_VISIBLE_DEVICES=1
python3 /home/gaoha/epe/code/epe/EPEExperiment.py\
        --log_dir /home/gaoha/epe/saved_tasks/carla-gbuffer-rain2acdc-rain/logs\
         $1 /home/gaoha/epe/code/config/carla-gbuffer-rain2acdc-rain.yaml
