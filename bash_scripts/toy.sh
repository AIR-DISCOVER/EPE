# Usage: bash <task_name>.sh train/TEST
export CUDA_VISIBLE_DEVICES=1
python3 epe/EPEExperiment.py\
        --log_dir saved_tasks/carla-test/logs\
         $1 config/toy.yaml ${@:2}