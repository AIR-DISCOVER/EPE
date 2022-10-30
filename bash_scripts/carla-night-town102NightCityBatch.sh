# Usage: bash <task_name>.sh train/TEST
export CUDA_VISIBLE_DEVICES=0
/home/gaoha/.conda/envs/epe/bin/python /home/gaoha/epe/code/epe/EPEExperiment.py\
        --log_dir /home/gaoha/epe/saved_tasks/carla-night-town102NightCity/logs\
         $@ /home/gaoha/epe/batch_eval/eval.yaml
