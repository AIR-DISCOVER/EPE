# Usage: bash <task_name>.sh train/TEST
export CUDA_VISIBLE_DEVICES=0
python3 epe/EPEExperiment.py\
        --log_dir %%task_dir%%/logs\
         $1 %%task_yaml%%
