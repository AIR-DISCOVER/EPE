export CUDA_VISIBLE_DEVICES=1
python3 /home/gaomx/epe/EPE/epe/EPEExperiment.py\
        --log_dir /home/gaomx/epe/EPE/saved_tasks/carla2cs/logs\
        --test_save_dir=/home/gaomx/epe/EPE/saved_tasks/train_infer\
        --test_file_path=/home/gaomx/epe/EPE/data/file_lists/fake_dataset/Carla/file_test.txt\
         $1 /home/gaomx/epe/EPE/config/calar2cs.yaml