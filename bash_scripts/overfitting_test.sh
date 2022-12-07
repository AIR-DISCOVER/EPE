export CUDA_VISIBLE_DEVICES=0
python3 /home/gaomx/epe/EPE/epe/EPEExperiment.py\
        --test_save_dir=/home/gaomx/epe/EPE/saved_tasks/infer\
        --test_file_path=/home/gaomx/epe/EPE/data/file_lists/fake_dataset/Carla/file_overfitting_test.txt\
         $1 /home/gaomx/epe/EPE/config/overfitting.yaml