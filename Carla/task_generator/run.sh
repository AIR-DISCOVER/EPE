export CUDA_VISIBLE_DEVICES=6
python3 task_generator.py --src Day0923 --dst acdc-night
#python3 task_generator.py --src carla-gbuffer --dst acdc-snow
python3 task_generator.py --src Day0923 --dst acdc-rain
python3 task_generator.py --src Day0923 --dst acdc-fog
# python3 task_generator.py --src carla-night-town10 --dst seq1
# python3 task_generator.py --src carla-night-town10 --dst seq2
# python3 task_generator.py --src carla-night-town10 --dst seq3
# python3 task_generator.py --src carla-night-town10 --dst seq4
# python3 task_generator.py --src carla-night-town10 --dst seq5
# python3 task_generator.py --src carla-night-town10 --dst seq6
# python3 task_generator.py --src carla-gbuffer-night-light --dst night-depth-pspnet
# python3 task_generator.py --src carla-gbuffer-night-light --dst night-depth-refinenet
# python3 task_generator.py --src carla-gbuffer-night --dst night-depth-deeplab
# python3 task_generator.py --src carla-gbuffer-night --dst night-depth-pspnet
# python3 task_generator.py --src carla-gbuffer-night --dst night-depth-refinenet
#python3 task_generator.py --src carla-gbuffer- --dst acdc-snow
#python3 task_generator.py --src carla-gbuffer-rain --dst acdc-rain
#python3 task_generator.py --src carla-gbuffer-fog --dst acdc-fog
