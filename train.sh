#/bin/sh
python3 train.py --dataroot ./datasets/depth --name depth_cyclegan --model cycle_gan --input_nc 9 --output_nc 9 --display_id 0 --no_html --dataset_mode aligne