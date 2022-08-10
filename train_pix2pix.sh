#/bin/sh
if [ -z "$1" ]
  then
    echo "No number of channels supplied"
    exit 1
fi
python3 train.py --dataroot ./datasets/depth --name depth_pix2pix_$1 --model pix2pix --input_nc $1 --output_nc $1 --display_id 0 --no_html --dataset_mode aligned
