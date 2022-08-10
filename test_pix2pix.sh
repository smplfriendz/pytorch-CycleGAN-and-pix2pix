#/bin/sh
if [ -z "$1" ]
  then
    echo "No number of channels supplied"
    exit 1
fi
python3 test.py --dataroot ./datasets/depth --name depth_pix2pix_$1 --model pix2pix --input_nc $1 --output_nc $1  --dataset_mode aligned