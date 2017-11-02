LOG="log/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"
nohup python train_pose.py --gpu 0 1 --train_dir /home/hypan/data/coco/filelist/train2014.txt /home/hypan/data/coco/masklist/train2014.txt /home/hypan/data/coco/json/train2014.json --val_dir /home/hypan/data/coco/filelist/val2014.txt /home/hypan/data/coco/masklist/val2014.txt /home/hypan/data/coco/json/val2014.json --config config.yml
