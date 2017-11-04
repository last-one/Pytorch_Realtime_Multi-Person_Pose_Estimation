LOG="log/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"
python train_pose.py --gpu 0 1 --train_dir /home/hypan/data/coco/filelist/train2017.txt /home/hypan/data/coco/masklist/train2017.txt /home/hypan/data/coco/json/train2017.json --val_dir /home/hypan/data/coco/filelist/val2017.txt /home/hypan/data/coco/masklist/val2017.txt /home/hypan/data/coco/json/val2017.json --config config.yml > $LOG
