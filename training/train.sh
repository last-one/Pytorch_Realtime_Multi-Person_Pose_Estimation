export PYTHONUNBUFFERED="True"
LOG="log/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"
python train_pose.py --gpu 0 1 --train_dir /home/code/panhongyu/datasets/coco/filelist/train2017.txt /home/code/panhongyu/datasets/coco/masklist/train2017.txt /home/code/panhongyu/datasets/coco/json/train2017.json --val_dir /home/code/panhongyu/datasets/coco/filelist/val2017.txt /home/code/panhongyu/datasets/coco/masklist/val2017.txt /home/code/panhongyu/datasets/coco/json/val2017.json --config config.yml > $LOG
