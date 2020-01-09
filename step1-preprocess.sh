cd script/github
DATA_DIR=/workspace/dataset/
python2 python_process.py -train_portion 0.6 -dev_portion 0.2 \
--data_dir $DATA_DIR \
> log.python_process