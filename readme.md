**Note: This repository is no longer maintained now. If you are interested in the deep learning for program analysis (e.g., code summarization, code retrieval, code completion and type inference), please refer to our new project NaturalCC (https://github.com/CGCL-codes/naturalcc).**

## Requirement
This repos is developed based on the environment of:
- Python 2.7
- PyTorch 0.2

However, now it is hard to build a environment meeting both.
There is a workaround without installing python2, if you use `anaconda` or `miniconda`, you can create a virtual environment by 
```
conda create -n py2 python=2.7.
```

Then you can install the pytorch0.2 by 

```
conda install pytorch=0.2.0 cuda90 -c pytorch
```
Before that you should be aware of your cuda version if you want to use GPU.

```
nvcc --version
```

## (Update guideline) Follow the steps using script.
After cloning the project, use `chmod +x file.sh` for each step to give them permission.

`step0-start_envirionment.sh` 
Build the environment.

`step1-preproces.sh`
Preprocess the raw data.

`step2-prepare_training.sh`
You should specify the DATA_DIR to the dataset (absolute).
The number -1s are just placeholder.

`step3-training.sh`
Start training, you should specify the `--data_dir` (same as above).
You'd better specify your log path in source code `run.py`.

`step4-testing.sh`
Start testing, you should specify the `--data_dir`.
The number -1s are just placeholder.
You'd better specify your log path in source code `run.py`

After testing, the results are in the `dataset/result` folder.

## Data folder structure
/media/BACKUP/ghproj_d/code_summarization/github-python/ is the folder to save all the data in this project, please replace it to your own folder.
The data files are organized as follows in my computer:

|- /media/BACKUP/ghproj_d/code_summarization/github-python

|--original (used to save the raw data)

|----data_ps.declbodies  data_ps.descriptions

|--processed (used to save the preprocessed data)

|----all.code  all.comment

|--result (used to save the results)

|--train (get the data files before training)

You need to get these files before you starting to train our model.
Here I put the original folder in the dataset foler of this project. You'd better copy them to your own folder.

## Data preprocess
```
cd script/github
```

```
python python_process.py -train_portion 0.6 -dev_portion 0.2 > log.python_process
```
## Training
Back to the projector folder
```
cd ../..
```

### Get the data for training
```
python run.py preprocess
```

### Training
```
python run.py train_a2c 10 30 10 hybrid 1 0
```

### Testing
```
python run.py test_a2c hybrid 1 0
```



## TODO
- To build the AST, on the data preprocessing, I parse the AST into a json and then parse the json into AST on training. This kind of approach is not elegant.
- On training, I don't know how to batchify the ASTs, so I have to put the ASTs into a list and encode them one by one. It's unefficient, making the training of one epoch takes about 2-3 hours. Please let me know if you have a better way to accelerate this process.
- On the encoder side, I am working on applying Tree-CNN and GraphCNN to represent the code in a better way.
- On the decoder side, GAN network will also be considered for the code summarization task.

## Acknowledgement
This repos is based on https://github.com/khanhptnk/bandit-nmt

Please cite our paper if you use this repos.

**Bibtex:**<br />
@Inproceedings{wan2018improving,<br />
  title={Improving automatic source code summarization via deep reinforcement learning},<br />
  author={Wan, Yao and Zhao, Zhou and Yang, Min and Xu, Guandong and Ying, Haochao and Wu, Jian and Yu, Philip S},<br />
  booktitle={Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering}<br />
  pages={397--407},<br />
  year={2018},<br />
  organization={ACM}<br />
}<br />
