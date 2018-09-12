import os
import subprocess
import os.path
import sys

hostname = 'ccnt-ubuntu'

if hostname == 'ccnt-ubuntu':
    print(hostname)
    def preprocess():
        log = '/media/BACKUP/log/code_summarization/log.preprocess'
        if os.path.exists(log):
            os.system("rm -rf %s" % log)

        run = 'python preprocess.py ' \
              '-data_name github-python ' \
              '-train_src /media/BACKUP/ghproj_d/code_summarization/github-python/train/train0.60.20.2.code ' \
              '-train_tgt /media/BACKUP/ghproj_d/code_summarization/github-python/train/train0.60.20.2.comment ' \
              '-train_xe_src /media/BACKUP/ghproj_d/code_summarization/github-python/train/train0.60.20.2.code ' \
              '-train_xe_tgt /media/BACKUP/ghproj_d/code_summarization/github-python/train/train0.60.20.2.comment ' \
              '-train_pg_src /media/BACKUP/ghproj_d/code_summarization/github-python/train/train0.60.20.2.code ' \
              '-train_pg_tgt /media/BACKUP/ghproj_d/code_summarization/github-python/train/train0.60.20.2.comment ' \
              '-valid_src /media/BACKUP/ghproj_d/code_summarization/github-python/train/dev0.60.20.2.code ' \
              '-valid_tgt /media/BACKUP/ghproj_d/code_summarization/github-python/train/dev0.60.20.2.comment ' \
              '-test_src /media/BACKUP/ghproj_d/code_summarization/github-python/train/test0.60.20.2.code ' \
              '-test_tgt /media/BACKUP/ghproj_d/code_summarization/github-python/train/test0.60.20.2.comment ' \
              '-save_data /media/BACKUP/ghproj_d/code_summarization/github-python/train/processed_all ' \
              '> /media/BACKUP/log/code_summarization/log.preprocess'
        print(run)
        a = os.system(run)
        if a == 0:
            print("finished.")
        else:
            print("failed.")
            sys.exit()

    def train_a2c(start_reinforce, end_epoch, critic_pretrain_epochs, data_type, has_attn, gpus):
        run = 'python a2c-train.py ' \
              '-data /media/BACKUP/ghproj_d/code_summarization/github-python/train/processed_all.train.pt ' \
              '-save_dir /media/BACKUP/ghproj_d/code_summarization/github-python/result/ ' \
              '-embedding_w2v /media/BACKUP/ghproj_d/code_summarization/github-python/train/ ' \
              '-start_reinforce %s ' \
              '-end_epoch %s ' \
              '-critic_pretrain_epochs %s ' \
              '-data_type %s ' \
              '-has_attn %s ' \
              '-gpus %s ' \
              '> /home/wanyao/log/code_summarization/log.a2c-train_%s_%s_%s_%s_%s_g%s.test' \
              % (start_reinforce, end_epoch, critic_pretrain_epochs, data_type, has_attn, gpus,
                 start_reinforce, end_epoch, critic_pretrain_epochs, data_type, has_attn, gpus)
        print(run)
        a = os.system(run)
        if a == 0:
            print("finished.")
        else:
            print("failed.")
            sys.exit()

    def test_a2c(data_type, has_attn, gpus):
        run = 'python a2c-train.py ' \
              '-data /media/BACKUP/ghproj_d/code_summarization/github-python/train/processed_all.train.pt ' \
              '-load_from /media/BACKUP/ghproj_d/code_summarization/github-python/result/model_rf_hybrid_1_29_reinforce.pt ' \
              '-embedding_w2v /media/BACKUP/ghproj_d/code_summarization/github-python/train/ ' \
              '-eval -save_dir . ' \
              '-data_type %s ' \
              '-has_attn %s ' \
              '-gpus %s ' \
              '> /home/wanyao/log/code_summarization/log.a2c-test_%s_%s_%s' \
              % (data_type, has_attn, gpus, data_type, has_attn, gpus)
        print(run)
        a = os.system(run)
        if a == 0:
            print("finished.")
        else:
            print("failed.")
            sys.exit()

    if sys.argv[1] == 'preprocess':
        preprocess()

    if sys.argv[1] == 'train_a2c':
        train_a2c(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])

    if sys.argv[1] == 'test_a2c':
        test_a2c(sys.argv[2], sys.argv[3], sys.argv[4])