from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from meteor.meteor import Meteor
import numpy as np
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.gridspec as gridspec
import sys
import numpy as np

result_file_sorted = "/media/BACKUP/ghproj_d/code_summarization/github-python/result/model_rf_hybrid_1_30_reinforce.test.pred.varcode"
# result_file = "/media/BACKUP/ghproj_d/code_summarization/github-python/result/model_rf_hybrid_1_30_reinforce.test.pred"
# result_file = "/media/BACKUP/ghproj_d/code_summarization/github-python/result/model_rf_hybrid_1_10_pretrain.test.pred"
# result_file = "/media/BACKUP/ghproj_d/code_summarization/github-python/result_nonhybrid/model_rf_code_1_10_pretrain.test.pred"
result_file = "/media/BACKUP/ghproj_d/code_summarization/github-python/result/model_rf_text_1_10_pretrain.test.pred"

def get_dics():
    # res, gts, src = {}, {}, {}
    res, gts, src = [], [], []
    with open(result_file_sorted, 'r') as file:
        lines = file.readlines()

        for i, line in enumerate(lines):
            if i % 3 == 0:
                src.append(line.strip('\n').split(':')[2])
            elif i%3 == 1:
                # print '00: ', line.split(':')[0], line.split(':')[2]
                res.append(line.strip('\n').split(':')[2])
                # res = {b: [' '.join(pred_tokens_list[b])] for b in range(len(pred_tokens_list))}
            elif i%3 == 2:
                # print '11: ', line.split(':')[0], line.split(':')[2]
                gts.append(line.strip('\n').split(':')[2])
                # gts = {b: [' '.join(gold_tokens_list[b])] for b in range(len(gold_tokens_list))}

    tgt_codelength_dic, tgt_commentlength_dic = {}, {}
    for i in range(len(src)):
        # print('key: ', gts[i][0])
        tgt_codelength_dic[gts[i]] = len(src[i].split())
        tgt_commentlength_dic[gts[i]] = len(gts[i].split())

    return tgt_codelength_dic, tgt_commentlength_dic

def eval_filt(filt_type, res, gts, tgt_codelength_dic, tgt_commentlength_dic, filter_lower, filter_upper):
    # print('tgt_commentlength_dic: ')
    # print(tgt_commentlength_dic)
    # print('=====%s======%s=====' %(filter_lower, filter_upper))
    res_filtered, gts_filtered = [], []
    err_num = 0
    if filt_type == 'code':
        for i in range(len(gts)):
            try:
                if tgt_codelength_dic[gts[i]] > filter_lower and tgt_codelength_dic[gts[i]] < filter_upper:
                    res_filtered.append(res[i])
                    gts_filtered.append(gts[i])
            except:
                err_num += 1
    elif filt_type == 'comment':
        for i in range(len(gts)):
            try:
                if tgt_commentlength_dic[gts[i]] > filter_lower and tgt_commentlength_dic[gts[i]] < filter_upper:
                    res_filtered.append(res[i])
                    gts_filtered.append(gts[i])
            except:
                err_num += 1
    # print('err_num: ', err_num)
    return res_filtered, gts_filtered

def cal_metric(res_filtered, gts_filtered):
    gts, res = {}, {}
    for i in range(len(res_filtered)):
        gts[i] = [res_filtered[i]]
        res[i] = [gts_filtered[i]]
    score_Bleu, scores_Bleu = Bleu(4).compute_score(gts, res)

    # print("score_Bleu: "), score_Bleu
    # print("scores_Bleu: "), len(scores_Bleu)
    # print("Bleu_1: "), np.mean(scores_Bleu[0])
    # print("Bleu_2: "), np.mean(scores_Bleu[1])
    # print("Bleu_3: "), np.mean(scores_Bleu[2])
    # print("Bleu_4: "), np.mean(scores_Bleu[3])

    score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    # print("Meteor: "), score_Meteor

    score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
    # print("ROUGe: "), score_Rouge

    score_Cider, scores_Cider = Cider().compute_score(gts, res)
    # print("Cider: "), score_Cider
    return np.mean(scores_Bleu[0]), np.mean(scores_Bleu[1]), np.mean(scores_Bleu[2]), np.mean(scores_Bleu[3]), score_Meteor, score_Rouge, score_Cider

def plot_hist(dist, bin_lower, bin_upper, xlabel, ylabel, file):
    fontsize = 20
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    # fixed bin size
    bins = np.arange(bin_lower, bin_upper, 5)  # fixed bin size
    plt.xlim([0, max(dist)])
    ax.hist(dist, bins=bins, alpha=0.5)
    # ax.legend(fontsize='16')  # , loc='upper center'
    ax.tick_params(labelsize=20)
    # plt.xlabel(xlabel, fontsize=fontsize)
    # plt.ylabel(ylabel, fontsize=fontsize)
    # plt.show()
    # fig.savefig('test.pdf')
    fig.savefig(file, bbox_inches='tight')

def statistics(tgt_codelength_dic, tgt_commentlength_dic):
    codelength = tgt_codelength_dic.values()
    commentlength = tgt_commentlength_dic.values()
    print('commentlength: ', len(commentlength))
    print(commentlength)
    plot_hist(codelength, 0, 100, 'Code length', 'Count', '/home/wanyao/Dropbox/ghproj-py36/code_summarization/lib/visual/code_length_distribution.pdf')
    plot_hist(commentlength, 0, 50, 'Comment length', 'Count', '/home/wanyao/Dropbox/ghproj-py36/code_summarization/lib/visual/comment_length_distribution.pdf')

if __name__ == '__main__':
    tgt_codelength_dic, tgt_commentlength_dic = get_dics()
    statistics(tgt_codelength_dic, tgt_commentlength_dic)
    sys.exit()
    # evaluation
    # res, gts = {}, {}
    res, gts = [], []
    with open(result_file, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i % 2 == 0:
                # print '00: ', line.split(':')[0], line.split(':')[2]
                # res[int(line.strip('\n').split(':')[0])] = [line.strip('\n').split(':')[2]]
                res.append(line.strip('\n').split(':')[2])
                # res = {b: [' '.join(pred_tokens_list[b])] for b in range(len(pred_tokens_list))}
            elif i % 2 == 1:
                # print '11: ', line.split(':')[0], line.split(':')[2]
                # gts[int(line.strip('\n').split(':')[0])] = [line.strip('\n').split(':')[2]]
                gts.append(line.strip('\n').split(':')[2])
                # gts = {b: [' '.join(gold_tokens_list[b])] for b in range(len(gold_tokens_list))}

    filter_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # with open('hybrid2seq_attn_a2c_varcode.txt', 'w') as file:
    # with open('hybrid2seq_attn_varcode.txt', 'w') as file:
    with open('seq2seq_attn_varcode.txt', 'w') as file:
    # with open('tree2seq_attn_varcode.txt', 'w') as file:
        for i in range(len(filter_points)-1):
            # print('res: ')
            # print(res)
            res_filtered, gts_filtered = eval_filt('code', res, gts, tgt_codelength_dic, tgt_commentlength_dic, filter_points[i], filter_points[i+1])

            b1, b2, b3, b4, m, r, c = cal_metric(res_filtered, gts_filtered)
            file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (b1, b2, b3, b4, m, r, c))

    # filter_points = [0, 10, 20, 30, 40, 50]
    # with open('hybrid2seq_attn_a2c_varcomment.txt', 'w') as file:
    # with open('hybrid2seq_attn_varcomment.txt', 'w') as file:
    # with open('tree2seq_attn_varcomment.txt', 'w') as file:
    # with open('seq2seq_attn_varcomment.txt', 'w') as file:
    #     for i in range(len(filter_points) - 1):
    #         # print('res: ')
    #         # print(res)
    #         res_filtered, gts_filtered = eval_filt('comment', res, gts, tgt_codelength_dic, tgt_commentlength_dic,
    #                                                filter_points[i], filter_points[i + 1])
    #
    #         b1, b2, b3, b4, m, r, c = cal_metric(res_filtered, gts_filtered)
    #         file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (b1, b2, b3, b4, m, r, c))
