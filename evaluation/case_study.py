from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from meteor.meteor import Meteor
import numpy as np
import sys

result_file_text_0_10 = "/home/wanyao/BACKUP/ghproj_d/code_summarization/github-python/result/model_rf_text_0_10_pretrain.test.pred"
result_file_text_1_10 = "/home/wanyao/BACKUP/ghproj_d/code_summarization/github-python/result/model_rf_text_1_10_pretrain.test.pred"
result_file_code_1_10 = "/home/wanyao/BACKUP/ghproj_d/code_summarization/github-python/result_nonhybrid/model_rf_code_1_10_pretrain.test.pred"
result_file_hybrid_1_10 = "/home/wanyao/BACKUP/ghproj_d/code_summarization/github-python/result/model_rf_hybrid_1_10_pretrain.test.pred"
result_file_hybrid_1_30 = "/home/wanyao/BACKUP/ghproj_d/code_summarization/github-python/result/model_rf_hybrid_1_30_reinforce.test.pred"

def process_result_file(result_file):
    res, gts = {}, {}
    with open(result_file, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i%2 == 0:
                # print '00: ', line.split(':')[0], line.split(':')[2]
                res[int(line.strip('\n').split(':')[0])] = [line.strip('\n').split(':')[2]]
                # res = {b: [' '.join(pred_tokens_list[b])] for b in range(len(pred_tokens_list))}
            elif i%2 == 1:
                # print '11: ', line.split(':')[0], line.split(':')[2]
                gts[int(line.strip('\n').split(':')[0])] = [line.strip('\n').split(':')[2]]
                # gts = {b: [' '.join(gold_tokens_list[b])] for b in range(len(gold_tokens_list))}
    return res, gts

res_text_0_10, gts = process_result_file(result_file_text_0_10)
res_text_1_10, gts = process_result_file(result_file_text_1_10)
res_code_1_10, gts = process_result_file(result_file_code_1_10)
res_hybrid_1_10, gts = process_result_file(result_file_hybrid_1_10)
res_hybrid_1_30, gts = process_result_file(result_file_hybrid_1_30)

# test_code_filepath = "/home/wanyao/BACKUP/ghproj_d/code_summarization/github-python/train/test0.60.20.2.code"
# test_tgt_filepath = "/home/wanyao/BACKUP/ghproj_d/code_summarization/github-python/train/test0.60.20.2.comment"
#
# comment2code_dic = {}
# with open(test_code_filepath, 'r') as test_code_file:
#     with open(test_tgt_filepath, 'r') as test_tgt_file:
#         test_codes = test_code_file.readlines()
#         test_tgts = test_tgt_file.readlines()
#         for i in range(len(test_codes)):
#             comment2code_dic[test_tgts[i].strip('\n')] = test_codes[i].strip('\n')

def get_bleu_dic(res, gts):
    # find the sentence with largest bleu
    bleu_dic = {}
    for k in range(len(res)):
        r, g = {}, {}
        r[0] = res[k]
        g[0] = gts[k]
        score_Bleu, scores_Bleu = Bleu(4).compute_score(r, g)

        # print "Bleu_1: ", np.mean(scores_Bleu[0])
        bleu_dic[k] = np.mean(scores_Bleu[0])

    return bleu_dic

bleu_dic_text_0_10 = get_bleu_dic(res_text_0_10, gts)
bleu_dic_text_1_10 = get_bleu_dic(res_text_1_10, gts)
bleu_dic_code_1_10 = get_bleu_dic(res_code_1_10, gts)
bleu_dic_hybrid_1_10 = get_bleu_dic(res_hybrid_1_10, gts)
bleu_dic_hybrid_1_30 = get_bleu_dic(res_hybrid_1_30, gts)

def output(i, outfile):
    # print "code: ", comment2code_dic[gts[i][0]]
    outfile.write("====%s: %s ==== \n " % (k, bleu_dic_hybrid_1_30[k]))
    outfile.write("gts: %s \n " % gts[i])
    outfile.write("pre1: %s \n " % res_text_0_10[i])
    outfile.write("pre2: %s \n " % res_text_1_10[i])
    outfile.write("pre3: %s \n " % res_code_1_10[i])
    outfile.write("pre4: %s \n " % res_hybrid_1_10[i])
    outfile.write("pre5: %s \n " % res_hybrid_1_30[i])

out_filepath = '/home/wanyao/www/Dropbox/ghproj/code_summarization/evaluation/casestudy.out.bad'
with open(out_filepath, 'w') as outfile:
    for k in range(len(bleu_dic_text_0_10)):
        # if bleu_dic_text_0_10[k] < bleu_dic_text_1_10[k] and bleu_dic_text_1_10[k] < bleu_dic_code_1_10[k] and bleu_dic_code_1_10[k] < bleu_dic_hybrid_1_10[k] and bleu_dic_hybrid_1_10[k] < bleu_dic_hybrid_1_30[k]:
        if bleu_dic_text_0_10[k] < bleu_dic_text_1_10[k] and bleu_dic_text_1_10[k] < bleu_dic_code_1_10[k] and bleu_dic_code_1_10[k] < bleu_dic_hybrid_1_10[k] and bleu_dic_hybrid_1_10[k] < bleu_dic_hybrid_1_30[k]:
            if bleu_dic_hybrid_1_30[k] > 0.5 and bleu_dic_hybrid_1_30[k] < 0.9:
                output(k, outfile)

# bleu_dic_sorted= sorted(bleu_dic.iteritems(), key=lambda d:d[1], reverse = True)
#
# for b_d in bleu_dic_sorted[0:100]:
#     print b_d[1]
#     print res[b_d[0]]
#     print gts[b_d[0]]
#     print "===="