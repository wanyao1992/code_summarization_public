from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from meteor.meteor import Meteor
import numpy as np
import sys
result_file = "/home/wanyao/BACKUP/ghproj_d/code_summarization/github-python/result_nonhybrid/model_rf_code_1_10_pretrain.test.pred"
# result_file = "/home/wanyao/BACKUP/ghproj_d/code_summarization/github-python/result/model_rf_text_1_10_pretrain.test.pred"

with open(result_file, 'r') as file:
    lines = file.readlines()
    res, gts = {}, {}
    for i, line in enumerate(lines):
        if i%2 == 0:
            # print '00: ', line.split(':')[0], line.split(':')[2]
            res[int(line.strip('\n').split(':')[0])] = [line.strip('\n').split(':')[2]]
            # res = {b: [' '.join(pred_tokens_list[b])] for b in range(len(pred_tokens_list))}
        elif i%2 == 1:
            # print '11: ', line.split(':')[0], line.split(':')[2]
            gts[int(line.strip('\n').split(':')[0])] = [line.strip('\n').split(':')[2]]
            # gts = {b: [' '.join(gold_tokens_list[b])] for b in range(len(gold_tokens_list))}

score_Bleu, scores_Bleu = Bleu(4).compute_score(gts, res)

print("score_Bleu: "), score_Bleu
print("scores_Bleu: "), len(scores_Bleu)
print("Bleu_1: "), np.mean(scores_Bleu[0])
print("Bleu_2: "), np.mean(scores_Bleu[1])
print("Bleu_3: "), np.mean(scores_Bleu[2])
print("Bleu_4: "), np.mean(scores_Bleu[3])

score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
print("Meteor: "), score_Meteor

score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
print("ROUGe: "), score_Rouge

score_Cider, scores_Cider = Cider().compute_score(gts, res)
print("Cider: "), score_Cider