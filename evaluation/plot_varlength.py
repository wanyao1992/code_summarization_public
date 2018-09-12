import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['ps.useafm'] = True
# matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.gridspec as gridspec
import sys
import numpy as np
var_type = 'code'

def file2numpy(file):
    with open(file, 'r') as fi:
        return np.loadtxt(file)

def plot_f(code_length, s1, s2, s3, file, xlabel, ylabel):
    fontsize = 20
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.plot(code_length, s1, marker='^', linewidth=2.0, markersize=15, label='Hybrid2Seq+Attn+DRL')
    ax.plot(code_length, s2, marker='*', linewidth=2.0, markersize=15, label='Tree2Seq+Attn')
    ax.plot(code_length, s3, marker='o', linewidth=2.0, markersize=15, label='Seq2Seq+Attn')
    ax.legend(fontsize='16') #, loc='upper center'
    ax.tick_params(labelsize=20)

    # plt.show()
    fig.savefig(file)

if var_type == 'code':
    hybrid2seq_attn_a2c_var = file2numpy('hybrid2seq_attn_a2c_varcode.txt')
    seq2seq_attn_var = file2numpy('seq2seq_attn_varcode.txt')
    tree2seq_attn_var = file2numpy('hybrid2seq_attn_varcode.txt')
    code_length = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
elif var_type == 'comment':
    hybrid2seq_attn_a2c_var = file2numpy('hybrid2seq_attn_a2c_varcomment.txt')
    seq2seq_attn_var = file2numpy('seq2seq_attn_varcomment.txt')
    tree2seq_attn_var = file2numpy('hybrid2seq_attn_varcomment.txt')
    code_length = [0, 10, 20, 30, 40]

hybrid2seq_attn_a2c_var_b1 = hybrid2seq_attn_a2c_var[:,0].tolist()
seq2seq_attn_var_b1 = seq2seq_attn_var[:,0].tolist()
tree2seq_attn_var_b1 = tree2seq_attn_var[:,0].tolist()
# print(seq2seq_attn_varcode_b1)

hybrid2seq_attn_a2c_var_m = hybrid2seq_attn_a2c_var[:,4].tolist()
seq2seq_attn_var_m = seq2seq_attn_var[:,4].tolist()
tree2seq_attn_var_m = tree2seq_attn_var[:,4].tolist()
# print(seq2seq_attn_varcode_m)

hybrid2seq_attn_a2c_var_r = hybrid2seq_attn_a2c_var[:,5].tolist()
seq2seq_attn_var_r = seq2seq_attn_var[:,5].tolist()
tree2seq_attn_var_r = tree2seq_attn_var[:,5].tolist()

hybrid2seq_attn_a2c_var_c = hybrid2seq_attn_a2c_var[:,6].tolist()
seq2seq_attn_var_c = seq2seq_attn_var[:,6].tolist()
tree2seq_attn_var_c = tree2seq_attn_var[:,6].tolist()

if var_type == 'code':
    plot_f(code_length, hybrid2seq_attn_a2c_var_b1, tree2seq_attn_var_b1, seq2seq_attn_var_b1, 'varcode_bleu.pdf', 'Code length', 'BLEU')
    plot_f(code_length, hybrid2seq_attn_a2c_var_m, tree2seq_attn_var_m, seq2seq_attn_var_m, 'varcode_meteor.pdf', 'Code length', 'METEOR')
    plot_f(code_length, hybrid2seq_attn_a2c_var_r, tree2seq_attn_var_r, seq2seq_attn_var_r, 'varcode_rouge.pdf', 'Code length', 'ROUGE-L')
    plot_f(code_length, hybrid2seq_attn_a2c_var_c, tree2seq_attn_var_c, seq2seq_attn_var_c, 'varcode_cider.pdf', 'Code length', 'CIDER')
elif var_type == 'comment':
    plot_f(code_length, hybrid2seq_attn_a2c_var_b1, tree2seq_attn_var_b1, seq2seq_attn_var_b1, 'varcomment_bleu.pdf', 'Comment length', 'BLEU')
    plot_f(code_length, hybrid2seq_attn_a2c_var_m, tree2seq_attn_var_m, seq2seq_attn_var_m, 'varcomment_meteor.pdf', 'Comment length', 'METEOR')
    plot_f(code_length, hybrid2seq_attn_a2c_var_r, tree2seq_attn_var_r, seq2seq_attn_var_r, 'varcomment_rouge.pdf', 'Comment length', 'ROUGE-L')
    plot_f(code_length, hybrid2seq_attn_a2c_var_c, tree2seq_attn_var_c, seq2seq_attn_var_c, 'varcomment_cider.pdf', 'Comment length', 'CIDER')

# fig = plt.figure()
#
# xlabel = 'x'
# ylabel = 'y'
# fontsize = 20
# ax1 = plt.subplot(141)
# ax1.set_xlabel(xlabel, fontsize=fontsize)
# ax1.set_ylabel(ylabel, fontsize=fontsize)
#
# ax1.plot(code_length, hybrid2seq_attn_a2c_var_b1, marker='^', linewidth=2.0, markersize=15)
# ax1.plot(code_length, tree2seq_attn_var_b1, marker='*', linewidth=2.0, markersize=15)
# ax1.plot(code_length, seq2seq_attn_var_b1, marker='o', linewidth=2.0, markersize=15)
#
# ax1 = plt.subplot(142)
# ax1.set_xlabel(xlabel, fontsize=fontsize)
# ax1.set_ylabel(ylabel, fontsize=fontsize)
#
# ax1.plot(code_length, hybrid2seq_attn_a2c_var_b1, marker='^', linewidth=2.0, markersize=15)
# ax1.plot(code_length, tree2seq_attn_var_b1, marker='*', linewidth=2.0, markersize=15)
# ax1.plot(code_length, seq2seq_attn_var_b1, marker='o', linewidth=2.0, markersize=15)
#
# ax1 = plt.subplot(143)
# ax1.set_xlabel(xlabel, fontsize=fontsize)
# ax1.set_ylabel(ylabel, fontsize=fontsize)
#
# ax1.plot(code_length, hybrid2seq_attn_a2c_var_b1, marker='^', linewidth=2.0, markersize=15)
# ax1.plot(code_length, tree2seq_attn_var_b1, marker='*', linewidth=2.0, markersize=15)
# ax1.plot(code_length, seq2seq_attn_var_b1, marker='o', linewidth=2.0, markersize=15)
#
# ax1 = plt.subplot(144)
# ax1.set_xlabel(xlabel, fontsize=fontsize)
# ax1.set_ylabel(ylabel, fontsize=fontsize)
#
# ax1.plot(code_length, hybrid2seq_attn_a2c_var_b1, marker='^', linewidth=2.0, markersize=15)
# ax1.plot(code_length, tree2seq_attn_var_b1, marker='*', linewidth=2.0, markersize=15)
# ax1.plot(code_length, seq2seq_attn_var_b1, marker='o', linewidth=2.0, markersize=15)

# fig.savefig(file)
# fig.legend((l1, l2), ('Line 1', 'Line 2'), 'upper left')
# fig.legend((l3, l4), ('Line 3', 'Line 4'), 'upper right')
# plt.show()
# fig.savefig('ttt.pdf')