from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import lib

class Dataset(object):
    def __init__(self, data, batchSize, cuda, eval=False):
        self.src = data["src"]
        self.tgt = data["tgt"]
        self.trees = data["trees"]
        self.leafs = data["leafs"]
        assert(len(self.src) == len(self.tgt))
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = int(math.ceil(len(self.src)/batchSize)-1)
        self.eval = eval

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(lib.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, src_lengths = self._batchify(self.src[index*self.batchSize:(index + 1)*self.batchSize], include_lengths=True)
        # print "srcBatch: "
        # print srcBatch
        leafBatch, leaf_lengths = self._batchify(self.leafs[index * self.batchSize:(index + 1) * self.batchSize], include_lengths=True)
        srcTrees = self.trees[index * self.batchSize:(index + 1) * self.batchSize]
        tgtBatch = self._batchify(self.tgt[index * self.batchSize:(index + 1) * self.batchSize])

        indices = range(len(srcBatch))
        src_batch = zip(indices, srcBatch, leafBatch, leaf_lengths, srcTrees, tgtBatch)
        src_batch, src_lengths = zip(*sorted(zip(src_batch, src_lengths), key=lambda x: -x[1]))
        indices, srcBatch, leafBatch, leaf_lengths, srcTrees, tgtBatch = zip(*src_batch)
        # print "-srcBatch: "
        # print srcBatch

        # indices = range(len(leafBatch))
        # leaf_batch = zip(indices, leafBatch, srcBatch, tgtBatch, srcTrees, src_lengths)
        # leaf_batch, leaf_lengths = zip(*sorted(zip(leaf_batch, leaf_lengths), key=lambda x: -x[1]))
        # leaf_indices, leafBatch, srcBatch, tgtBatch_leaf, srcTrees, src_lengths = zip(*leaf_batch)

        tree_lengths = []
        for tree in srcTrees:
            l_c = tree.leaf_count()
            tree_lengths.append(l_c)
        # print "src_lengths: ", src_lengths
        # print "leaf_lengths: ", leaf_lengths
        # print "src_lengths_leaf: ", src_lengths_leaf

        def wrap(b):
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.eval)
            return b

        return (wrap(srcBatch), src_lengths), \
               (srcTrees, tree_lengths, (wrap(leafBatch), leaf_lengths)), \
               wrap(tgtBatch), \
               indices

        # return (wrap(srcBatch_src), src_lengths), \
        #        wrap(tgtBatch_src), \
        #        (srcTrees, tree_lengths , (wrap(leafBatch), leaf_lengths), (wrap(srcBatch), src_lengths)), \
        #        wrap(tgtBatch_leaf), \
        #        src_indices, \
        #        leaf_indices
        # (wrap(leafBatch), leaf_lengths), wrap(tgtBatch_leaf), \

    def __len__(self):
        return self.numBatches

    # def shuffle(self):
    #     data = list(zip(self.src, self.tgt, self.trees, self.leafs, self.pos))
    #     random.shuffle(data)
    #     self.src, self.tgt, self.trees, self.leafs, self.pos = zip(*data)
    #
    # def restore_pos(self, sents):
    #     sorted_sents = [None] * len(self.pos)
    #     for sent, idx in zip(sents, self.pos):
    #       sorted_sents[idx] = sent
    #     return sorted_sents
