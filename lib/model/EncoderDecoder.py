import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import gensim
import numpy as np
import lib
import sys
import re

class Encoder_W2V(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions

        super(Encoder_W2V, self).__init__()
        # self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=lib.Constants.PAD)
        self.embeddings = gensim.models.Word2Vec.load(opt.embedding_w2v + 'processed_all.train_xe.code.gz')

        self.rnn = nn.LSTM(opt.word_vec_size, self.hidden_size,  num_layers=opt.layers, dropout=opt.dropout, bidirectional=opt.brnn)
        self.dicts = dicts
        self.opt = opt

    def embedding(self, input):
        emb = []
        for i in range(input.shape[0]):
            emb_row = []
            for w in self.dicts.convertToLabels(input[i].tolist(), lib.Constants.UNK_WORD):
                try:
                    emb_row.append(self.embeddings.wv[w].astype(float))
                except:
                    emb_row.append(np.zeros((self.opt.word_vec_size), dtype=float))
            emb.append(emb_row)
        emb = Variable(torch.Tensor(emb))
        if self.opt.gpus:
            emb = emb.cuda()
        # print "decoder-emb: "
        # print emb
        return emb

    def forward(self, inputs, hidden=None):
        input = inputs[0].data.cpu().numpy()
        emb = self.embedding(input)
        emb = pack(emb, inputs[1])
        outputs, hidden_t = self.rnn(emb, hidden)
        outputs = unpack(outputs)[0]
        return hidden_t, outputs

class Encoder(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=lib.Constants.PAD)

        self.rnn = nn.LSTM(opt.word_vec_size, self.hidden_size,  num_layers=opt.layers, dropout=opt.dropout, bidirectional=opt.brnn)
        self.dicts = dicts
        self.opt = opt

    def forward(self, inputs, hidden=None):
        emb = pack(self.word_lut(inputs[0]), inputs[1])

        outputs, hidden_t = self.rnn(emb, hidden)
        outputs = unpack(outputs)[0]
        return hidden_t, outputs

class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, inputs, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(inputs, (h_0[i], c_0[i]))
            inputs = h_1_i
            if i != self.num_layers:
                inputs = self.dropout(inputs)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return inputs, (h_1, c_1)

class BinaryTreeLeafModule(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeLeafModule, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.cx = nn.Linear(self.in_dim, self.mem_dim)
        self.ox = nn.Linear(self.in_dim, self.mem_dim)
        if self.cudaFlag:
            self.cx = self.cx.cuda()
            self.ox = self.ox.cuda()

    def forward(self, input):
        c = self.cx(input)
        o = F.sigmoid(self.ox(input))
        h = o * F.tanh(c)
        return h, (c, h)

class BinaryTreeComposer(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim, gate_output=False):
        super(BinaryTreeComposer, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.gate_output = gate_output
        def new_gate():
            lh = nn.Linear(self.mem_dim, self.mem_dim)
            rh = nn.Linear(self.mem_dim, self.mem_dim)
            return lh, rh

        self.ilh, self.irh = new_gate()
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()
        if self.cudaFlag:
            self.ilh = self.ilh.cuda()
            self.irh = self.irh.cuda()
            self.lflh = self.lflh.cuda()
            self.lfrh = self.lfrh.cuda()
            self.rflh = self.rflh.cuda()
            self.rfrh = self.rfrh.cuda()
            self.ulh = self.ulh.cuda()
            self.urh = self.urh.cuda()

        if self.gate_output:
            self.olh, self.orh = new_gate()
            if self.cudaFlag:
                self.olh = self.olh.cuda()
                self.orh = self.orh.cuda()

    def forward(self, lc, lh , rc, rh):
        i = F.sigmoid(self.ilh(lh) + self.irh(rh))
        lf = F.sigmoid(self.lflh(lh) + self.lfrh(rh))
        rf = F.sigmoid(self.rflh(lh) + self.rfrh(rh))
        update = F.tanh(self.ulh(lh) + self.urh(rh))
        c =  i* update + lf*lc + rf*rc
        if self.gate_output:
            o = F.sigmoid(self.olh(lh) + self.orh(rh))
            h = o*F.tanh(c)
        else:
            h = F.tanh(c)
        return c, h

class TreeEncoder_W2V(nn.Module):
    def __init__(self, opt, dicts):
        super(TreeEncoder_W2V, self).__init__()
        self.layers = opt.layers
        self.opt = opt
        self.dicts = dicts
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        self.embeddings = gensim.models.Word2Vec.load(opt.embedding_w2v + 'processed_all.train_xe.code.gz')
        # self.embeddings = Embeddings(opt, dicts)
        self.input_size = self.opt.word_vec_size #self.embeddings.embedding_size #100

        if len(self.opt.gpus) >= 1:
            self.cudaFlag = True
        else:
            self.cudaFlag = False

        self.leaf_module = BinaryTreeLeafModule(self.cudaFlag, self.input_size, self.hidden_size)
        self.composer = BinaryTreeComposer(self.cudaFlag, self.input_size, self.hidden_size)

    def forward(self, tree, lengths):
        if not tree.children:
            try:
                node = torch.Tensor(self.embeddings.wv[tree.content]).unsqueeze(0)
            except:
                node = torch.zeros(1, self.input_size)
            if self.cudaFlag:
                node = node.cuda()
            # node = self.embeddings(Variable(torch.LongTensor([self.dicts.lookup(tree.content, onmt.Constants.UNK)]).unsqueeze(1)).cuda())
            # node.data.squeeze_(1)
            # print "node: ", node.size()
            # print node
            output, state = self.leaf_module.forward(Variable(node, requires_grad=True))

        elif tree.children:
            # for idx in xrange(tree.num_children):
            lo, (lc, lh) = self.forward(tree.children[0], lengths)
            ro, (rc, rh) = self.forward(tree.children[1], lengths)
            # lc, lh, lo, rc, rh, ro = self.get_child_state(tree)
            state = self.composer.forward(lc, lh, rc, rh)

            output = torch.cat([lo, ro])
            # del lc, lh, lo, rc, rh, ro
            if not tree.parent:
                # max_length = int(torch.max(lengths.data))
                max_length = np.max(lengths)
                output.data.unsqueeze_(1)
                supl = max_length - output.size()[0]
                if supl > 0:
                    output.data = torch.cat([output.data, torch.zeros((supl, output.size()[1], output.size()[2])).cuda()], 0)

                state[0].data.unsqueeze_(1)
                state[1].data.unsqueeze_(1)
        return output, state

    # def get_child_state(self, tree):
    #     lc, lh = tree.children[0].state
    #     lo = tree.children[0].output
    #     rc, rh = tree.children[1].state
    #     ro = tree.children[1].output
    #     return lc, lh, lo, rc, rh, ro

class TreeEncoder(nn.Module):
    def __init__(self, opt, dicts):
        super(TreeEncoder, self).__init__()
        self.layers = opt.layers
        self.opt = opt
        self.dicts = dicts
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        # self.embeddings = gensim.models.Word2Vec.load(opt.embedding_w2v + 'processed_all.train_xe.code.gz')
        self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=lib.Constants.PAD)
        self.input_size = self.opt.word_vec_size #self.embeddings.embedding_size #100

        if len(self.opt.gpus) >= 1:
            self.cudaFlag = True
        else:
            self.cudaFlag = False

        self.leaf_module = BinaryTreeLeafModule(self.cudaFlag, self.input_size, self.hidden_size)
        self.composer = BinaryTreeComposer(self.cudaFlag, self.input_size, self.hidden_size)

    def forward(self, tree, lengths):
        if not tree.children:
            # try:
            #     node = torch.Tensor(self.embeddings.wv[tree.content]).unsqueeze(0)
            # except:
            #     node = torch.zeros(1, self.input_size)
            # if self.cudaFlag:
            #     node = node.cuda()
            node = self.word_lut(Variable(torch.LongTensor([self.dicts.lookup(tree.content, lib.Constants.UNK)])).cuda())

            output, state = self.leaf_module.forward(node) #  Variable(node, requires_grad=True)

        elif tree.children:
            # for idx in xrange(tree.num_children):
            lo, (lc, lh) = self.forward(tree.children[0], lengths)
            ro, (rc, rh) = self.forward(tree.children[1], lengths)
            # lc, lh, lo, rc, rh, ro = self.get_child_state(tree)
            state = self.composer.forward(lc, lh, rc, rh)

            output = torch.cat([lo, ro])
            # del lc, lh, lo, rc, rh, ro
            if not tree.parent:
                # max_length = int(torch.max(lengths.data))
                max_length = np.max(lengths)
                output.data.unsqueeze_(1)
                supl = max_length - output.size()[0]
                if supl > 0:
                    output.data = torch.cat([output.data, torch.zeros((supl, output.size()[1], output.size()[2])).cuda()], 0)

                state[0].data.unsqueeze_(1)
                state[1].data.unsqueeze_(1)
        return output, state

    # def get_child_state(self, tree):
    #     lc, lh = tree.children[0].state
    #     lo = tree.children[0].output
    #     rc, rh = tree.children[1].state
    #     ro = tree.children[1].output
    #     return lc, lh, lo, rc, rh, ro

class HybridEncoder(nn.Module):
    def __init__(self, opt, dicts):
        super(HybridEncoder, self).__init__()
        self.layers = opt.layers
        self.opt = opt
        self.dicts = dicts
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        # self.embeddings = gensim.models.Word2Vec.load(opt.embedding_w2v + 'processed_all.train_xe.code.gz')
        self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=lib.Constants.PAD)
        self.input_size = self.opt.word_vec_size #self.embeddings.embedding_size #100

        if len(self.opt.gpus) >= 1:
            self.cudaFlag = True
        else:
            self.cudaFlag = False

        self.leaf_module = BinaryTreeLeafModule(self.cudaFlag, self.input_size, self.hidden_size)
        self.composer = BinaryTreeComposer(self.cudaFlag, self.input_size, self.hidden_size)

    def forward(self, tree, lengths):
        if not tree.children:
            node = self.word_lut(Variable(torch.LongTensor([self.dicts.lookup(tree.content, lib.Constants.UNK)])).cuda())

            output, state = self.leaf_module.forward(node) #  Variable(node, requires_grad=True)

        elif tree.children:
            # for idx in xrange(tree.num_children):
            lo, (lc, lh) = self.forward(tree.children[0], lengths)
            ro, (rc, rh) = self.forward(tree.children[1], lengths)
            # lc, lh, lo, rc, rh, ro = self.get_child_state(tree)
            state = self.composer.forward(lc, lh, rc, rh)

            output = torch.cat([lo, ro])
            # del lc, lh, lo, rc, rh, ro
            if not tree.parent:
                # max_length = int(torch.max(lengths.data))
                max_length = np.max(lengths)
                output.data.unsqueeze_(1)
                supl = max_length - output.size()[0]
                if supl > 0:
                    output.data = torch.cat([output.data, torch.zeros((supl, output.size()[1], output.size()[2])).cuda()], 0)

                state[0].data.unsqueeze_(1)
                state[1].data.unsqueeze_(1)
        return output, state

class TreeDecoder_W2V(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(TreeDecoder_W2V, self).__init__()
        # self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=lib.Constants.PAD)
        self.embeddings = gensim.models.Word2Vec.load(opt.embedding_w2v + 'processed_all.train_xe.comment.gz')

        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        if opt.has_attn:
            self.attn = lib.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.hidden_size =   opt.rnn_size
        self.opt = opt
        self.dicts = dicts

    def embedding(self, input):
        # print "emb-input: "
        # print input
        emb = []
        for i in range(input.shape[0]):
            emb_row = []
            for w in self.dicts.convertToLabels(input[i].tolist(), lib.Constants.UNK_WORD):
                try:
                    emb_row.append(self.embeddings.wv[w].astype(float))
                except:
                    emb_row.append(np.zeros((self.opt.word_vec_size), dtype=float))
            emb.append(emb_row)
        emb = Variable(torch.Tensor(emb))
        if self.opt.gpus:
            emb = emb.cuda()
        # print "decoder-emb: "
        # print emb
        return emb

    def step(self, emb, output, hidden, context):
        if self.input_feed:
            emb = torch.cat([emb, output], 1)
        output, hidden = self.rnn(emb, hidden)
        # print "decoder-output: "
        # print output
        # print "decoder-context: "
        # print context
        if self.opt.has_attn:
            output, attn = self.attn(output, context)
        output = self.dropout(output)
        return output, hidden

    def forward(self, inputs, init_states):
        emb, output, hidden, context = init_states
        # print "decoder-inputs: "
        # print inputs
        # embs = self.word_lut(inputs)
        # print "decoder-embs: "
        # print embs
        input = inputs.data.cpu().numpy()
        embs = self.embedding(input)

        outputs = []
        for i in range(inputs.size(0)):
            output, hidden = self.step(emb, output, hidden, context)
            outputs.append(output)
            emb = embs[i]

        outputs = torch.stack(outputs)
        return outputs

class TreeDecoder(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(TreeDecoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=lib.Constants.PAD)
        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        if opt.has_attn:
            self.attn = lib.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.hidden_size =   opt.rnn_size
        self.opt = opt

    def step(self, emb, output, hidden, context):
        if self.input_feed:
            emb = torch.cat([emb, output], 1)
        output, hidden = self.rnn(emb, hidden)

        if self.opt.has_attn:
            output, attn = self.attn(output, context)
        output = self.dropout(output)
        return output, hidden

    def forward(self, inputs, init_states):
        emb, output, hidden, context = init_states
        embs = self.word_lut(inputs)

        outputs = []
        for i in range(inputs.size(0)):
            output, hidden = self.step(emb, output, hidden, context)
            outputs.append(output)
            emb = embs[i]

        outputs = torch.stack(outputs)
        return outputs

class HybridDecoder(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(HybridDecoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=lib.Constants.PAD)
        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        if opt.has_attn:
            # self.text_attn = lib.GlobalAttention(opt.rnn_size)
            self.attn = lib.HybridAttention(opt.rnn_size)
        else:
            self.linear_out = nn.Linear(opt.rnn_size * 2, opt.rnn_size, bias=False)

        self.dropout = nn.Dropout(opt.dropout)
        self.hidden_size = opt.rnn_size
        self.opt = opt

    def step(self, emb, output, hidden_tree, context_tree, hidden_txt, context_txt):
        if self.input_feed:
            emb = torch.cat([emb, output], 1)
        output_tree, hidden_tree = self.rnn(emb, hidden_tree)

        output_txt, hidden_txt = self.rnn(emb, hidden_txt)
        if self.opt.has_attn:
            output, attn_tree, attn_txt = self.attn(output_tree, context_tree, output_txt, context_txt)
        else:
            output = self.linear_out(torch.cat((output_tree, output_txt), 1))

        output = self.dropout(output)
        return output, hidden_tree, hidden_txt

    def forward(self, inputs, init_states):
        emb, output, hidden_tree, context_tree, hidden_txt, context_txt = init_states
        embs = self.word_lut(inputs)
        outputs = []
        for i in range(inputs.size(0)):
            output, hidden_tree, hidden_txt = self.step(emb, output, hidden_tree, context_tree, hidden_txt, context_txt)
            outputs.append(output)
            emb = embs[i]

        outputs = torch.stack(outputs)
        return outputs

class Hybrid2SeqModel(nn.Module):
    def __init__(self, code_encoder, text_encoder, decoder, generator, opt):
        super(Hybrid2SeqModel, self).__init__()
        self.code_encoder = code_encoder
        self.text_encoder = text_encoder
        self.decoder = decoder
        self.generator = generator
        self.opt = opt

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def initialize(self, inputs, eval):
        tgt = inputs[2]
        trees = inputs[1][0]
        lengths = inputs[1][1]
        src_txt = inputs[0]
        enc_context_padded_tree, enc_hidden_tree0, enc_hidden_tree1 = [], [], []
        # code encoder
        for i, tree in enumerate(trees):
            enc_ctx_txt, enc_hidden_tree = self.code_encoder(tree, lengths)  # enc_contex <=> outputs
            enc_context_padded_tree.append(enc_ctx_txt)
            enc_hidden_tree0.append(enc_hidden_tree[0])
            enc_hidden_tree1.append(enc_hidden_tree[1])

        enc_context_padded_tree = torch.cat(enc_context_padded_tree, 1)
        enc_hidden_tree = (torch.cat(enc_hidden_tree0, 1), torch.cat(enc_hidden_tree1, 1))

        enc_hidden_txt, enc_context_txt = self.text_encoder(src_txt)
        init_output = self.make_init_decoder_output(enc_context_txt)

        init_token = Variable(torch.LongTensor([lib.Constants.BOS] * init_output.size(0)), volatile=eval)
        if self.opt.cuda:
            init_token = init_token.cuda()
        emb = self.decoder.word_lut(init_token)

        return tgt, (emb, init_output, enc_hidden_tree, enc_context_padded_tree.transpose(0, 1), enc_hidden_txt, enc_context_txt.transpose(0,1))

    def forward(self, inputs, eval, regression=False):
        targets, init_states = self.initialize(inputs, eval)
        outputs = self.decoder(targets, init_states)

        if regression:
            logits = self.generator(outputs)
            return logits.view_as(targets)
        return outputs

    def backward(self, outputs, targets, weights, normalizer, criterion, regression=False):
        grad_output, loss = self.generator.backward(outputs, targets, weights, normalizer, criterion, regression)
        outputs.backward(grad_output)
        return loss

    def predict(self, outputs, targets, weights, criterion):
        return self.generator.predict(outputs, targets, weights, criterion)

    def translate(self, inputs, max_length):
        targets, init_states = self.initialize(inputs, eval=True)
        # emb, output, hidden, context = init_states
        emb, output, hidden_tree, context_tree, hidden_txt, context_txt = init_states

        preds = []
        batch_size = targets.size(1)
        num_eos = targets[0].data.byte().new(batch_size).zero_()

        for i in range(max_length):
            # output, hidden = self.decoder.step(emb, output, hidden, context)
            output, hidden_tree, hidden_txt = self.decoder.step(emb, output, hidden_tree, context_tree, hidden_txt, context_txt)

            logit = self.generator(output)
            pred = logit.max(1)[1].view(-1).data
            preds.append(pred)

            # Stop if all sentences reach EOS.
            num_eos |= (pred == lib.Constants.EOS)
            if num_eos.sum() == batch_size: break

            emb = self.decoder.word_lut(Variable(pred))

        preds = torch.stack(preds)
        return preds

    def sample(self, inputs, max_length):
        targets, init_states = self.initialize(inputs, eval=False)
        emb, output, hidden_tree, context_tree, hidden_txt, context_txt = init_states

        outputs = []
        samples = []
        batch_size = targets.size(1)
        num_eos = targets[0].data.byte().new(batch_size).zero_()

        for i in range(max_length):
            # output, hidden = self.decoder.step(emb, output, hidden, context)
            output, hidden_tree, hidden_txt = self.decoder.step(emb, output, hidden_tree, context_tree, hidden_txt, context_txt)

            outputs.append(output)
            dist = F.softmax(self.generator(output))
            sample = dist.multinomial(1, replacement=False).view(-1).data
            samples.append(sample)

            # Stop if all sentences reach EOS.
            num_eos |= (sample == lib.Constants.EOS)
            if num_eos.sum() == batch_size: break

            emb = self.decoder.word_lut(Variable(sample))

        outputs = torch.stack(outputs)
        samples = torch.stack(samples)
        return samples, outputs

class Tree2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, generator, opt):
        super(Tree2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.opt = opt

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def initialize(self, inputs, eval):
        # src = inputs[2]
        tgt = inputs[2]
        # trees = inputs[4][0]
        # lengths = inputs[4][1]
        trees = inputs[1][0]
        lengths = inputs[1][1]
        # lengths = [tree.leaf_count() for tree in trees]
        enc_context_padded, enc_hidden0, enc_hidden1 = [], [], []
        # print "tree_lengths: ", lengths

        for i, tree in enumerate(trees):
            enc_ctx, enc_hidden = self.encoder(tree, lengths)  # enc_contex <=> outputs
            enc_context_padded.append(enc_ctx)
            enc_hidden0.append(enc_hidden[0])
            enc_hidden1.append(enc_hidden[1])
        # print "enc_context_padded: "
        # print enc_context_padded
        enc_context_padded = torch.cat(enc_context_padded, 1)
        enc_hidden = (torch.cat(enc_hidden0, 1), torch.cat(enc_hidden1, 1))

        init_output = self.make_init_decoder_output(enc_context_padded)
        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]), self._fix_enc_hidden(enc_hidden[1]))
        init_token = Variable(torch.LongTensor([lib.Constants.BOS] * init_output.size(0)), volatile=eval)
        if self.opt.cuda:
            init_token = init_token.cuda()
        emb = self.decoder.word_lut(init_token)
        return tgt, (emb, init_output, enc_hidden, enc_context_padded.transpose(0, 1))

    def forward(self, inputs, eval, regression=False):
        targets, init_states = self.initialize(inputs, eval)
        outputs = self.decoder(targets, init_states)

        if regression:
            logits = self.generator(outputs)
            return logits.view_as(targets)
        return outputs

    def backward(self, outputs, targets, weights, normalizer, criterion, regression=False):
        grad_output, loss = self.generator.backward(outputs, targets, weights, normalizer, criterion, regression)
        outputs.backward(grad_output)
        return loss

    def predict(self, outputs, targets, weights, criterion):
        return self.generator.predict(outputs, targets, weights, criterion)

    def translate(self, inputs, max_length):
        targets, init_states = self.initialize(inputs, eval=True)
        emb, output, hidden, context = init_states

        preds = []
        batch_size = targets.size(1)
        num_eos = targets[0].data.byte().new(batch_size).zero_()

        for i in range(max_length):
            output, hidden = self.decoder.step(emb, output, hidden, context)
            logit = self.generator(output)
            pred = logit.max(1)[1].view(-1).data
            preds.append(pred)

            # Stop if all sentences reach EOS.
            num_eos |= (pred == lib.Constants.EOS)
            if num_eos.sum() == batch_size: break

            emb = self.decoder.word_lut(Variable(pred))

        preds = torch.stack(preds)
        return preds

    def sample(self, inputs, max_length):
        targets, init_states = self.initialize(inputs, eval=False)
        emb, output, hidden, context = init_states

        outputs = []
        samples = []
        batch_size = targets.size(1)
        num_eos = targets[0].data.byte().new(batch_size).zero_()

        for i in range(max_length):
            output, hidden = self.decoder.step(emb, output, hidden, context)
            outputs.append(output)
            dist = F.softmax(self.generator(output))
            sample = dist.multinomial(1, replacement=False).view(-1).data
            samples.append(sample)

            # Stop if all sentences reach EOS.
            num_eos |= (sample == lib.Constants.EOS)
            if num_eos.sum() == batch_size: break

            emb = self.decoder.word_lut(Variable(sample))

        outputs = torch.stack(outputs)
        samples = torch.stack(samples)
        return samples, outputs

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, generator, opt):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.opt = opt

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def initialize(self, inputs, eval):
        src = inputs[0]
        tgt = inputs[2]
        enc_hidden, context = self.encoder(src)
        init_output = self.make_init_decoder_output(context)
        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]), self._fix_enc_hidden(enc_hidden[1]))
        init_token = Variable(torch.LongTensor([lib.Constants.BOS] * init_output.size(0)), volatile=eval)

        if self.opt.cuda:
            init_token = init_token.cuda()
        emb = self.decoder.word_lut(init_token)

        return tgt, (emb, init_output, enc_hidden, context.transpose(0, 1))

    def forward(self, inputs, eval, regression=False):
        targets, init_states = self.initialize(inputs, eval)
        outputs = self.decoder(targets, init_states)

        if regression:
            logits = self.generator(outputs)
            return logits.view_as(targets)
        return outputs

    def backward(self, outputs, targets, weights, normalizer, criterion, regression=False):
        grad_output, loss = self.generator.backward(outputs, targets, weights, normalizer, criterion, regression)
        outputs.backward(grad_output)
        return loss

    def predict(self, outputs, targets, weights, criterion):
        return self.generator.predict(outputs, targets, weights, criterion)

    def translate(self, inputs, max_length):
        targets, init_states = self.initialize(inputs, eval=True)
        emb, output, hidden, context = init_states
        
        preds = [] 
        batch_size = targets.size(1)
        num_eos = targets[0].data.byte().new(batch_size).zero_()

        for i in range(max_length):
            output, hidden = self.decoder.step(emb, output, hidden, context)
            logit = self.generator(output)
            pred = logit.max(1)[1].view(-1).data
            preds.append(pred)

            num_eos |= (pred == lib.Constants.EOS)
            if num_eos.sum() == batch_size: break

            emb = self.decoder.word_lut(Variable(pred))

        preds = torch.stack(preds)
        return preds

    def sample(self, inputs, max_length):
        targets, init_states = self.initialize(inputs, eval=False)
        emb, output, hidden, context = init_states

        outputs = []
        samples = []
        batch_size = targets.size(1)
        num_eos = targets[0].data.byte().new(batch_size).zero_()

        for i in range(max_length):
            output, hidden = self.decoder.step(emb, output, hidden, context)
            outputs.append(output)
            dist = F.softmax(self.generator(output))
            sample = dist.multinomial(1, replacement=False).view(-1).data
            samples.append(sample)

            # Stop if all sentences reach EOS.
            num_eos |= (sample == lib.Constants.EOS)
            if num_eos.sum() == batch_size: break

            emb = self.decoder.word_lut(Variable(sample))
            # emb = self.decoder.embedding(sample.unsqueeze(1).cpu().numpy()).squeeze(1)

        outputs = torch.stack(outputs)
        samples = torch.stack(samples)
        return samples, outputs