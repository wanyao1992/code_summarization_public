import torch
import torch.nn as nn
import math

_INF = float('inf')

class HybridAttention(nn.Module):
    def __init__(self, dim):
        super(HybridAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*4, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask_tree = None
        self.mask_txt = None

    def applyMask(self, mask_tree, mask_txt):
        self.mask_tree = mask_tree
        self.mask_txt = mask_txt

    def forward(self, inputs_tree, context_tree, inputs_txt, context_txt):
        """
        inputs: batch x dim
        context: batch x sourceL x dim
        """
        targetT_tree = self.linear_in(inputs_tree).unsqueeze(2)  # batch x dim x 1
        targetT_txt = self.linear_in(inputs_txt).unsqueeze(2)  # batch x dim x 1

        attn_tree = torch.bmm(context_tree, targetT_tree).squeeze(2)  # batch x sourceL
        attn_txt = torch.bmm(context_txt, targetT_txt).squeeze(2)  # batch x sourceL

        if self.mask_tree is not None and self.mask_txt is not None:
            attn_tree.data.masked_fill_(self.mask_tree, -_INF)
            attn_tree = self.sm(attn_tree)
            attn_txt.data.masked_fill_(self.mask_txt, -_INF)
            attn_txt = self.sm(attn_txt)

        attn3_tree = attn_tree.view(attn_tree.size(0), 1, attn_tree.size(1))  # batch x 1 x sourceL
        attn3_txt = attn_txt.view(attn_txt.size(0), 1, attn_txt.size(1))  # batch x 1 x sourceL

        weightedContext_tree = torch.bmm(attn3_tree, context_tree).squeeze(1)  # batch x dim
        contextCombined_tree = torch.cat((weightedContext_tree, inputs_tree), 1)

        weightedContext_txt = torch.bmm(attn3_txt, context_txt).squeeze(1)  # batch x dim
        contextCombined_txt = torch.cat((weightedContext_txt, inputs_txt), 1)

        contextCombined = torch.cat((contextCombined_tree, contextCombined_txt), 1)

        contextOutput = self.tanh(self.linear_out(contextCombined))

        return contextOutput, attn_tree, attn_txt
