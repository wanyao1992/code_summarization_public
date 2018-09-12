# -*- coding: utf-8 -*-
# import sys
# sys.path.append('/usr/local/lib/python2.7/dist-packages/')

import lib
import codecs
import Constants
from Dict import Dict
import ast, asttokens
# from ..parser.JavaLexer import JavaLexer
# from ..parser.JavaParser import JavaParser
# from ..parser.JavaListener import JavaListener
# from antlr4 import *
import re
import copy

class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.leaf_states = list()
        self._leaf_count = 0
        self._leaf_contents = []
        self.content = None

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        # if getattr(self,'_size'):
        #     return self._size
        count = 1
        for i in xrange(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        # if getattr(self,'_depth'):
        #     return self._depth
        count = 0
        if self.num_children>0:
            for i in xrange(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def leaf_count_(self):
        # count = 0
        if not self.children:
            self._leaf_count = 1
        else:
            for c in self.children:
                leaf_count = c.leaf_count()
                self._leaf_count += leaf_count
                # print "self._leaf_count: ", self._leaf_count
        # self._leaf_count = count
        return self._leaf_count

    def leaf_count(self):
        count = 0
        # if not self.children:
        #     self._leaf_count = 1
        # else:
        # if not self.children:
        #     count += 1
        if self.children:
            for c in self.children:
                leaf_count = c.leaf_count()
                count += leaf_count
                # if leaf_count>count:
                #     count = leaf_count

        else:
            count += 1
        # self._leaf_count = count
        return count
                # self._leaf_count += leaf_count
                # print "self._leaf_count: ", self._leaf_count
        # self._leaf_count = count
        # return self._leaf_count

    def leaf_contents(self):
        if not self.children:
            # print "self.content: "
            # print self.content
            self._leaf_contents.append(self.content)
        else:
            for c in self.children:
                leaf_content = c.leaf_contents()
                self._leaf_contents.extend(leaf_content)

        return self._leaf_contents

def python_tokenize(line):
    tokens = re.split('\.|\(|\)|\:| |;|,|!|=|[|]', line)
    return [t for t in tokens if t.strip()]

def java_tokenize(line):
    stream = InputStream(line.decode('utf-8', 'ignore'))
    lexer = JavaLexer(stream)
    toks = CommonTokenStream(lexer)
    toks.fetch(500)
    return toks.tokens

def makeVocabulary(opt, name, filename, size):
    "Construct the word and feature vocabs."
    print("opt.lower: ", opt.lower)
    print(Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD)

    vocab = Dict([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD], lower=opt.lower)
    with codecs.open(filename, "r", "utf-8", errors='ignore') as f:
        for sent in f.readlines():
            if name == 'code':
                if opt.data_name.split('-')[1] == 'python':
                    words = python_tokenize(sent)
                elif opt.data_name.split('-')[1] == 'java':
                    words = java_tokenize(sent)
            elif name == 'comment':
                words = sent.split()
            for i in range(len(words)):
                vocab.add(words[i])

    originalSize = vocab.size()

    if size != 0:
        vocab = vocab.prune(size)
        print('Created dictionary of size %d (pruned from %d)' % (vocab.size(), originalSize))
    else:
        print('Created dictionary of size %d' % (vocab.size()))

    return vocab

def initVocabulary(opt, name, dataFile, vocabSize):
    print('Building ' + name + ' vocabulary...')
    genWordVocab = makeVocabulary(opt, name, dataFile, vocabSize)
    vocab = genWordVocab
    return vocab

def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)

def java2tree(line):
    stream = InputStream(line.decode('utf-8', 'ignore'))
    lexer = JavaLexer(stream)
    toks = CommonTokenStream(lexer)
    parser = JavaParser(toks)

    tree = parser.compilationUnit()

    return tree

def python2tree(line):
    atok = asttokens.ASTTokens(line, parse=True)
    return atok, atok.tree

def traverse_python_tree(atok, root):
    iter_children = asttokens.util.iter_children_func(root)
    node_json = {}
    current_global = {}
    current_idx, global_idx = 1, 1
    for node in asttokens.util.walk(root):
        if not next(iter_children(node), None) is None:
            child_num = 0
            for child in iter_children(node):
                child_num += 1
            global_idx = global_idx + child_num
            current_global[current_idx] = global_idx
        current_idx += 1
    # print current_global
    current_idx = 1
    for node in asttokens.util.walk(root):
        # print current_idx
        # idx_upper = current_idx
        node_json["%s%s"%(Constants.NODE_FIX, current_idx)] = {"node": type(node).__name__, "children": [], "parent": None}
        # idx_upper = len(node_json)
        if not next(iter_children(node), None) is None:
            child_idx = 0
            for child in iter_children(node):
                child_idx += 1
                node_json["%s%s"%(Constants.NODE_FIX, current_idx)]['children'].insert(0, "%s%s"%(Constants.NODE_FIX, current_global[current_idx]-child_idx+1))
        else: # leaf node
            node_json["%s%s"%(Constants.NODE_FIX, current_idx)]['children'].append(atok.get_text(node))

        current_idx += 1

    # update_parent
    for k, node in node_json.iteritems():
        children = [c for c in node['children'] if c.startswith(Constants.NODE_FIX)]
        if len(children):
            for c in children:
                node_json[c]['parent'] = k

    return node_json

def traverse_java_tree(tree, node_json, idx=1, global_idx=1):
    current_idx = idx
    if tree.getChildCount() == 0:
        node_json['%s%s'%(Constants.NODE_FIX, current_idx)] = {
        'node': '%s%s'%(Constants.NODE_FIX, current_idx),
        'children': [tree.getText()],
        'parent': None}

    elif tree.getChildCount():
        print('num. children: ', tree.getChildCount())
        node_json['%s%s'%(Constants.NODE_FIX, current_idx)] = {
        'node': '%s%s'%(Constants.NODE_FIX, current_idx),
        'children': [], 'parent': None}
        cx = global_idx
        for c in tree.getChildren():
            idx += 1
            global_idx += 1
            node_json['%s%s'%(Constants.NODE_FIX, current_idx)]['children'].append('%s%s'
            %(Constants.NODE_FIX, global_idx))
        for c in tree.getChildren():
            cx += 1
            node_json, global_idx = traverse_java_tree(c, node_json, cx, global_idx)
    return node_json, global_idx

def split_tree(tree_json, idx_upper):
    tree_json_splited = copy.deepcopy(tree_json)
    for k, node in tree_json.iteritems():
        if len(node['children']) > 2:
            tree_json_splited['%s%s'%(Constants.NODE_FIX, idx_upper + 1)] = {'node': 'Tmp', 'children': node['children'][1:], 'parent': k} # idx_upper + 2
            for ch in node['children'][1:]:
                tree_json_splited[ch]['parent'] = '%s%s'%(Constants.NODE_FIX, idx_upper + 1)
            tree_json_splited[k]['children'] =  [tree_json_splited[k]['children'][0],'%s%s'%(Constants.NODE_FIX, idx_upper + 1)]# idx_upper + 2

            idx_upper += 1
    for k, node in tree_json_splited.iteritems():
        children_length = len([c for c in node['children'] if c.startswith(Constants.NODE_FIX)])
        if children_length > 2:
            tree_json_splited = split_tree(tree_json_splited, idx_upper)
    return tree_json_splited

def _removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def merge_tree(tree_json):
    for k, node in tree_json.iteritems():
        children_length = len([c for c in node['children'] if c.startswith(Constants.NODE_FIX)])
        if children_length == 1:
            del_key = node['children'][0]
            node['children'] = tree_json[node['children'][0]]['children']

            for ch in tree_json[del_key]['children']:
                if ch.startswith(Constants.NODE_FIX) and ch in tree_json:
                    tree_json[ch]['parent'] = k

            tree_json = _removekey(tree_json, del_key)
            break

    for k, node in tree_json.iteritems():
        children_length = len([c for c in node['children'] if c.startswith(Constants.NODE_FIX)])
        if children_length == 1:
            tree_json = merge_tree(tree_json)
    return tree_json

def json2tree_binary(tree_json, tree, idx, prev=None):
    if prev == None:
        tree.parent = prev
        tree.idx = idx
    children = [c for c in tree_json[idx]['children'] if c.startswith(Constants.NODE_FIX)]
    if len(children) == 2:
        for c in children:
            t = Tree()
            t.parent = prev
            t.idx = c
            t.num_children = 0
            k = json2tree_binary(tree_json, t, c, tree)
            tree.add_child(k)
    else:
        tree.parent = prev
        tree.num_children = 0
        # print "tree_json[idx]['children']: "
        # print tree_json[idx]['children']
        tree.content = tree_json[idx]['children'][0]
    return tree

def json2tree_single(tree):
    leafs = []
    def dfs(tree):
        if not tree.children:
            leafs.append(tree.content)
        for c in tree.children:
            dfs(c)
    dfs(tree)

    def build_tree(leafs):
        tree = Tree()
        if len(leafs) == 2:
            ln = Tree()
            ln.content = leafs[0]
            rn = Tree()
            rn.content = leafs[1]
            tree.add_child(ln)
            tree.add_child(rn)
        else:
            rn = Tree()
            rn.content = leafs.pop()
            ln = build_tree(leafs)
            tree.add_child(ln)
            tree.add_child(rn)
        return tree

    tee = build_tree(leafs)
    return tee
