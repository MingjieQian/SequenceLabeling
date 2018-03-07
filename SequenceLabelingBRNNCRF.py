import os
import re
import sys
import math
import nltk
import time
import pickle
import random
import shutil
import argparse
import itertools
import numpy as np
from enum import Enum
from gensim.models import word2vec
from collections import OrderedDict
from nltk.tag.senna import SennaTagger
from nltk.tag.perceptron import PerceptronTagger

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.platform import tf_logging as logging

ver = tf.__version__.split('.')
if ver[0] == '0':
    from tensorflow.python.ops.rnn_cell import GRUCell
    from tensorflow.python.ops.rnn_cell import LSTMCell
else:
    from tensorflow.contrib.rnn import GRUCell
    from tensorflow.contrib.rnn import LSTMCell

from utilities import TextDataset, fetch_minibatches, get_data_input, get_glove_vocab, get_logger

__author__ = "Mingjie Qian"
__date__ = "January 15th, 2018"

'''
1. Using TreebankWordTokenizer - July 17th, 2017.
2. Numbers are collapsed into a single NUM token - June 11th, 2017.
3. Multiple tag types are learned in a single model.
4. Annotation scheme choices include
        a. XML tags '<TYPE>'w and w</TYPE> - June 26th, 2017
        b. Bracket tags '[TYPE w' and 'w]' - July 5th, 2017
        c. CoNLL - January 18th, 2018
5. Class weights are automatically set up per class distribution.
6. Bidirectional RNN is based on LSTM or GRU - June 2nd, 2017.
7. Backward state sequence comprise
        a. zero initial state vector
        b. first T - 1 state vectors
8. Glove embeddings are supported - January 18th, 2018
9. Tensorflow >= 1.0 can be used as backend as well - January 23rd, 2018
10. Add an option to extend an input sequence by extra tokens, e.g., "^start" or "end$" - January 26th, 2018
11. Add an option to read a minibatch from a text dataset each time 
    to make training and validation memory efficient. - January 31st, 2018
12. Support multiple BRNN architectures - February 2nd, 2018:
        a. Vanilla
        b. Backward shift
        c. Residual
'''

NUM = "NUM"
UNK = "UNK"
PAD = "PAD"
STT = "^start"
END = "end$"

# tag_types = set()
B = {}
I = {}
tag_open_pattern = re.compile(r'<([A-Z0-9]+)\b[^>]*>(.+)')
O = "O"
N = "N"

# class_weights = {B_ans: 5, I_ans:4, B_ctx: 4, I_ctx:3, O:1}

NULL = "NULL"

# max_gradient_norm = 15.0
# max_gradient_norm = -1

# encoding = "latin-1"
encoding = "utf-8"
checkpoint_name = "concepts.ckpt"  # Checkpoint filename
max_word_length = 30


# pos_tagger = PerceptronTagger()


def build_char_dictionary():
    char_dict = {PAD: 0}
    # chars = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\|_@#$%ˆ&*~‘+-=<>()[]{}'
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'
    for char in chars:
        char_dict[char] = len(char_dict)
    char_dict[UNK] = len(char_dict)
    return char_dict


# def build_pos_dictionary():
#     pos_dict = {}
#     pos_tags = [NULL, '$', "''", '(', ')', ',', '--', '.', ':', 'CC', 'CD',
#                 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
#                 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
#                 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
#                 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']
#     for pos_tag in pos_tags:
#         pos_dict[pos_tag] = len(pos_dict)
#     return pos_dict


def build_tag_dictionary(tag_type_list):
    # tag_dict = {B_ans: 0, I_ans: 1, B_ctx: 2, I_ctx: 3, O: 4, N: 5}
    tag_dict = {}
    for tag_type in tag_type_list:
        B[tag_type] = 'B-' + tag_type
        I[tag_type] = 'I-' + tag_type
        tag_dict[B[tag_type]] = len(tag_dict)
        tag_dict[I[tag_type]] = len(tag_dict)
    tag_dict[O] = len(tag_dict)
    tag_dict[N] = len(tag_dict)
    return tag_dict


def save_tag_type_list(model_dir, tag_type_list):
    filepath = os.path.join(model_dir, 'tag_type_list.txt')
    with open(filepath, 'w', encoding=encoding) as f:
        for tag_type in tag_type_list:
            f.write("%s\n" % tag_type)
    print("Tag types were saved in %s" % filepath)


def load_tag_type_list(model_dir):
    filepath = os.path.join(model_dir, 'tag_type_list.txt')
    tag_type_list = []
    with open(filepath, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tag_type_list.append(line)
    return tag_type_list


def build_class_id_weight_dict(tag_dict, class_weights):
    class_id_weight_dict = [1.0] * len(class_weights)
    for (tag, id) in tag_dict.items():
        if tag in class_weights:
            class_id_weight_dict[id] = class_weights[tag]
    return class_id_weight_dict


def save_dictionary(model_dir, vocab):
    dict_path = os.path.join(model_dir, "vocab.txt")
    with open(dict_path, 'w', encoding=encoding) as f:
        for (word, index) in vocab.items():
            f.write("%s\t%d\n" % (word, index))
    print("Dictionary size: %s" % len(vocab))
    print("Dictionary file was saved in %s" % dict_path)


def load_dictionary(model_dir):
    dict_path = os.path.join(model_dir, "vocab.txt")
    vocab = {}
    with open(dict_path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            # print line
            if not line:
                continue
            entry = line.split("\t")
            vocab[entry[0]] = int(entry[1])
    return vocab


def save_class_weights(model_dir, class_weights):
    filepath = os.path.join(model_dir, 'class_weights.txt')
    with open(filepath, 'w', encoding=encoding) as f:
        for (tag, weight) in class_weights.items():
            f.write("%s\t%f\n" % (tag, weight))
    print("Class weights file was saved in %s" % filepath)


def save_TF_version(model_dir):
    filepath = os.path.join(model_dir, 'tf_version.txt')
    with open(filepath, 'w', encoding=encoding) as f:
        f.write(tf.__version__ + '\n')


def save_max_word_length(max_word_length):
    filepath = os.path.join(model_dir, 'max_word_length.txt')
    with open(filepath, 'w', encoding=encoding) as f:
        # f.write(max_word_length + '\n')
        # TypeError: unsupported operand type(s) for +: 'int' and 'str'
        f.write(str(max_word_length) + '\n')


def load_max_word_length():
    global max_word_length
    filepath = os.path.join(model_dir, 'max_word_length.txt')
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding=encoding) as f:
        max_word_length = int(f.readline().strip())


def extract_word_xml(token, update_tags=False, tag_types=None):
    # if token.startswith('<'):
    match_o = tag_open_pattern.match(token)
    if match_o:
        tag_type = match_o.group(1)
        token = str(match_o.group(2))
        if update_tags:
            if tag_type not in tag_types:
                tag_types.add(tag_type)
    if token.endswith('>'):
        # match_c = tag_end_pattern.match(token)
        # if match_c:
        #     token = str(match_c.group(1))
        idx = token.rfind("</")
        if idx != -1:
            token = token[:idx]
    return token


def extract_word_bracket(token, update_tags=False, tag_types=None):
    if token.startswith('[') and len(token) > 1:
        tag_type = token[1:]
        token = ''
        if update_tags:
            if tag_type not in tag_types:
                tag_types.add(tag_type)
    elif token.endswith(']') and len(token) > 1:
        token = token[:-1]
    return token


def is_number(s):
    return s.isdigit()


def is_number_(s):
    # i = 0
    # while i < len(s) and (s[i] == '-' or s[i] == '+'):
    #     i += 1
    # s = s[i:]
    s = s.strip().lstrip('+-')
    if not s:
        return False
    if s == '.':
        return False
    if s[0] == '.':
        s = s[1:]
    if s[0] == 'e':
        return False
    # Finite state machine
    # 0: before 'e'
    # 1: 1st position after '.'
    # 2: 2nd or later position after '.'
    # 3: 1st position after 'e'
    # 4: 2nd or later position after 'e'
    state = 0
    for c in s:
        if state == 0:
            if '0' <= c <= '9':
                pass
            elif c == '.':
                state = 1
            elif c == 'e':
                state = 3
            else:
                return False
        elif state == 1:
            if '0' <= c <= '9':
                state = 2
            elif c == 'e':
                state = 3
            else:
                return False
        elif state == 2:
            if '0' <= c <= '9':
                pass
            elif c == 'e':
                state = 3
            else:
                return False
        elif state == 3:
            if '0' <= c <= '9':
                pass
            elif c == '+' or c == '-':
                pass
            else:
                return False
            state = 4
        elif state == 4:
            if '0' <= c <= '9':
                pass
            else:
                return False
    return state != 3


def compute_counts(data_dir, training_filename, tag_types, counts=None, last_dataset=True):
    global max_word_length
    print('Counting frequency for each term...')
    train_path = os.path.join(data_dir, training_filename)
    if counts is None:
        counts = {NUM: 0}
    if annotation_scheme == 'CoNLL':
        cnt = 0
        with open(train_path, 'r', encoding=encoding) as f:
            split_pattern = re.compile(r'[ \t]')
            wrd_seq = []
            for line in f:
                line = line.strip()
                if not line or line.startswith('-DOCSTART-'):
                    if wrd_seq:
                        cnt += 1
                        if cnt % 100 == 0:
                            print("  read %d tagged examples" % cnt, end="\r")
                        wrd_seq = []
                    continue
                # container = line.split(' \t')
                container = split_pattern.split(line)
                token, tag = container[0], container[-1]
                token = token.lower()
                if not token:
                    continue
                # if use_all_chars:
                max_word_length = max(max_word_length, len(token))
                if token in counts:
                    counts[token] += 1
                else:
                    if is_number(token):
                        counts[NUM] += 1
                    else:
                        counts[token] = 1
                tag_type = tag[2:].upper()
                if tag_type and tag_type not in tag_types:
                    tag_types.add(tag_type)
                wrd_seq.append(token)
            print("  read %d tagged examples" % cnt)
    else:
        with open(train_path, "r", encoding=encoding) as f:
            cnt = 0
            for tagged_query in f:
                tagged_query = tagged_query.strip()
                if not tagged_query:
                    continue
                for token in tagged_query.split():
                    token = extract_word(token, True, tag_types)
                    if not token:
                        continue
                    max_word_length = max(max_word_length, len(token))
                    if token in counts:
                        counts[token] += 1
                    else:
                        if is_number(token):
                            counts[NUM] += 1
                        else:
                            counts[token] = 1
                cnt += 1
                if cnt % 100 == 0:
                    print("  read %d tagged examples" % cnt, end="\r")
            print("  read %d tagged examples" % cnt)
    print("counts['NUM']: %d" % counts[NUM])

    if last_dataset:
        del counts[NUM]
        counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return counts


def save_count_file(model_dir, counts):
    counts_path = os.path.join(model_dir, "counts.txt")
    with open(counts_path, 'w', encoding=encoding) as f:
        for (word, count) in counts:
            f.write("%s\t%d\n" % (word, count))
    print("Term frequency file was saved in %s" % counts_path)


def parse_a_tagged_query_xml(tagged_query, vocab, update_vocab, extend_sequence=False):
    """
    Parse a tagged query.

    :param tagged_query:
    :param vocab:
    :param update_vocab: vocab would be updated if a new word is discovered.
    :return: tag_seq, wrd_seq
    """
    tagged_query = tagged_query.strip()

    tag_seq = []
    wrd_seq = []
    if extend_sequence:
        wrd_seq += [STT]
        tag_seq += [O]
    # Finite state machine
    # 0: outside a concept
    # 1: within a concept, that said, the current token is inside a concept but is not the beginning token
    state = 0
    tag_type = ''
    for token in tagged_query.split():
        # match_o = None
        # if token.startswith('<'):
        match_o = tag_open_pattern.match(token)
        if match_o:
            tag_type = match_o.group(1)
            token = str(match_o.group(2))
        # match_c = None
        idx = -1
        if token.endswith('>'):
            idx = token.rfind("</")
            if idx != -1:
                token = token[:idx]
                # match_c = tag_end_pattern.match(token)
                # if match_c:
                #     token = str(match_c.group(1))

        if state == 1:
            tag_seq.append(I[tag_type])
        elif match_o:  # beginning = match_o is not None
            tag_seq.append(B[tag_type])
            state = 1
        else:
            tag_seq.append(O)

        # if match_c:
        if idx != -1:  # ending = idx != -1
            state = 0

        word = token.lower()
        wrd_seq.append(word)
        if update_vocab and word not in vocab and not is_number(word):
            vocab[word] = len(vocab)

    # pos_seq = [pos_pair[1] for pos_pair in pos_tagger.tag(wrd_seq)]
    if extend_sequence:
        wrd_seq += [END]
        tag_seq += [O]
    return tag_seq, wrd_seq


def parse_a_tagged_query_bracket(tagged_query, vocab, update_vocab, extend_sequence=False):
    """
    Parse a tagged query.

    :param tagged_query:
    :param vocab:
    :param update_vocab: vocab would be updated if a new word is discovered.
    :return: tag_seq, wrd_seq
    """
    tagged_query = tagged_query.strip()

    tag_seq = []
    wrd_seq = []
    if extend_sequence:
        wrd_seq += [STT]
        tag_seq += [O]
    # Finite state machine
    # 0: outside a concept
    # 1: within a concept, that said, the current token is inside a concept but is not the beginning token
    state = 0
    tag_type = ''
    inside = False  # inside a concept, either beginning or within a concept. So beginning = inside and state != 1
    ending = False
    for token in tagged_query.split():
        if token.startswith('[') and len(token) > 1:
            tag_type = token[1:]
            inside = True  # inside is only used for a beginning token.
            continue
        elif token.endswith(']') and len(token) > 1:
            token = token[:-1]
            ending = True

        if state == 1:
            tag_seq.append(I[tag_type])
        elif inside:
            tag_seq.append(B[tag_type])
            state = 1
        else:
            tag_seq.append(O)

        if ending:
            state = 0
            inside = False
            ending = False

        word = token.lower()
        wrd_seq.append(word)
        if update_vocab and word not in vocab and not is_number(word):
            vocab[word] = len(vocab)

    # pos_seq = [pos_pair[1] for pos_pair in pos_tagger.tag(wrd_seq)]
    if extend_sequence:
        wrd_seq += [END]
        tag_seq += [O]
    return tag_seq, wrd_seq


def build_sequence_from_a_tagged_query(tagged_query, tag_dict, vocab, update_vocab=True):
    """
    Build data instances from a tagged query by parsing the tagged query.
    The vocabulary vocab must not be None. Tagged queries are required For training, cross validation, and
    evaluation. In general, vocab would be updated if a new word is discovered during training and cross
    validation, while it would not be updated for evaluation since vocab must be loaded from a vocabulary file.

    :param tagged_query:
    :param tag_dict:
    :param vocab:
    :param update_vocab: vocab would be updated if a new word is discovered
    :return: prev_tag_id_seq, wrd_id_seq, label_seq, wrd_seq
    """
    global verbose
    global extend_sequence

    tag_seq, wrd_seq = parse_a_tagged_query(tagged_query, vocab, update_vocab, extend_sequence)

    if verbose:
        print("Tagged query: %s" % tagged_query)
        print(tag_seq)
        print(wrd_seq)
        # print(pos_seq)

    # Build training examples for this tagged query.
    # Format for a training example:
    # sequence = (prev_tag_id_seq, pos_id_seq, wrd_id_seq)
    # label = label_seq
    # prev_tag_id_seq = [tag_dict[tag_seq[t - 1] if t > 0 else _N] for t in range(len(tag_seq))]
    # pos_id_seq = [pos_dict[pos_tag if pos_tag in pos_dict else NULL] for pos_tag in pos_seq]
    wrd_id_seq = [vocab[w if w in vocab else NUM if is_number(w) else UNK] for w in wrd_seq]
    label_seq = [tag_dict[tag] for tag in tag_seq]
    prev_tag_id_seq = label_seq[:-1]
    prev_tag_id_seq.insert(0, tag_dict[N])

    return prev_tag_id_seq, wrd_id_seq, label_seq, wrd_seq


def build_sequence_from_a_query(query, vocab):
    # wrd_seq = query.strip().lower().split()
    # wrd_seq = basic_tokenizer(query)
    query = query.strip()
    tokens = nltk.word_tokenize(query)
    if extend_sequence:
        wrd_seq = [STT] + [token.lower() for token in tokens] + [END]
    else:
        wrd_seq = [token.lower() for token in tokens]
    # wrd_seq = [token.lower() for token in tokens]
    # pos_seq = [pos_pair[1] for pos_pair in pos_tagger.tag(wrd_seq)]
    # pos_id_seq = [pos_dict[pos_tag if pos_tag in pos_dict else NULL] for pos_tag in pos_seq]
    wrd_id_seq = [vocab[w if w in vocab else NUM if is_number(w) else UNK] for w in wrd_seq]
    return wrd_id_seq, tokens


def build_sequence_from_a_word_sequence_in_char_level(wrd_seq, char_dict, max_word_length):
    """
    wrd_seq must be generated from a lowercase well-formed clean query.

    :param wrd_seq:
    :param char_dict:
    :return: character id matrix with size [len(wrd_seq), max_word_length]
    """
    char_id_matrix = []
    for w in wrd_seq:
        char_id_seq = []
        i = 0
        while i < len(w) and i < max_word_length:
            char_id_seq.append(char_dict[w[i] if w[i] in char_dict else 'UNK'])
            i += 1
        while i < max_word_length:
            char_id_seq.append(char_dict[PAD])
            i += 1
        char_id_matrix.append(char_id_seq)
    return char_id_matrix


def build_training_data(data_dir, training_filename, tag_dict, vocab, update_vocab=True):
    train_path = os.path.join(data_dir, training_filename)
    tag_id_seq_map = {}
    # pos_id_seq_map = {}
    wrd_id_seq_map = {}
    label_seq_map = {}
    with open(train_path, "r", encoding=encoding) as f:
        cnt = 0
        for tagged_query in f:
            tagged_query = tagged_query.strip()
            if not tagged_query:
                continue
            prev_tag_id_seq, wrd_id_seq, label_seq, wrd_seq = build_sequence_from_a_tagged_query(
                tagged_query, tag_dict, vocab, update_vocab)
            T = len(label_seq)
            if T not in label_seq_map:
                label_seq_map[T] = []
                tag_id_seq_map[T] = []
                # pos_id_seq_map[T] = []
                wrd_id_seq_map[T] = []
            label_seq_map[T].append(label_seq)
            tag_id_seq_map[T].append(prev_tag_id_seq)
            # pos_id_seq_map[T].append(pos_id_seq)
            wrd_id_seq_map[T].append(wrd_id_seq)

            cnt += 1
            if cnt % 100 == 0:
                print("  read %d tagged queries" % cnt, end="\r")
        print("  read %d tagged queries" % cnt)
    return tag_id_seq_map, wrd_id_seq_map, label_seq_map, vocab


def export_glove_vectors(glove_filename, output_filename):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = {}
    # embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            embeddings[word] = np.asarray(embedding)
    # np.savez_compressed(output_filename, embeddings=embeddings)
    pickle_path = output_filename
    f = open(pickle_path, 'wb')
    # pickle.dump(pos_id_seq_map, f)
    pickle.dump(embeddings, f)
    f.close()


def get_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        embeddings dictionary

    """
    pickle_path = filename
    with open(pickle_path, 'rb') as handle:
        # pos_id_seq_map = pickle.load(handle)
        embeddings = pickle.load(handle)
    return embeddings


def length(sequence, zero_padded=False):
    used = tf.sign(sequence + (1 if not zero_padded else 0))
    res = tf.reduce_sum(used, axis=1)
    res = tf.cast(res, tf.int32)
    return res


def length_with_padding(sequence):
    """
    Padding index must be zero.

    :param sequence:
    :return:
    """
    used = tf.sign(sequence)
    res = tf.reduce_sum(used, axis=2)
    res = tf.cast(res, tf.int32)
    return res


def create_model(model_dir,
                 char_embedding_classes,
                 tag_embedding_classes,
                 # pos_embedding_classes,
                 wrd_embedding_classes,
                 embedding_size,
                 hidden_size,
                 vocab=None,
                 ):
    global initialized_by_pretrained_embedding
    global pretrained_embedding_path
    global train_word_embedding
    global zero_padded

    sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.

    # tag_initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3, seed=1)
    # tag_embedding = tf.get_variable("tag_embedding",
    #                                 [tag_embedding_classes, embedding_size],
    #                                 trainable=True,
    #                                 initializer=tag_initializer)

    # pos_initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3, seed=2)
    # pos_embedding = tf.get_variable("pos_embedding",
    #                                 [pos_embedding_classes, embedding_size],
    #                                 trainable=True,
    #                                 initializer=pos_initializer)

    if initialized_by_pretrained_embedding:
        if not os.path.exists(pretrained_embedding_path):
            print("Pre-trained embedding path %s doesn't exist." % pretrained_embedding_path, file=sys.stderr)
            shutil.rmtree(model_dir)
            exit()

        if pretrained_embedding_path.find('glove') == -1:
            pretrained_word_embedding = [[] for i in range(wrd_embedding_classes)]
            model = word2vec.KeyedVectors.load_word2vec_format(pretrained_embedding_path, binary=True)
            print("Pre-trained embedding size is %d." % model.syn0.shape[1])
            print("Specified embedding size is %d." % embedding_size)
            if embedding_size != model.syn0.shape[1]:
                print("Specified embedding size (%d) doesn't match pre-trained embedding size (%d)."
                      % (embedding_size, model.syn0.shape[1]), file=sys.stderr)
                print("Please make sure the pre-trained embedding dimensionality matches the specified embedding size.")
                shutil.rmtree(model_dir)
                exit()

            cnt = 0
            for (word, index) in vocab.items():
                if word in model.vocab:
                    pretrained_word_embedding[index] = model.word_vec(word)
                    cnt += 1
                else:
                    pretrained_word_embedding[index] = np.random.uniform(-sqrt3, sqrt3, (embedding_size,))
                    # pretrained_word_embedding[index] = np.zeros((embedding_size,), dtype=np.float32)
            pretrained_word_embedding = np.array(pretrained_word_embedding)
        else:
            pretrained_word_embedding = [[] for i in range(wrd_embedding_classes)]
            # trimmed_filepath = pretrained_embedding_path + '.trimmed.npz'
            glove_embedding_filepath = pretrained_embedding_path + '.npz'
            # embeddings = get_glove_vectors(glove_embedding_filepath)
            # embeddings = {}
            cnt = 0
            processed_words = set()
            with open(pretrained_embedding_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip().split(' ')
                    word = line[0]
                    embedding = [float(x) for x in line[1:]]
                    if word in vocab:
                        pretrained_word_embedding[vocab[word]] = np.asarray(embedding)
                        processed_words.add(word)
                        cnt += 1
            for (word, index) in vocab.items():
                if word not in processed_words:
                    pretrained_word_embedding[index] = np.zeros((embedding_size,), dtype=np.float32)
            # pretrained_word_embedding = get_trimmed_glove_vectors(trimmed_filepath)
            # cnt = 0
            # with open(pretrained_embedding_path, 'r', encoding='utf-8') as f:
            #     for line in f:
            #         line = line.strip().split(' ')
            #         word = line[0]
            #         # embedding = [float(x) for x in line[1:]]
            #         if word in vocab:
            #             cnt += 1
            #             # word_idx = vocab[word]
            #             # embeddings[word_idx] = np.asarray(embedding)
            # cnt = 0
            # for (word, embedding) in embeddings.items():
            #     print(word, embedding)
            #     cnt += 1
            #     if cnt == 5:
            #         break
            # cnt = 0
            # for (word, index) in vocab.items():
            #     if word in embeddings:
            #         pretrained_word_embedding[index] = embeddings[word]
            #         cnt += 1
            #     else:
            #         # pretrained_word_embedding[index] = np.random.uniform(-sqrt3, sqrt3, (embedding_size,))
            #         pretrained_word_embedding[index] = np.zeros((embedding_size,), dtype=np.float32)
            pretrained_word_embedding = np.array(pretrained_word_embedding)
        print('wrd_embedding_classes:', wrd_embedding_classes)
        print('len(vocab):', len(vocab))
        print('Number of word existing in pretrained embedding vocab:', cnt)
        word_initializer = init_ops.constant_initializer(pretrained_word_embedding)
    else:
        # Default initializer for embeddings should have variance=1.
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        word_initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3, seed=0)
    wrd_embedding = tf.get_variable("wrd_embedding",
                                    [wrd_embedding_classes, embedding_size],
                                    trainable=train_word_embedding,
                                    initializer=word_initializer)

    # input_tag_ids = tf.placeholder(tf.int32, shape=[None, None], name="input_tag_ids")
    # input_pos_ids = tf.placeholder(tf.int32, shape=[None, None], name="input_pos_ids")
    input_wrd_ids = tf.placeholder(tf.int32, shape=[None, None], name="input_wrd_ids")

    # tag_embedded = embedding_ops.embedding_lookup(tag_embedding, input_tag_ids)
    # pos_embedded = embedding_ops.embedding_lookup(pos_embedding, input_pos_ids)
    wrd_embedded = embedding_ops.embedding_lookup(wrd_embedding, input_wrd_ids)

    # shape = (batch size, length of sentence, max length of word)
    input_chr_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                   name="input_chr_ids")
    ver = tf.__version__.split('.')
    with tf.variable_scope("chars"):
        if use_chars:
            # get char embeddings matrix
            _char_embeddings = tf.get_variable(
                name="_char_embeddings",
                dtype=tf.float32,
                shape=[char_embedding_classes, embedding_size_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                     input_chr_ids, name="char_embeddings")

            # put the time dimension on axis=1
            s = tf.shape(char_embeddings)
            char_embeddings = tf.reshape(char_embeddings,
                                         shape=[s[0] * s[1], s[-2], embedding_size_char])
            word_lengths = length_with_padding(input_chr_ids)
            word_lengths = tf.reshape(word_lengths, shape=[s[0] * s[1]])

            # bi lstm on chars
            cell_fw = LSTMCell(hidden_size_char, state_is_tuple=True)
            cell_bw = LSTMCell(hidden_size_char, state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, char_embeddings,
                sequence_length=word_lengths, dtype=tf.float32)

            # read and concat output
            _, ((_, output_fw), (_, output_bw)) = _output
            print('output_fw:', output_fw)
            print('output_bw:', output_bw)
            # For TF version 0.12, concat_dim doesn't support -1. If -1 is used, output will have shape [?, 200, ?, 100]
            if ver[0] == '0':
                output = tf.concat(1, [output_fw, output_bw])
            else:
                output = tf.concat([output_fw, output_bw], 1)
            print('output:', output)

            # shape = (batch size, sentence length, 2*hidden_size_char)
            output = tf.reshape(output, shape=[s[0], s[1], 2 * hidden_size_char])
            print('output:', output)
            if ver[0] == '0':
                wrd_embedded = tf.concat(2, [wrd_embedded, output])
            else:
                wrd_embedded = tf.concat([wrd_embedded, output], 2)
            print('wrd_embedded:', wrd_embedded)

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    word_embedding = tf.nn.dropout(wrd_embedded, keep_prob, seed=1, name='word_embedding')

    # sequence_length_ori = length(input_wrd_ids)
    sequence_length = length(input_wrd_ids, zero_padded)

    use_brnn_old = False
    if use_brnn_old:
        # Forward
        # embedded_concat_fw = tf.concat(2, [tag_embedded, wrd_embedded], name='embedded_concat_fw')
        # initial_state_fw = tf.placeholder(tf.float32, [None, hidden_size], 'initial_state_fw')
        with tf.variable_scope('Forward') as scope:
            if cell_type == "GRU":
                cell_fw = GRUCell(hidden_size)
            elif cell_type == "LSTM":
                cell_fw = LSTMCell(hidden_size)
            else:
                cell_fw = GRUCell(hidden_size)
            outputs_fw, rnn_state_fw = tf.nn.dynamic_rnn(cell_fw, word_embedding, sequence_length, dtype=tf.float32,
                                                         scope=scope)
        print(rnn_state_fw)

        # Backward
        word_embedding_bw = tf.reverse_v2(word_embedding, [1], 'word_embedding_bw')
        initial_hidden_bw = tf.placeholder(tf.float32, [None, hidden_size], 'initial_hidden_bw')
        with tf.variable_scope('Backward') as scope:
            if cell_type == "GRU":
                cell_bw = GRUCell(hidden_size)
            elif cell_type == "LSTM":
                cell_bw = LSTMCell(hidden_size)
            else:
                cell_bw = GRUCell(hidden_size)
            if shift_backward:
                outputs_bw, rnn_state_bw = tf.nn.dynamic_rnn(cell_bw, word_embedding_bw,
                                                             sequence_length=sequence_length - 1, dtype=tf.float32,
                                                             scope=scope)
            else:
                outputs_bw, rnn_state_bw = tf.nn.dynamic_rnn(cell_bw, word_embedding_bw,
                                                             sequence_length=sequence_length, dtype=tf.float32,
                                                             scope=scope)
            print(rnn_state_bw)
        if shift_backward:
            outputs_bw_slice = tf.slice(outputs_bw, [0, 0, 0],
                                        [tf.shape(outputs_bw)[0], tf.shape(outputs_bw)[1] - 1, tf.shape(outputs_bw)[2]])
            if ver[0] == '0':
                outputs_concat_bw = tf.concat(1, [
                    tf.reshape(initial_hidden_bw, [tf.shape(initial_hidden_bw)[0], 1, hidden_size]), outputs_bw_slice])
            else:
                outputs_concat_bw = tf.concat(
                    [tf.reshape(initial_hidden_bw, [tf.shape(initial_hidden_bw)[0], 1, hidden_size]), outputs_bw_slice],
                    1)
            hidden_bw_reverse = tf.reverse_v2(outputs_concat_bw, [1], 'hidden_bw_reverse')
        else:
            hidden_bw_reverse = tf.reverse_v2(outputs_bw, [1], 'hidden_bw_reverse')
        if ver[0] == '0':
            hidden_concat = tf.concat(2, [outputs_fw, hidden_bw_reverse], 'hidden_concat')
        else:
            hidden_concat = tf.concat([outputs_fw, hidden_bw_reverse], 2, 'hidden_concat')

    w_fc_width = 2 * hidden_size
    with tf.variable_scope("brnn"):
        cell_fw = LSTMCell(hidden_size, state_is_tuple=True)
        cell_bw = LSTMCell(hidden_size, state_is_tuple=True)
        _output = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, word_embedding,
            sequence_length=sequence_length, dtype=tf.float32)
        (output_fw, output_bw), _ = _output
        if brnn_type == 'vanilla':
            if ver[0] == '0':
                hidden_concat = tf.concat(2, [output_fw, output_bw])
            else:
                hidden_concat = tf.concat([output_fw, output_bw], 2)
        else:
            s = tf.shape(input_wrd_ids)
            output_bw_slice = tf.slice(output_bw, [0, 1, 0], [s[0], s[1] - 1, hidden_size])
            # TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'
            # initial_hidden = tf.constant(0, dtype=tf.float32, shape=[None, hidden_size])
            initial_state_bw = cell_bw.zero_state(s[0], tf.float32)
            initial_cell_state_bw, initial_hidden_bw = initial_state_bw
            print(initial_hidden_bw)
            print(initial_state_bw[1])
            if ver[0] == '0':
                output_bw_shift = tf.concat(1, [output_bw_slice, tf.expand_dims(initial_hidden_bw, 1)])
            else:
                output_bw_shift = tf.concat([output_bw_slice, tf.expand_dims(initial_hidden_bw, 1)], 1)

            if brnn_type == 'backward_shift':
                if ver[0] == '0':
                    hidden_concat = tf.concat(2, [output_fw, output_bw_shift])
                else:
                    hidden_concat = tf.concat([output_fw, output_bw_shift], 2)
            elif brnn_type == 'residual':
                initial_state_fw = cell_fw.zero_state(s[0], tf.float32)
                initial_cell_state_fw, initial_hidden_fw = initial_state_fw
                output_fw_slice = tf.slice(output_fw, [0, 0, 0], [s[0], s[1] - 1, hidden_size])
                # output_fw_shift = tf.concat(1, [tf.expand_dims(initial_hidden_fw, 1), output_fw_slice])
                if ver[0] == '0':
                    hidden_concat = tf.concat(2, [output_fw, wrd_embedded, output_bw_shift])
                else:
                    hidden_concat = tf.concat([output_fw, wrd_embedded, output_bw_shift], 2)
                w_fc_width += embedding_size
                if use_chars:
                    w_fc_width += 2 * hidden_size_char
            else:
                raise Exception('BRNN type %s is not supported.' % brnn_type)

    hidden_dropout = tf.nn.dropout(hidden_concat, keep_prob, seed=1, name='hidden_dropout')
    hidden_dropout_reshape = tf.reshape(hidden_dropout, [-1, tf.shape(hidden_concat)[2]])

    # Softmax Layer
    tag_size = tag_embedding_classes - 1
    # w_fc = tf.Variable(tf.truncated_normal([2 * hidden_size, tag_size]), name='w_fc')
    # b_fc = tf.Variable(tf.truncated_normal([tag_size]), name='b_fc')
    w_fc = tf.get_variable("w_fc", dtype=tf.float32,
                           shape=[w_fc_width, tag_size])

    b_fc = tf.get_variable("b_fc", shape=[tag_size],
                           dtype=tf.float32, initializer=tf.constant_initializer(0, dtype=tf.float32))

    logits_flatten = nn_ops.xw_plus_b(hidden_dropout_reshape, w_fc, b_fc, name='logits_flatten')
    logits = tf.reshape(logits_flatten, [-1, tf.shape(hidden_dropout)[1], tag_size], name='logits')

    # CRF layer
    labels = tf.placeholder(tf.int32, [None, None], name='labels')
    log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
        logits, labels, sequence_length)
    # self.trans_params = trans_params # need to evaluate it for decoding
    loss = tf.reduce_mean(-log_likelihood, name='loss')
    tf.add_to_collection('CRF', trans_params)

    # labels_flatten = tf.reshape(labels, [-1])

    # output_tag_ids_int64 = tf.argmax(logits, 2)
    # output_tag_ids = math_ops.cast(output_tag_ids_int64, tf.int32, name='output_tag_ids')
    # correct_prediction = tf.reduce_sum(tf.cast(tf.equal(output_tag_ids, labels), tf.int32), name='correct_prediction')
    # cross_entropy = nn_ops.sparse_softmax_cross_entropy_with_logits(logits, labels)
    #
    # weights = tf.placeholder(tf.float32, [None, None], 'weights')
    # batch_loss = cross_entropy * weights
    # cost = tf.reduce_sum(batch_loss, name='cost')

    # Gradients and SGD update operation for training the model.
    lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
    # params = tf.trainable_variables()
    # opt = tf.train.AdamOptimizer(lr)
    # gradients = tf.gradients(loss, params)
    # if max_gradient_norm > 0:
    #     clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    #     updates = opt.apply_gradients(zip(clipped_gradients, params), name='updates')
    # else:
    #     updates = opt.apply_gradients(zip(gradients, params), name='updates')
    optimizer = tf.train.AdamOptimizer(lr)
    if max_gradient_norm > 0: # gradient clipping if clip is positive
        grads, vs     = zip(*optimizer.compute_gradients(loss))
        grads, gnorm  = tf.clip_by_global_norm(grads, max_gradient_norm)
        updates = optimizer.apply_gradients(zip(grads, vs), name='updates')
    else:
        updates = optimizer.minimize(loss, name='updates')

    # Inference token by token
    # input_tag = tf.placeholder(tf.int32, [None], name='input_tag_id')
    # embedding_tag = embedding_ops.embedding_lookup(tag_embedding, input_tag)
    # input_pos = tf.placeholder(tf.int32, [None], name='input_pos_id')
    # # embedding_pos = embedding_ops.embedding_lookup(pos_embedding, input_pos)
    # input_wrd = tf.placeholder(tf.int32, [None], name='input_wrd_id')
    # embedding_wrd = embedding_ops.embedding_lookup(wrd_embedding, input_wrd)
    # input_fw = tf.concat(1, [embedding_tag, embedding_wrd])
    # if cell_type == "GRU":
    #     state_fw = tf.placeholder(tf.float32, [None, hidden_size], 'input_state_fw')
    # elif cell_type == "LSTM":
    #     state_fw = (tf.placeholder(tf.float32, [None, hidden_size], 'input_state_fw_c'),
    #                 tf.placeholder(tf.float32, [None, hidden_size], 'input_state_fw_h'))
    # else:
    #     raise CustomException("Cell type %s is not supported." % cell_type)
    # with tf.variable_scope('Forward', reuse=True) as scope:
    #     output_fw, state_next_fw = cell_fw(input_fw, state_fw)
    # print(output_fw)
    # print(state_next_fw)
    # hidden_bw = tf.placeholder(tf.float32, [None, hidden_size], 'input_hidden_bw')
    # hidden_concatenated = tf.concat(1, [output_fw, hidden_bw], 'hidden_concatenated')
    # logits_pred = nn_ops.xw_plus_b(hidden_concatenated, w_fc, b_fc, name='logits_pred')
    # tag_id_pred = tf.argmax(logits_pred, 1, name='tag_id_pred')

    # return trans_params


def f_measure(beta, precision, recall):
    if precision == 0.0 or recall == 0.0:
        return 0.0
    else:
        return (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)


class CustomException(Exception):
    def __init__(self, value):
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)


def compute_metrics_memory_efficient(sess,
                                     data_dev,
                                     vocab,
                                     tag_dict,
                                     use_chars,
                                     char_dict,
                                     batch_size,
                                     tags_map,
                                     tag_type_list,
                                     fout=None,
                                     measure_all=False
                                     ):
    T = {tag_type: 0.0 for tag_type in tag_type_list}
    P = {tag_type: 0.0 for tag_type in tag_type_list}
    TP = {tag_type: 0.0 for tag_type in tag_type_list}
    acc = 0
    num_token = 0
    for i, (x_batch, y_batch) in enumerate(fetch_minibatches(data_dev, batch_size)):
        num_token += sum(map(lambda ys: len(ys) - (2 if extend_sequence else 0), y_batch))
        wrd_id_seqs, chr_id_mats, label_seqs = get_data_input(x_batch, y_batch, vocab, tag_dict, use_chars, char_dict)

        # Set the feed dictionary
        feed_dict = {  # 'input_tag_ids:0': tag_id_seqs[data_index:end_index],
            # 'input_pos_ids:0': pos_id_seqs[data_index:end_index],
            'input_wrd_ids:0': wrd_id_seqs,
            'input_chr_ids:0': chr_id_mats,
            # 'initial_hidden_bw:0': np.zeros([len(wrd_id_seqs), hidden_size], np.float32),
            'keep_prob:0': 1.0}
        # print(feed_dict)

        # Predict the tag sequence after updating
        feed_dict['keep_prob:0'] = 1.0
        logits, trans_params = sess.run(['logits:0', tf.get_collection('CRF')[0]], feed_dict=feed_dict)
        # iterate over the sentences because no batching in vitervi_decode
        labels_pred = []
        for logit, tag_seq in zip(logits, y_batch):
            sequence_length = len(tag_seq)
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            if extend_sequence:
                viterbi_seq = viterbi_seq[1:-1]
            labels_pred.append(viterbi_seq)
        labels = [label_seq[:len(tag_seq)] for label_seq, tag_seq in zip(label_seqs, y_batch)]
        if extend_sequence:
            labels = [label_seq[1:-1] for label_seq in labels]
        for lab, lab_pred in zip(labels, labels_pred):
            acc += sum([a == b for (a, b) in zip(lab, lab_pred)])

        for tag_ids_true, tag_ids_pred, wrd_seq in zip(labels, labels_pred, x_batch):
            tags_pred = [tags_map[tag_id_pred] for tag_id_pred in tag_ids_pred]
            chunks_map_pred = get_entity_map(tags_pred)
            tags_true = [tags_map[tag_id_true] for tag_id_true in tag_ids_true]
            chunks_map_true = get_entity_map(tags_true)
            for tag_type in tag_type_list:
                chunks_pred = set(chunks_map_pred[tag_type]) if tag_type in chunks_map_pred else set()
                chunks_true = set(chunks_map_true[tag_type]) if tag_type in chunks_map_true else set()
                TP[tag_type] += len(chunks_pred & chunks_true)
                P[tag_type] += len(chunks_pred)
                T[tag_type] += len(chunks_true)

            if fout:
                tags_pred = [tags_map[tag_id_pred] for tag_id_pred in tag_ids_pred]
                # Decode tags_pred using wrd_seq to get the target concepts in the query
                concepts = {tag_type: [] for tag_type in tag_type_list}
                tag_type_prev = ''
                concept = ''
                for i, tag_curr in enumerate(tags_pred):
                    word = wrd_seq[i]
                    if tag_curr.startswith('B'):
                        tag_type_curr = tag_curr[2:]
                        if concept:
                            concepts[tag_type_prev].append(concept)
                        concept = word
                    elif tag_curr.startswith('I'):
                        tag_type_curr = tag_curr[2:]
                        if tag_type_prev != tag_type_curr:
                            if concept:
                                concepts[tag_type_prev].append(concept)
                            concept = word
                        else:
                            concept += ' ' + word
                    else:
                        tag_type_curr = ''
                        if concept:
                            concepts[tag_type_prev].append(concept)
                        concept = ''
                    tag_type_prev = tag_type_curr
                    # tag_prev = tag_curr
                if concept:
                    concepts[tag_type_prev].append(concept)

                query = ' '.join(wrd_seq)
                fout.write(query)
                for tag_type in tag_type_list:
                    fout.write('\t')
                    fout.write(','.join(concepts[tag_type]))
                fout.write('\n')
    acc /= num_token
    recall = {}
    precision = {}
    F1 = {}
    for tag_type in tag_type_list:
        recall[tag_type] = TP[tag_type] / T[tag_type] if T[tag_type] > 0 else 0.0
        precision[tag_type] = TP[tag_type] / P[tag_type] if P[tag_type] > 0 else 0.0
        F1[tag_type] = f_measure(1.0, precision[tag_type], recall[tag_type])

    if measure_all:
        tp = sum([TP[tag_type] for tag_type in tag_type_list])
        t = sum([T[tag_type] for tag_type in tag_type_list])
        p = sum([P[tag_type] for tag_type in tag_type_list])
        R_all = tp / t
        P_all = tp / p
        F_all = f_measure(1.0, P_all, R_all)
        return recall, precision, F1, R_all, P_all, F_all, acc
    else:
        return recall, precision, F1


def compute_metrics(sess,
                    # pos_id_seq_map_val,
                    wrd_id_seq_map_val,
                    chr_id_mat_map_val,
                    label_seq_map_val,
                    tag_dict,
                    batch_size,
                    hidden_size,
                    tags_map,
                    tag_type_list,
                    fout=None,
                    wrd_seq_map_val=None,
                    measure_all=False
                    ):
    T = {tag_type: 0.0 for tag_type in tag_type_list}
    P = {tag_type: 0.0 for tag_type in tag_type_list}
    TP = {tag_type: 0.0 for tag_type in tag_type_list}
    accs = []
    # ver = tf.__version__.split('.')
    for length, label_seqs in label_seq_map_val.items():
        # During the feed-forwarding process, there's no error like TensorArray has size zero.
        # For training, gradients with TensorArray fail after scatter of 0 size.
        # if ver[0] == '0' and task == 'train':
        #     if length == 1:
        #         continue
        # pos_id_seqs = pos_id_seq_map_val[length]
        wrd_id_seqs = wrd_id_seq_map_val[length]
        chr_id_mats = chr_id_mat_map_val[length]
        if fout:
            wrd_seqs = wrd_seq_map_val[length]
        bucket_size = len(label_seqs)
        data_index = 0
        while True:
            if data_index >= bucket_size:
                break
            end_index = data_index + batch_size
            if end_index > bucket_size:
                end_index = bucket_size
            # pos_id_seq_batch = np.array(pos_id_seqs[data_index:end_index])
            # wrd_id_seq_batch = np.array(wrd_id_seqs[data_index:end_index])
            tag_ids_true_batch = np.array(label_seqs[data_index:end_index])
            tag_ids_pred_batch = np.zeros([end_index - data_index, length], np.int32)
            # prev_tag_ids_batch = np.zeros([end_index - data_index, length])
            if fout:
                wrd_seq_batch = wrd_seqs[data_index:end_index]
                if extend_sequence:
                    wrd_seq_batch = [wrd_seq[1:-1] for wrd_seq in wrd_seqs]
            else:
                wrd_seq_batch = [[] for i in range(end_index - data_index)]

            # Set the feed dictionary
            feed_dict = {  # 'input_tag_ids:0': tag_id_seqs[data_index:end_index],
                # 'input_pos_ids:0': pos_id_seqs[data_index:end_index],
                'input_wrd_ids:0': wrd_id_seqs[data_index:end_index],
                'input_chr_ids:0': chr_id_mats[data_index:end_index],
                # 'labels:0': label_seqs[data_index:end_index],
                # 'weights:0': [[class_id_weight_dict[tag] for tag in label_seq]
                #               for label_seq in label_seqs[data_index:end_index]],
                # 'initial_hidden_bw:0': np.zeros([end_index - data_index, hidden_size], np.float32),
                'keep_prob:0': 1.0}

            # Predict the tag sequence
            logits, trans_params = sess.run(['logits:0', tf.get_collection('CRF')[0]], feed_dict=feed_dict)
            # iterate over the sentences because no batching in vitervi_decode
            labels_pred = []
            for logit in logits:
                # logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                if extend_sequence:
                    viterbi_seq = viterbi_seq[1:-1]
                labels_pred.append(viterbi_seq)
            if extend_sequence:
                labels = tag_ids_true_batch[data_index:end_index, 1:-1]
            else:
                labels = label_seqs[data_index:end_index]

            for lab, lab_pred in zip(labels, labels_pred):
                accs += [a == b for (a, b) in zip(lab, lab_pred)]

            for tag_ids_true, tag_ids_pred, wrd_seq in zip(labels, labels_pred, wrd_seq_batch):
                tags_pred = [tags_map[tag_id_pred] for tag_id_pred in tag_ids_pred]
                chunks_map_pred = get_entity_map(tags_pred)
                tags_true = [tags_map[tag_id_true] for tag_id_true in tag_ids_true]
                chunks_map_true = get_entity_map(tags_true)
                for tag_type in tag_type_list:
                    chunks_pred = set(chunks_map_pred[tag_type]) if tag_type in chunks_map_pred else set()
                    chunks_true = set(chunks_map_true[tag_type]) if tag_type in chunks_map_true else set()
                    TP[tag_type] += len(chunks_pred & chunks_true)
                    P[tag_type] += len(chunks_pred)
                    T[tag_type] += len(chunks_true)

                if fout:
                    tags_pred = [tags_map[tag_id_pred] for tag_id_pred in tag_ids_pred]
                    # Decode tags_pred using wrd_seq to get the target concepts in the query
                    concepts = {tag_type: [] for tag_type in tag_type_list}
                    tag_type_prev = ''
                    concept = ''
                    for i, tag_curr in enumerate(tags_pred):
                        word = wrd_seq[i]
                        if tag_curr.startswith('B'):
                            tag_type_curr = tag_curr[2:]
                            if concept:
                                concepts[tag_type_prev].append(concept)
                            concept = word
                        elif tag_curr.startswith('I'):
                            tag_type_curr = tag_curr[2:]
                            if tag_type_prev != tag_type_curr:
                                if concept:
                                    concepts[tag_type_prev].append(concept)
                                concept = word
                            else:
                                concept += ' ' + word
                        else:
                            tag_type_curr = ''
                            if concept:
                                concepts[tag_type_prev].append(concept)
                            concept = ''
                        tag_type_prev = tag_type_curr
                        # tag_prev = tag_curr
                    if concept:
                        concepts[tag_type_prev].append(concept)

                    query = ' '.join(wrd_seq)
                    fout.write(query)
                    for tag_type in tag_type_list:
                        fout.write('\t')
                        fout.write(','.join(concepts[tag_type]))
                    fout.write('\n')

            data_index = end_index
    acc = np.mean(accs)
    recall = {}
    precision = {}
    F1 = {}
    for tag_type in tag_type_list:
        recall[tag_type] = TP[tag_type] / T[tag_type] if T[tag_type] > 0 else 0.0
        precision[tag_type] = TP[tag_type] / P[tag_type] if P[tag_type] > 0 else 0.0
        F1[tag_type] = f_measure(1.0, precision[tag_type], recall[tag_type])

    if measure_all:
        tp = sum([TP[tag_type] for tag_type in tag_type_list])
        t = sum([T[tag_type] for tag_type in tag_type_list])
        p = sum([P[tag_type] for tag_type in tag_type_list])
        R_all = tp / t
        P_all = tp / p
        F_all = f_measure(1.0, P_all, R_all)
        return recall, precision, F1, R_all, P_all, F_all, acc
    else:
        return recall, precision, F1


def get_chunks(tag_seq):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: ['B-PER', 'I-PER', 'O', 'B-LOC'] sequence of tags

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        tag_seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        result = [('PER', 0, 2), ('LOC', 3, 4)]

    """
    default = O
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tag in enumerate(tag_seq):
        # End of a chunk 1
        if tag == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tag != default:
            parts = tag.split('-')
            tag_chunk_class, tag_chunk_type = parts[0], parts[1]
            if chunk_type is None:
                chunk_type, chunk_start = tag_chunk_type, i
            elif tag_chunk_type != chunk_type or tag_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tag_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(tag_seq))
        chunks.append(chunk)

    return chunks


def get_entities(tag_seq):
    """Given a sequence of tags, group entities and their position

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        tag_seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        result = [('PER', 0, 2), ('LOC', 3, 4)]

    """

    # concepts = {tag_type: [] for tag_type in tag_type_list}
    concepts = []
    tag_type_prev = ''
    concept = ''
    for i, tag_curr in enumerate(tag_seq):
        word = i
        if tag_curr.startswith('B'):
            tag_type_curr = tag_curr[2:]
            if concept:
                chunk = (tag_type_prev, concept[0], concept[-1] + 1)
                # concepts[tag_type_prev].append(chunk)
                concepts.append(chunk)
            concept = [word]
        elif tag_curr.startswith('I'):
            tag_type_curr = tag_curr[2:]
            if tag_type_prev != tag_type_curr:
                if concept:
                    chunk = (tag_type_prev, concept[0], concept[-1] + 1)
                    # concepts[tag_type_prev].append(chunk)
                    concepts.append(chunk)
                concept = [word]
            else:
                concept += [word]
        else:
            tag_type_curr = ''
            if concept:
                chunk = (tag_type_prev, concept[0], concept[-1] + 1)
                # concepts[tag_type_prev].append(chunk)
                concepts.append(chunk)
            concept = []
        tag_type_prev = tag_type_curr
        # tag_prev = tag_curr
    if concept:
        chunk = (tag_type_prev, concept[0], concept[-1] + 1)
        # concepts[tag_type_prev].append(chunk)
        concepts.append(chunk)
    return concepts


def get_entity_map(tag_seq):
    """Given a sequence of tags, group entities and their position

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        tag_seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        result = {'PER': [('PER', 0, 2)], 'LOC': [('LOC', 3, 4)]}

    """
    # Evaluation results on testb.bio show that get_chunks and get_entities return the same results.
    # Evaluation using get_chunks costs 19.8729s
    # Evaluation using get_entities costs 17.9526s
    # entities = get_chunks(tag_seq)
    entities = get_entities(tag_seq)
    entity_map = {key: list(group) for key, group in itertools.groupby(entities, lambda t: t[0])}
    return entity_map


def softmax(z):
    # assert len(z.shape) == 2
    if len(z.shape) == 1:
        z = np.expand_dims(z, 0)
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


################################################################################
# Training                                                                     #
################################################################################
def train(data_dir, training_filename, validation_filename, model_dir,
          embedding_size, hidden_size, keep_prob, learning_rate, batch_size, acc_stop,
          chopoff, resume):
    global max_iter
    begin = time.time()
    # Cannot use global filename as filename is used both as a parameter and a global
    # Here filename's value only comes from the function input argument filename rather
    # than the global parameter filename.
    print("Training on %s" % os.path.join(data_dir, training_filename))

    if use_chars:
        char_dict = build_char_dictionary()
    else:
        char_dict = {}

    pickle_path = os.path.join(model_dir, 'training_data')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        tag_types = set()
        if chopoff >= 1:
            # Build and chop rare terms off vocabulary
            # counts = compute_counts(data_dir, training_filename, tag_types)
            counts = compute_counts(data_dir, training_filename, tag_types, None,
                                    False if validation_filename else True)
            if validation_filename:
                counts = compute_counts(data_dir, validation_filename, tag_types, counts, True)
            save_count_file(model_dir, counts)
            vocab = {PAD: 0, UNK: 1, NUM: 2}
            if extend_sequence:
                vocab[STT] = len(vocab)
                vocab[END] = len(vocab)
            for (token, count) in [pair for pair in counts if pair[1] >= chopoff]:
                vocab[token] = len(vocab)
            update_vocab = False
            tag_type_list = sorted(tag_types)
            tag_dict = build_tag_dictionary(tag_type_list)
        else:
            vocab = {PAD: 0, UNK: 1, NUM: 2}
            if extend_sequence:
                vocab[STT] = len(vocab)
                vocab[END] = len(vocab)
            update_vocab = True
            tag_type_list = []
            tag_dict = {}
            # TODO
            # If tag_dict is empty, both B and I would be empty, then parsing training data will be problematic.
        print('max_word_length:', max_word_length)
        save_max_word_length(max_word_length)

        # pos_dict = build_pos_dictionary()

        # tag_id_seq_map, wrd_id_seq_map, label_seq_map, vocab = \
        #     build_training_data(data_dir, training_filename, tag_dict, vocab, update_vocab)

        train_path = os.path.join(data_dir, training_filename)
        # tag_id_seq_map = {}
        # pos_id_seq_map = {}
        wrd_id_seq_map = {}
        chr_id_mat_map = {}
        label_seq_map = {}
        if annotation_scheme == 'CoNLL':
            with open(train_path, 'r', encoding=encoding) as f:
                split_pattern = re.compile(r'[ \t]')
                wrd_seq, tag_seq = [], []
                if extend_sequence:
                    wrd_seq += [STT]
                    tag_seq += [O]
                cnt = 0
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('-DOCSTART-'):
                        if extend_sequence and len(wrd_seq) > 1 or not extend_sequence and wrd_seq:
                            if extend_sequence:
                                wrd_seq += [END]
                                tag_seq += [O]
                            wrd_id_seq = [vocab[w if w in vocab else NUM if is_number(w) else UNK] for w in wrd_seq]
                            label_seq = [tag_dict[tag] for tag in tag_seq]
                            if use_chars:
                                chr_id_mat = build_sequence_from_a_word_sequence_in_char_level(wrd_seq, char_dict,
                                                                                               max_word_length)
                            T = len(wrd_seq)
                            if T not in label_seq_map:
                                label_seq_map[T] = []
                                # tag_id_seq_map[T] = []
                                # pos_id_seq_map[T] = []
                                wrd_id_seq_map[T] = []
                                if use_chars:
                                    chr_id_mat_map[T] = []
                            label_seq_map[T].append(label_seq)
                            # tag_id_seq_map[T].append(prev_tag_id_seq)
                            # pos_id_seq_map[T].append(pos_id_seq)
                            wrd_id_seq_map[T].append(wrd_id_seq)
                            if use_chars:
                                chr_id_mat_map[T].append(chr_id_mat)
                            cnt += 1
                            if cnt % 100 == 0:
                                print("  read %d tagged examples" % cnt, end="\r")
                            wrd_seq, tag_seq = [], []
                            if extend_sequence:
                                wrd_seq += [STT]
                                tag_seq += [O]
                        continue
                    # container = line.split(' \t')
                    container = split_pattern.split(line)
                    token, tag = container[0], container[-1]
                    token = token.lower()
                    tag = tag.upper()
                    wrd_seq.append(token)
                    tag_seq.append(tag)
                print("  read %d tagged examples" % cnt)
        else:
            with open(train_path, "r", encoding=encoding) as f:
                cnt = 0
                for tagged_query in f:
                    tagged_query = tagged_query.strip()
                    if not tagged_query:
                        continue
                    prev_tag_id_seq, wrd_id_seq, label_seq, wrd_seq = build_sequence_from_a_tagged_query(
                        tagged_query, tag_dict, vocab, update_vocab)
                    if use_chars:
                        chr_id_mat = build_sequence_from_a_word_sequence_in_char_level(wrd_seq, char_dict,
                                                                                       max_word_length)
                    T = len(label_seq)
                    if T not in label_seq_map:
                        label_seq_map[T] = []
                        # tag_id_seq_map[T] = []
                        # pos_id_seq_map[T] = []
                        wrd_id_seq_map[T] = []
                        if use_chars:
                            chr_id_mat_map[T] = []
                    label_seq_map[T].append(label_seq)
                    # tag_id_seq_map[T].append(prev_tag_id_seq)
                    # pos_id_seq_map[T].append(pos_id_seq)
                    wrd_id_seq_map[T].append(wrd_id_seq)
                    if use_chars:
                        chr_id_mat_map[T].append(chr_id_mat)
                    cnt += 1
                    if cnt % 100 == 0:
                        print("  read %d tagged examples" % cnt, end="\r")
                print("  read %d tagged examples" % cnt)

        save_dictionary(model_dir, vocab)
        save_tag_type_list(model_dir, tag_type_list)
        f = open(pickle_path, 'wb')
        # pickle.dump(pos_id_seq_map, f)
        pickle.dump(wrd_id_seq_map, f)
        pickle.dump(chr_id_mat_map, f)
        pickle.dump(label_seq_map, f)
        f.close()
        print('training data was saved at %s' % pickle_path)
        # if pretrained_embedding_path.find('glove') != -1:
        #     # trimmed_filepath = os.path.join(model_dir, os.path.basename(pretrained_embedding_path) + '.trimmed.npz')
        #     trimmed_filepath = pretrained_embedding_path + '.trimmed.npz'
        #     glove_embedding_filepath = pretrained_embedding_path + '.npz'
        #     if not os.path.exists(glove_embedding_filepath):
        #         # export_trimmed_glove_vectors(vocab, pretrained_embedding_path, trimmed_filepath, embedding_size)
        #         export_glove_vectors(pretrained_embedding_path, glove_embedding_filepath)
    else:
        # Load existing formatted training data and dictionaries
        with open(pickle_path, 'rb') as handle:
            # pos_id_seq_map = pickle.load(handle)
            wrd_id_seq_map = pickle.load(handle)
            chr_id_mat_map = pickle.load(handle)
            label_seq_map = pickle.load(handle)
        tag_type_list = load_tag_type_list(model_dir)
        tag_dict = build_tag_dictionary(tag_type_list)
        # tag_size = len(tag_dict)
        # pos_dict = build_pos_dictionary()
        vocab = load_dictionary(model_dir)

    tags_map = [""] * len(tag_dict)
    for (tag, index) in tag_dict.items():
        tags_map[index] = tag

    label_counts = [0. for i in range(len(tag_dict) - 1)]
    for T, label_seqs in label_seq_map.items():
        for label_seq in label_seqs:
            for label in label_seq:
                label_counts[label] += 1.
    class_weights = {
    tags_map[tag_id]: label_counts[tag_dict[O]] / label_counts[tag_id] if label_counts[tag_id] > 0. else 0.
    for tag_id in range(len(tag_dict) - 1) if tag_id != tag_dict[N]}
    print('class_weights:', class_weights)
    class_id_weight_dict = build_class_id_weight_dict(tag_dict, class_weights)
    save_class_weights(model_dir, class_weights)

    if validation_filename:
        validation_filepath = os.path.join(data_dir, validation_filename)
        print("Validation on %s" % validation_filepath)
        cnt = 0
        # tag_id_seq_map_val = {}
        # pos_id_seq_map_val = {}
        wrd_id_seq_map_val = {}
        label_seq_map_val = {}
        wrd_seq_map_val = {}
        chr_id_mat_map_val = {}
        if annotation_scheme == 'CoNLL':
            with open(validation_filepath, 'r', encoding=encoding) as f:
                split_pattern = re.compile(r'[ \t]')
                wrd_seq, tag_seq = [], []
                if extend_sequence:
                    wrd_seq += [STT]
                    tag_seq += [O]
                cnt = 0
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('-DOCSTART-'):
                        if extend_sequence and len(wrd_seq) > 1 or not extend_sequence and wrd_seq:
                            if extend_sequence:
                                wrd_seq += [END]
                                tag_seq += [O]
                            wrd_id_seq = [vocab[w if w in vocab else NUM if is_number(w) else UNK] for w in wrd_seq]
                            label_seq = [tag_dict[tag] for tag in tag_seq]
                            if use_chars:
                                chr_id_mat = build_sequence_from_a_word_sequence_in_char_level(wrd_seq, char_dict,
                                                                                               max_word_length)
                            T = len(wrd_seq)
                            if T not in label_seq_map_val:
                                label_seq_map_val[T] = []
                                # tag_id_seq_map[T] = []
                                # pos_id_seq_map[T] = []
                                wrd_id_seq_map_val[T] = []
                                if use_chars:
                                    chr_id_mat_map_val[T] = []
                            label_seq_map_val[T].append(label_seq)
                            # tag_id_seq_map[T].append(prev_tag_id_seq)
                            # pos_id_seq_map[T].append(pos_id_seq)
                            wrd_id_seq_map_val[T].append(wrd_id_seq)
                            if use_chars:
                                chr_id_mat_map_val[T].append(chr_id_mat)
                            cnt += 1
                            if cnt % 500 == 0:
                                print("  formatted %d tagged examples" % cnt, end="\r")
                            wrd_seq, tag_seq = [], []
                            if extend_sequence:
                                wrd_seq += [STT]
                                tag_seq += [O]
                        continue
                    # container = line.split(' \t')
                    container = split_pattern.split(line)
                    token, tag = container[0], container[-1]
                    token = token.lower()
                    tag = tag.upper()
                    wrd_seq.append(token)
                    tag_seq.append(tag)
                print("  formatted %d tagged examples" % cnt)
        else:
            f = open(validation_filepath, 'r')

            # open_tag = '{' if model_type == ModelType.ANSWER else '['
            # close_tag = '}' if model_type == ModelType.ANSWER else ']'

            for tagged_query in f:
                tagged_query = tagged_query.strip()
                if not tagged_query:
                    continue
                prev_tag_id_seq, wrd_id_seq, label_seq, wrd_seq = build_sequence_from_a_tagged_query(
                    tagged_query, tag_dict, vocab, update_vocab=False)
                if use_chars:
                    chr_id_mat = build_sequence_from_a_word_sequence_in_char_level(wrd_seq, char_dict, max_word_length)
                T = len(label_seq)
                if T not in label_seq_map_val:
                    label_seq_map_val[T] = []
                    # tag_id_seq_map_val[T] = []
                    # pos_id_seq_map_val[T] = []
                    wrd_id_seq_map_val[T] = []
                    wrd_seq_map_val[T] = []
                    if use_chars:
                        chr_id_mat_map_val[T] = []
                label_seq_map_val[T].append(label_seq)
                # tag_id_seq_map_val[T].append(prev_tag_id_seq)
                # pos_id_seq_map_val[T].append(pos_id_seq)
                wrd_id_seq_map_val[T].append(wrd_id_seq)
                wrd_seq_map_val[T].append(wrd_seq)
                if use_chars:
                    chr_id_mat_map_val[T].append(chr_id_mat)
                cnt += 1
                if cnt % 500 == 0:
                    print("  formatted %d tagged examples" % cnt, end="\r")
            print("  formatted %d tagged examples" % cnt)
            f.close()

    ################################################################################
    # Start session
    ################################################################################

    sess = tf.Session()

    tag_embedding_classes = len(tag_dict)
    char_embedding_classes = len(char_dict)
    # pos_embedding_classes = len(pos_dict)
    wrd_embedding_classes = len(vocab)
    create_model(model_dir,
                 char_embedding_classes,
                 tag_embedding_classes,
                 # pos_embedding_classes,
                 wrd_embedding_classes,
                 embedding_size,
                 hidden_size,
                 vocab,
                 )
    # for variable in tf.global_variables():
    #     print(variable)

    # Training loop
    counter = 0
    start = time.time()
    saver = tf.train.Saver(tf.global_variables())
    if resume and tf.train.get_checkpoint_state(model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        model_checkpoint_path = ckpt.model_checkpoint_path
        if not os.path.exists(model_checkpoint_path):
            model_checkpoint_path = os.path.join(model_dir, os.path.basename(model_checkpoint_path))
        saver.restore(sess, model_checkpoint_path)
        model_checkpoint_name = os.path.basename(model_checkpoint_path)
        epoch = int(model_checkpoint_name[model_checkpoint_name.rindex('-') + 1:])
    else:
        # tf.add_to_collection('outputs', outputs)
        sess.run(tf.global_variables_initializer())
        epoch = 0
        # Use a saver_def to get the "magic" strings to restore
        saver_def = saver.as_saver_def()
        print(saver_def.filename_tensor_name)
        print(saver_def.restore_op_name)
        tf.train.write_graph(sess.graph_def, '.', os.path.join(model_dir, 'model.proto'), as_text=False)
        tf.train.write_graph(sess.graph_def, '.', os.path.join(model_dir, 'model.txt'), as_text=True)

    data_size = sum([len(label_seqs) for label_seqs in label_seq_map.values()])
    num_token = sum([len(label_seqs) * length for length, label_seqs in label_seq_map.items()])

    print("data_size: %s" % data_size)

    recall_best = {tag_type: 0.0 for tag_type in tag_type_list}
    precision_best = {tag_type: 0.0 for tag_type in tag_type_list}
    F1_best = {tag_type: 0.0 for tag_type in tag_type_list}
    F_all_best = 0.0
    epoch_best = 0
    loss_best = 0.0
    accuracy_best = 0.0
    acc_dev_best = 0.0
    lr = learning_rate
    ver = tf.__version__.split('.')
    save_TF_version(model_dir)
    while True:

        accuracy = 0.0
        total_loss = 0.0
        processed = 0

        # Train the model for one pass of training data
        for T, label_seqs in label_seq_map.items():
            # print('T:', T)
            # print('len(label_seqs):', len(label_seqs))
            # if ver[0] == '0' or ver[0] == '1':
            #     if T == 1:
            #         continue
            if not extend_sequence:
                if T == 1:
                    continue
            # tag_id_seqs = tag_id_seq_map[T]
            # pos_id_seqs = pos_id_seq_map[T]
            wrd_id_seqs = wrd_id_seq_map[T]
            chr_id_mats = chr_id_mat_map[T]
            bucket_size = len(label_seqs)
            data_index = 0
            # Train the model in a bucket of training data
            while True:
                if data_index >= bucket_size:
                    break
                end_index = data_index + batch_size
                if end_index > bucket_size:
                    end_index = bucket_size
                print("  training [%d, %d] of %d" % (processed + data_index,
                                                     processed + end_index,
                                                     data_size), end="\r")
                # Set the feed dictionary
                feed_dict = {  # 'input_tag_ids:0': tag_id_seqs[data_index:end_index],
                    # 'input_pos_ids:0': pos_id_seqs[data_index:end_index],
                    'input_wrd_ids:0': wrd_id_seqs[data_index:end_index],
                    'input_chr_ids:0': chr_id_mats[data_index:end_index],
                    'labels:0': label_seqs[data_index:end_index],
                    # 'weights:0': [[class_id_weight_dict[tag] for tag in label_seq]
                    #               for label_seq in label_seqs[data_index:end_index]],
                    # 'initial_hidden_bw:0': np.zeros([end_index - data_index, hidden_size], np.float32),
                    'keep_prob:0': keep_prob,
                    'lr:0': lr}
                # print(feed_dict)
                loss, _ = \
                    sess.run(['loss:0',
                              # 'logits:0',
                              # 'correct_prediction:0',
                              'updates'],
                             feed_dict=feed_dict)
                loss *= (end_index - data_index)
                # if T == 4:
                #     print(tag_id_seqs[data_index:end_index])
                #     print(label_seqs[data_index:end_index])
                #     print(output_tag_ids)
                #     print()

                # print(label_seqs[data_index:end_index])
                # print(output_tag_ids)
                # print()

                # Predict the tag sequence after updating
                feed_dict['keep_prob:0'] = 1.0
                logits, trans_params = sess.run(['logits:0', tf.get_collection('CRF')[0]], feed_dict=feed_dict)
                # iterate over the sentences because no batching in vitervi_decode
                labels_pred = []
                for logit in logits:
                    # logit = logit[:sequence_length] # keep only the valid steps
                    viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                    labels_pred.append(viterbi_seq)
                labels = label_seqs[data_index:end_index]
                for lab, lab_pred in zip(labels, labels_pred):
                    accuracy += sum([a == b for (a, b) in zip(lab, lab_pred)])
                total_loss += loss
                data_index += batch_size
                counter += 1
            processed += bucket_size
        epoch += 1
        # print("Total training loss: %f, epoch: %d" % (total_loss, epoch))
        accuracy /= num_token
        lr *= decay
        # print("Training accuracy: %f, epoch: %d" % (accuracy, epoch))
        if validation_filename:
            R, P, F, R_all, P_all, F_all, acc = \
                compute_metrics(sess,
                                # pos_id_seq_map_val,
                                wrd_id_seq_map_val,
                                chr_id_mat_map_val,
                                label_seq_map_val,
                                tag_dict,
                                batch_size,
                                hidden_size,
                                tags_map,
                                tag_type_list,
                                measure_all=True
                                )
            print("Epoch %d - training loss: %.4f acc: %.4f - val" % (epoch, total_loss, accuracy), end='')
            # for tag_type in tag_type_list:
            #     print(" %s: r%.4f p%.4f f%.4f" % (tag_type, R[tag_type], P[tag_type], F[tag_type]), end='')
            print(" F1: %.2f - acc: %.2f" % (100 * F_all, 100 * acc), end='')
            print()
            # F1_avg = sum(F.values()) / len(F)
            if F_all > F_all_best:
                epoch_best = epoch
                loss_best = total_loss
                accuracy_best = accuracy
                recall_best = R
                precision_best = P
                F1_best = F
                F_all_best = F_all
                acc_dev_best = acc
                now = time.time()
                delta = now - start
                print("Counter: %d, delta time = %f" % (counter, delta))
                start = time.time()
                checkpoint_path = os.path.join(model_dir, checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=epoch)
        else:
            print("Epoch %d\t- training loss: %.4f\taccuracy: %.4f" % (epoch, total_loss, accuracy))
            if epoch % 5 == 0:
                now = time.time()
                delta = now - start
                print("Counter: %d, delta time = %f" % (counter, delta))
                start = time.time()
                checkpoint_path = os.path.join(model_dir, checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=epoch)

        if acc_stop and accuracy >= acc_stop:
            if not validation_filename:
                now = time.time()
                delta = now - start
                print("Counter: %d, delta time = %f" % (counter, delta))
                start = time.time()
                checkpoint_path = os.path.join(model_dir, checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=epoch)
            break
        elif 0 <= max_iter <= epoch:
            if not validation_filename:
                now = time.time()
                delta = now - start
                print("Counter: %d, delta time = %f" % (counter, delta))
                start = time.time()
                checkpoint_path = os.path.join(model_dir, checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=epoch)
            break
    elapsed_time = time.time() - begin
    if validation_filename:
        print("Best model - epoch %d - loss %.4f - acc %.4f - val" % (epoch_best, loss_best, accuracy_best), end='')
        for tag_type in tag_type_list:
            print(" %s: r%.4f p%.4f f%.4f" % (
            tag_type, recall_best[tag_type], precision_best[tag_type], F1_best[tag_type]), end='')
        print()
        print("F1: %.2f - acc: %.2f" % (100 * F_all_best, 100 * acc_dev_best))
        # Save validation metrics
        with open(os.path.join(model_dir, 'validation_metrics.txt'), 'w', encoding=encoding) as f:
            f.write("Best model - epoch %d - training loss %.4f - acc %.4f\n" % (epoch_best, loss_best, accuracy_best))
            f.write("Model name: %s\n" % os.path.basename(model_dir))
            f.write("F1: %.2f - acc: %.2f\n" % (100 * F_all_best, 100 * acc_dev_best))
            for tag_type in tag_type_list:
                f.write("%s: r%.4f p%.4f f%.4f\n" % (
                tag_type, recall_best[tag_type], precision_best[tag_type], F1_best[tag_type]))
            f.write("Elapsed time: %.4fs\n" % elapsed_time)
    print("Elapsed time: %.4fs\n" % elapsed_time)


def train_memory_efficient():
    begin = time.time()
    # Cannot use global filename as filename is used both as a parameter and a global
    # Here filename's value only comes from the function input argument filename rather
    # than the global parameter filename.
    print("Training on %s" % os.path.join(data_dir, training_filename))

    if use_chars:
        char_dict = build_char_dictionary()
    else:
        char_dict = {}

    # pickle_path = os.path.join(model_dir, 'training_data')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        tag_types = set()
        if chopoff >= 1:
            # Build and chop rare terms off vocabulary
            counts = compute_counts(data_dir, training_filename, tag_types, None,
                                    False if validation_filename else True)
            if validation_filename:
                counts = compute_counts(data_dir, validation_filename, tag_types, counts, True)
            save_count_file(model_dir, counts)
            vocab = {PAD: 0, UNK: 1, NUM: 2}
            glove_vocab = None
            if pretrained_embedding_path and pretrained_embedding_path.find('glove') != -1:
                glove_vocab = get_glove_vocab(pretrained_embedding_path)
            if extend_sequence:
                vocab[STT] = len(vocab)
                vocab[END] = len(vocab)
            for (token, count) in [pair for pair in counts if pair[1] >= chopoff]:
                if glove_vocab and token not in glove_vocab:
                    continue
                vocab[token] = len(vocab)
            update_vocab = False
            tag_type_list = sorted(tag_types)
            tag_dict = build_tag_dictionary(tag_type_list)
        else:
            vocab = {PAD: 0, UNK: 1, NUM: 2}
            if extend_sequence:
                vocab[STT] = len(vocab)
                vocab[END] = len(vocab)
            update_vocab = True
            tag_type_list = []
            tag_dict = {}
            # TODO
            # If tag_dict is empty, both B and I would be empty, then parsing training data will be problematic.
        print('max_word_length:', max_word_length)
        save_max_word_length(max_word_length)

        print('Vocabulary size: %s' % len(vocab))

        # pos_dict = build_pos_dictionary()

        save_dictionary(model_dir, vocab)
        save_tag_type_list(model_dir, tag_type_list)
        # f = open(pickle_path, 'wb')
        # # pickle.dump(pos_id_seq_map, f)
        # pickle.dump(wrd_id_seq_map, f)
        # pickle.dump(chr_id_mat_map, f)
        # pickle.dump(label_seq_map, f)
        # f.close()
        # print('training data was saved at %s' % pickle_path)
        # if pretrained_embedding_path.find('glove') != -1:
        #     # trimmed_filepath = os.path.join(model_dir, os.path.basename(pretrained_embedding_path) + '.trimmed.npz')
        #     trimmed_filepath = pretrained_embedding_path + '.trimmed.npz'
        #     glove_embedding_filepath = pretrained_embedding_path + '.npz'
        #     if not os.path.exists(glove_embedding_filepath):
        #         # export_trimmed_glove_vectors(vocab, pretrained_embedding_path, trimmed_filepath, embedding_size)
        #         export_glove_vectors(pretrained_embedding_path, glove_embedding_filepath)
    else:
        # Load existing formatted training data and dictionaries
        # with open(pickle_path, 'rb') as handle:
        #     # pos_id_seq_map = pickle.load(handle)
        #     wrd_id_seq_map = pickle.load(handle)
        #     chr_id_mat_map = pickle.load(handle)
        #     label_seq_map = pickle.load(handle)
        tag_type_list = load_tag_type_list(model_dir)
        tag_dict = build_tag_dictionary(tag_type_list)
        # tag_size = len(tag_dict)
        # pos_dict = build_pos_dictionary()
        vocab = load_dictionary(model_dir)

    path_log = os.path.join(model_dir, "log.txt")
    logger = get_logger(path_log)

    tags_map = [""] * len(tag_dict)
    for (tag, index) in tag_dict.items():
        tags_map[index] = tag

    # label_counts = [0. for i in range(len(tag_dict) - 1)]
    # for T, label_seqs in label_seq_map.items():
    #     for label_seq in label_seqs:
    #         for label in label_seq:
    #             label_counts[label] += 1.
    # class_weights = {tags_map[tag_id]: label_counts[tag_dict[O]] / label_counts[tag_id] if label_counts[tag_id] > 0. else 0.
    #                  for tag_id in range(len(tag_dict) - 1) if tag_id != tag_dict[N]}
    # print('class_weights:', class_weights)
    # class_id_weight_dict = build_class_id_weight_dict(tag_dict, class_weights)
    # save_class_weights(model_dir, class_weights)
    load_validation_data = not True
    if validation_filename:
        validation_filepath = os.path.join(data_dir, validation_filename)
        print("Validation on %s" % validation_filepath)
        if load_validation_data:
            cnt = 0
            # tag_id_seq_map_val = {}
            # pos_id_seq_map_val = {}
            wrd_id_seq_map_val = {}
            label_seq_map_val = {}
            wrd_seq_map_val = {}
            chr_id_mat_map_val = {}
            if annotation_scheme == 'CoNLL':
                with open(validation_filepath, 'r', encoding=encoding) as f:
                    split_pattern = re.compile(r'[ \t]')
                    wrd_seq, tag_seq = [], []
                    if extend_sequence:
                        wrd_seq += [STT]
                        tag_seq += [O]
                    cnt = 0
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('-DOCSTART-'):
                            if extend_sequence and len(wrd_seq) > 1 or not extend_sequence and wrd_seq:
                                if extend_sequence:
                                    wrd_seq += [END]
                                    tag_seq += [O]
                                wrd_id_seq = [vocab[w if w in vocab else NUM if is_number(w) else UNK] for w in wrd_seq]
                                label_seq = [tag_dict[tag] for tag in tag_seq]
                                if use_chars:
                                    chr_id_mat = build_sequence_from_a_word_sequence_in_char_level(wrd_seq, char_dict,
                                                                                                   max_word_length)
                                T = len(wrd_seq)
                                if T not in label_seq_map_val:
                                    label_seq_map_val[T] = []
                                    # tag_id_seq_map[T] = []
                                    # pos_id_seq_map[T] = []
                                    wrd_id_seq_map_val[T] = []
                                    if use_chars:
                                        chr_id_mat_map_val[T] = []
                                label_seq_map_val[T].append(label_seq)
                                # tag_id_seq_map[T].append(prev_tag_id_seq)
                                # pos_id_seq_map[T].append(pos_id_seq)
                                wrd_id_seq_map_val[T].append(wrd_id_seq)
                                if use_chars:
                                    chr_id_mat_map_val[T].append(chr_id_mat)
                                cnt += 1
                                if cnt % 500 == 0:
                                    print("  formatted %d tagged examples" % cnt, end="\r")
                                wrd_seq, tag_seq = [], []
                                if extend_sequence:
                                    wrd_seq += [STT]
                                    tag_seq += [O]
                            continue
                        # container = line.split(' \t')
                        container = split_pattern.split(line)
                        token, tag = container[0], container[-1]
                        token = token.lower()
                        tag = tag.upper()
                        wrd_seq.append(token)
                        tag_seq.append(tag)
                    print("  formatted %d tagged examples" % cnt)
            else:
                f = open(validation_filepath, 'r')

                # open_tag = '{' if model_type == ModelType.ANSWER else '['
                # close_tag = '}' if model_type == ModelType.ANSWER else ']'

                for tagged_query in f:
                    tagged_query = tagged_query.strip()
                    if not tagged_query:
                        continue
                    prev_tag_id_seq, wrd_id_seq, label_seq, wrd_seq = build_sequence_from_a_tagged_query(
                        tagged_query, tag_dict, vocab, update_vocab=False)
                    if use_chars:
                        chr_id_mat = build_sequence_from_a_word_sequence_in_char_level(wrd_seq, char_dict,
                                                                                       max_word_length)
                    T = len(label_seq)
                    if T not in label_seq_map_val:
                        label_seq_map_val[T] = []
                        # tag_id_seq_map_val[T] = []
                        # pos_id_seq_map_val[T] = []
                        wrd_id_seq_map_val[T] = []
                        wrd_seq_map_val[T] = []
                        if use_chars:
                            chr_id_mat_map_val[T] = []
                    label_seq_map_val[T].append(label_seq)
                    # tag_id_seq_map_val[T].append(prev_tag_id_seq)
                    # pos_id_seq_map_val[T].append(pos_id_seq)
                    wrd_id_seq_map_val[T].append(wrd_id_seq)
                    wrd_seq_map_val[T].append(wrd_seq)
                    if use_chars:
                        chr_id_mat_map_val[T].append(chr_id_mat)
                    cnt += 1
                    if cnt % 500 == 0:
                        print("  formatted %d tagged examples" % cnt, end="\r")
                print("  formatted %d tagged examples" % cnt)
                f.close()
        else:
            pass

    ################################################################################
    # Start session
    ################################################################################

    sess = tf.Session()

    tag_embedding_classes = len(tag_dict)
    char_embedding_classes = len(char_dict)
    # pos_embedding_classes = len(pos_dict)
    wrd_embedding_classes = len(vocab)
    create_model(model_dir,
                 char_embedding_classes,
                 tag_embedding_classes,
                 # pos_embedding_classes,
                 wrd_embedding_classes,
                 embedding_size,
                 hidden_size,
                 vocab,
                 )
    # for variable in tf.global_variables():
    #     print(variable)

    # Training loop
    counter = 0
    start = time.time()
    saver = tf.train.Saver(tf.global_variables())
    if resume and tf.train.get_checkpoint_state(model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        model_checkpoint_path = ckpt.model_checkpoint_path
        if not os.path.exists(model_checkpoint_path):
            model_checkpoint_path = os.path.join(model_dir, os.path.basename(model_checkpoint_path))
        saver.restore(sess, model_checkpoint_path)
        model_checkpoint_name = os.path.basename(model_checkpoint_path)
        epoch = int(model_checkpoint_name[model_checkpoint_name.rindex('-') + 1:])
    else:
        # tf.add_to_collection('outputs', outputs)
        sess.run(tf.global_variables_initializer())
        epoch = 0
        # Use a saver_def to get the "magic" strings to restore
        saver_def = saver.as_saver_def()
        print(saver_def.filename_tensor_name)
        print(saver_def.restore_op_name)
        # tf.train.write_graph(sess.graph_def, '.', os.path.join(model_dir, 'model.proto'), as_text=False)
        # tf.train.write_graph(sess.graph_def, '.', os.path.join(model_dir, 'model.txt'), as_text=True)

    train_path = os.path.join(data_dir, training_filename)
    data_train = TextDataset(train_path, annotation_scheme, extend_sequence)
    if validation_filename and not load_validation_data:
        validation_filepath = os.path.join(data_dir, validation_filename)
        data_dev = TextDataset(validation_filepath, annotation_scheme, extend_sequence)

    data_size = len(data_train)
    print("data_size: %s" % data_size)

    recall_best = {tag_type: 0.0 for tag_type in tag_type_list}
    precision_best = {tag_type: 0.0 for tag_type in tag_type_list}
    F1_best = {tag_type: 0.0 for tag_type in tag_type_list}
    F_all_best = 0.0
    epoch_best = 0
    loss_best = 0.0
    accuracy_best = 0.0
    acc_dev_best = 0.0
    lr = learning_rate
    ver = tf.__version__.split('.')
    save_TF_version(model_dir)
    while True:

        accuracy = 0.0
        total_loss = 0.0
        num_token = 0

        # One epoch
        nbatches = (len(data_train) + batch_size - 1) // batch_size

        # iterate over dataset
        processed = 0
        for i, (x_batch, y_batch) in enumerate(fetch_minibatches(data_train, batch_size)):

            num_token += sum(map(lambda ys: len(ys) - (2 if extend_sequence else 0), y_batch))
            wrd_id_seqs, chr_id_mats, label_seqs = get_data_input(x_batch, y_batch, vocab, tag_dict, use_chars,
                                                                  char_dict)
            # Set the feed dictionary
            feed_dict = {  # 'input_tag_ids:0': tag_id_seqs[data_index:end_index],
                # 'input_pos_ids:0': pos_id_seqs[data_index:end_index],
                'input_wrd_ids:0': wrd_id_seqs,
                'input_chr_ids:0': chr_id_mats,
                'labels:0': label_seqs,
                # 'weights:0': [[class_id_weight_dict[tag] for tag in label_seq]
                #               for label_seq in label_seqs[data_index:end_index]],
                # 'initial_hidden_bw:0': np.zeros([len(wrd_id_seqs), hidden_size], np.float32),
                'keep_prob:0': keep_prob,
                'lr:0': lr}
            # print(feed_dict)
            _, loss = sess.run(['updates',
                                'loss:0',
                                # 'logits:0',
                                # 'correct_prediction:0',
                                ],
                               feed_dict=feed_dict)
            print("  trained [%d, %d] of %d - batch loss %.4f" % (processed,
                                                                  processed + len(y_batch),
                                                                  data_size, loss), end="\r")
            loss *= len(y_batch)

            # Predict the tag sequence after updating
            feed_dict['keep_prob:0'] = 1.0
            logits, trans_params = sess.run(['logits:0', tf.get_collection('CRF')[0]], feed_dict=feed_dict)
            # iterate over the sentences because no batching in vitervi_decode
            labels_pred = []
            for logit, tag_seq in zip(logits, y_batch):
                sequence_length = len(tag_seq)
                logit = logit[:sequence_length]  # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                if extend_sequence:
                    viterbi_seq = viterbi_seq[1:-1]
                labels_pred.append(viterbi_seq)
            labels = [label_seq[:len(tag_seq)] for label_seq, tag_seq in zip(label_seqs, y_batch)]
            if extend_sequence:
                labels = [label_seq[1:-1] for label_seq in labels]
            for lab, lab_pred in zip(labels, labels_pred):
                accuracy += sum([a == b for (a, b) in zip(lab, lab_pred)])
            total_loss += loss
            counter += 1
            processed += len(y_batch)
        epoch += 1
        # print("Total training loss: %f, epoch: %d" % (total_loss, epoch))
        accuracy /= num_token
        lr *= decay
        # print("Training accuracy: %f, epoch: %d" % (accuracy, epoch))
        if validation_filename:
            if load_validation_data:
                R, P, F, R_all, P_all, F_all, acc = \
                    compute_metrics(sess,
                                    # pos_id_seq_map_val,
                                    wrd_id_seq_map_val,
                                    chr_id_mat_map_val,
                                    label_seq_map_val,
                                    tag_dict,
                                    batch_size,
                                    hidden_size,
                                    tags_map,
                                    tag_type_list,
                                    measure_all=True
                                    )
            else:
                R, P, F, R_all, P_all, F_all, acc = \
                    compute_metrics_memory_efficient(sess,
                                                     data_dev,
                                                     vocab,
                                                     tag_dict,
                                                     use_chars,
                                                     char_dict,
                                                     batch_size,
                                                     tags_map,
                                                     tag_type_list,
                                                     fout=None,
                                                     measure_all=True
                                                     )

            msg = "Epoch %d - training loss: %.4f acc: %.4f - val" % (epoch, total_loss, accuracy)
            print(msg, end='')

            # for tag_type in tag_type_list:
            #     print(" %s: r%.4f p%.4f f%.4f" % (tag_type, R[tag_type], P[tag_type], F[tag_type]), end='')
            print(" F1: %.2f - acc: %.2f" % (100 * F_all, 100 * acc), end='')
            logger.info(msg + " F1: %.2f - acc: %.2f" % (100 * F_all, 100 * acc))
            print()
            # F1_avg = sum(F.values()) / len(F)
            if F_all > F_all_best:
                epoch_best = epoch
                loss_best = total_loss
                accuracy_best = accuracy
                recall_best = R
                precision_best = P
                F1_best = F
                F_all_best = F_all
                acc_dev_best = acc
                now = time.time()
                delta = now - start
                print("Counter: %d, delta time = %f" % (counter, delta))
                logger.info("Counter: %d, delta time = %f" % (counter, delta))
                start = time.time()
                checkpoint_path = os.path.join(model_dir, checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=epoch)
        else:
            print("Epoch %d\t- training loss: %.4f\taccuracy: %.4f" % (epoch, total_loss, accuracy))
            if epoch % 5 == 0:
                now = time.time()
                delta = now - start
                print("Counter: %d, delta time = %f" % (counter, delta))
                start = time.time()
                checkpoint_path = os.path.join(model_dir, checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=epoch)

        if acc_stop and accuracy >= acc_stop:
            if not validation_filename:
                now = time.time()
                delta = now - start
                print("Counter: %d, delta time = %f" % (counter, delta))
                start = time.time()
                checkpoint_path = os.path.join(model_dir, checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=epoch)
            break
        elif 0 <= max_iter <= epoch:
            if not validation_filename:
                now = time.time()
                delta = now - start
                print("Counter: %d, delta time = %f" % (counter, delta))
                start = time.time()
                checkpoint_path = os.path.join(model_dir, checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=epoch)
            break
    elapsed_time = time.time() - begin
    if validation_filename:
        print("Best model - epoch %d - loss %.4f - acc %.4f - val" % (epoch_best, loss_best, accuracy_best), end='')
        logger.info("Best model - epoch %d - loss %.4f - acc %.4f - val" % (epoch_best, loss_best, accuracy_best))
        for tag_type in tag_type_list:
            print(" %s: r%.4f p%.4f f%.4f" % (
            tag_type, recall_best[tag_type], precision_best[tag_type], F1_best[tag_type]), end='')
            logger.info(" %s: r%.4f p%.4f f%.4f" % (
                tag_type, recall_best[tag_type], precision_best[tag_type], F1_best[tag_type]))
        print()
        print("F1: %.2f - acc: %.2f" % (100 * F_all_best, 100 * acc_dev_best))
        logger.info("F1: %.2f - acc: %.2f" % (100 * F_all_best, 100 * acc_dev_best))
        # Save validation metrics
        with open(os.path.join(model_dir, 'validation_metrics.txt'), 'w', encoding=encoding) as f:
            f.write("Best model - epoch %d - training loss %.4f - acc %.4f\n" % (epoch_best, loss_best, accuracy_best))
            f.write("Model name: %s\n" % os.path.basename(model_dir))
            f.write("F1: %.2f - acc: %.2f\n" % (100 * F_all_best, 100 * acc_dev_best))
            for tag_type in tag_type_list:
                f.write("%s: r%.4f p%.4f f%.4f\n" % (
                tag_type, recall_best[tag_type], precision_best[tag_type], F1_best[tag_type]))
            f.write("Elapsed time: %.4fs\n" % elapsed_time)
    print("Elapsed time: %.4fs\n" % elapsed_time)


def cross_validation(data_dir, training_filename, cv_metrics_filepath_prefix,
                     embedding_size, hidden_size, keep_prob, learning_rate, batch_size,
                     split_size, num_epochs_without_improvement, acc_stop, chopoff):
    global max_iter
    begin = time.time()
    train_path = os.path.join(data_dir, training_filename)
    print("Doing cross validation on %s" % train_path)

    # Build dictionary
    if chopoff > 1:
        # Build and chop rare terms off vocabulary
        tag_types = set()
        counts = compute_counts(data_dir, training_filename, tag_types)
        vocab = {UNK: 0, NUM: 1}
        for (token, count) in [pair for pair in counts if pair[1] >= chopoff]:
            vocab[token] = len(vocab)
        update_vocab = False
        tag_type_list = sorted(tag_types)
        tag_dict = build_tag_dictionary(tag_type_list)
    else:
        vocab = {UNK: 0, NUM: 1}
        update_vocab = True
        tag_type_list = []
        tag_dict = {}
        # TODO
        # If tag_dict is empty, both B and I would be empty, then parsing training data will be problematic.

    # pos_dict = build_pos_dictionary()

    tagged_queries = []
    cnt = 0
    with open(train_path, "r", encoding=encoding) as f:
        for tagged_query in f:
            tagged_query = tagged_query.strip()
            if not tagged_query:
                continue
            tagged_queries.append(tagged_query)
            cnt += 1
            if cnt % 1000 == 0:
                print("  read %d tagged queries" % cnt, end="\r")
    print("  read %d tagged queries" % cnt)

    seed = 1
    random.seed(seed)
    random.shuffle(tagged_queries)
    tagged_queries_cv = [[] for i in range(split_size)]
    for (i, tagged_query) in enumerate(tagged_queries):
        tagged_queries_cv[i % split_size].append(tagged_query)

    tag_id_seq_map_cv = [{} for i in range(split_size)]
    pos_id_seq_map_cv = [{} for i in range(split_size)]
    wrd_id_seq_map_cv = [{} for i in range(split_size)]
    label_seq_map_cv = [{} for i in range(split_size)]

    for i in range(split_size):
        cnt = 0
        for tagged_query in tagged_queries_cv[i]:
            prev_tag_id_seq, wrd_id_seq, label_seq, _ = build_sequence_from_a_tagged_query(
                tagged_query, tag_dict, vocab, update_vocab)
            T = len(label_seq)
            if T not in label_seq_map_cv[i]:
                label_seq_map_cv[i][T] = []
                tag_id_seq_map_cv[i][T] = []
                pos_id_seq_map_cv[i][T] = []
                wrd_id_seq_map_cv[i][T] = []
                # wrd_seq_map_cv[i][T] = []
            label_seq_map_cv[i][T].append(label_seq)
            tag_id_seq_map_cv[i][T].append(prev_tag_id_seq)
            # pos_id_seq_map_cv[i][T].append(pos_id_seq)
            wrd_id_seq_map_cv[i][T].append(wrd_id_seq)
            # wrd_seq_map_cv[i][T].append(wrd_seq)
            cnt += 1
            if cnt % 500 == 0:
                print("  formatted %d tagged queries in split %d" % (cnt, i), end="\r")
        print("  formatted %d tagged queries in split %d" % (cnt, i))
    # save_dictionary(model_dir, vocab)

    tags_map = [""] * len(tag_dict)
    for (tag, index) in tag_dict.items():
        tags_map[index] = tag

    # class_id_weight_dict = build_class_id_weight_dict(tag_dict, class_weights)

    # Do cross validation
    # i-th experiment:
    #   folds[i, ..., i + split_size - 3] for training
    #   folds[i + split_size - 2] for development or validation
    #   folds[i + split_size - 1] for test
    # instance_length = 1 + 2 * window_size + 1 + 2 * window_size + 1

    tag_embedding_classes = len(tag_dict)
    # pos_embedding_classes = len(pos_dict)
    wrd_embedding_classes = len(vocab)
    create_model(model_dir,
                 tag_embedding_classes,
                 # pos_embedding_classes,
                 wrd_embedding_classes,
                 embedding_size,
                 hidden_size,
                 learning_rate,
                 vocab,
                 )
    Rs = {tag_type: [] for tag_type in tag_types}
    Ps = {tag_type: [] for tag_type in tag_types}
    Fs = {tag_type: [] for tag_type in tag_types}
    for i in range(split_size):
        print(''.rjust(80, '-'))
        print(('-Experiment %d' % i).ljust(79, '-') + '-')
        print(''.rjust(80, '-'))
        print()
        sess = tf.Session()
        # Training stage for the i-th experiment
        print(''.rjust(80, '-'))
        print('-training'.ljust(79, '-') + '-')
        print(''.rjust(80, '-'))
        sess.run(tf.global_variables_initializer())
        counter = 0
        start = time.time()
        saver = tf.train.Saver(tf.global_variables())
        epoch = 0

        if max_iter < 0 and not acc_stop:
            train_fold_id_end = i + split_size - 2
        else:
            train_fold_id_end = i + split_size - 1
        tag_id_seq_map_train = {}
        for j in range(i, train_fold_id_end):
            for T, tag_id_seqs in tag_id_seq_map_cv[j - split_size].items():
                if T not in tag_id_seq_map_train:
                    tag_id_seq_map_train[T] = []
                tag_id_seq_map_train[T].extend(tag_id_seqs)
        # pos_id_seq_map_train = {}
        # for j in range(i, train_fold_id_end):
        #     for T, pos_id_seqs in pos_id_seq_map_cv[j - split_size].items():
        #         if T not in pos_id_seq_map_train:
        #             pos_id_seq_map_train[T] = []
        #         pos_id_seq_map_train[T].extend(pos_id_seqs)
        wrd_id_seq_map_train = {}
        for j in range(i, train_fold_id_end):
            for T, wrd_id_seqs in wrd_id_seq_map_cv[j - split_size].items():
                if T not in wrd_id_seq_map_train:
                    wrd_id_seq_map_train[T] = []
                wrd_id_seq_map_train[T].extend(wrd_id_seqs)
        label_seq_map_train = {}
        for j in range(i, train_fold_id_end):
            for T, label_seqs in label_seq_map_cv[j - split_size].items():
                if T not in label_seq_map_train:
                    label_seq_map_train[T] = []
                label_seq_map_train[T].extend(label_seqs)

        label_counts = [0. for i in range(len(tag_dict) - 1)]
        for T, label_seqs in label_seq_map_train.items():
            for label_seq in label_seqs:
                for label in label_seq:
                    label_counts[label] += 1.
        class_weights = {
        tags_map[tag_id]: label_counts[tag_dict[O]] / label_counts[tag_id] if label_counts[tag_id] > 0. else 0.
        for tag_id in range(len(tag_dict) - 1) if tag_id != tag_dict[N]}
        print('class_weights:', class_weights)
        class_id_weight_dict = build_class_id_weight_dict(tag_dict, class_weights)

        data_size = sum([len(label_seqs) for label_seqs in label_seq_map_train.values()])
        num_token = sum([len(label_seqs) * length for length, label_seqs in label_seq_map_train.items()])
        print("data_size: %s" % data_size)
        F1_best_avg = 0.0
        epoch_best = 0
        while True:

            accuracy = 0.0
            total_loss = 0.0
            processed = 0

            # Train the model for one pass of training data
            for T, label_seqs in label_seq_map_train.items():
                tag_id_seqs = tag_id_seq_map_train[T]
                # pos_id_seqs = pos_id_seq_map_train[T]
                wrd_id_seqs = wrd_id_seq_map_train[T]
                bucket_size = len(label_seqs)
                data_index = 0
                # Train the model in a bucket of training data
                while True:
                    if data_index >= bucket_size:
                        break
                    end_index = data_index + batch_size
                    if end_index > bucket_size:
                        end_index = bucket_size
                    print("  training [%d, %d] of %d" % (processed + data_index,
                                                         processed + end_index,
                                                         data_size), end="\r")
                    # Set the feed dictionary
                    feed_dict = {'input_tag_ids:0': tag_id_seqs[data_index:end_index],
                                 # 'input_pos_ids:0': pos_id_seqs[data_index:end_index],
                                 'input_wrd_ids:0': wrd_id_seqs[data_index:end_index],
                                 'labels:0': label_seqs[data_index:end_index],
                                 'weights:0': [[class_id_weight_dict[tag] for tag in label_seq]
                                               for label_seq in label_seqs[data_index:end_index]],
                                 'initial_hidden_bw:0': np.zeros([end_index - data_index, hidden_size], np.float32),
                                 'keep_prob:0': keep_prob}

                    loss, correct_hits, _ = \
                        sess.run(['cost:0',
                                  'correct_prediction:0',
                                  'updates'],
                                 feed_dict=feed_dict)
                    # if T == 4:
                    #     print(tag_id_seqs[data_index:end_index])
                    #     print(label_seqs[data_index:end_index])
                    #     print(output_tag_ids)
                    #     print()

                    # print(label_seqs[data_index:end_index])
                    # print(output_tag_ids)
                    # print()

                    accuracy += correct_hits
                    total_loss += loss
                    data_index += batch_size
                    counter += 1
                processed += bucket_size

            epoch += 1
            # print("Total training loss: %f, epoch: %d" % (total_loss, epoch))
            # accuracy /= data_size
            # print("Training accuracy: %f, epoch: %d" % (accuracy, epoch))
            acc = accuracy / num_token
            print("Epoch %d\t- training loss: %f\taccuracy: %f" % (epoch, total_loss, acc))
            sys.stdout.flush()
            if acc_stop:
                if acc >= acc_stop:
                    break
            elif max_iter < 0:
                # Now we have finished one epoch, we need to compute metrics on development or validation set.
                dev_fold_id = i - 2
                # pos_id_seq_map_val = pos_id_seq_map_cv[dev_fold_id]
                wrd_id_seq_map_val = wrd_id_seq_map_cv[dev_fold_id]
                label_seq_map_val = label_seq_map_cv[dev_fold_id]
                R, P, F = compute_metrics(sess,
                                          # pos_id_seq_map_val,
                                          wrd_id_seq_map_val,
                                          label_seq_map_val,
                                          tag_dict,
                                          batch_size,
                                          hidden_size,
                                          tags_map,
                                          tag_type_list
                                          )
                print("Validation:", end='')
                for tag_type in tag_type_list:
                    print(" %s: r%.4f p%.4f f%.4f" % (tag_type, R[tag_type], P[tag_type], F[tag_type]), end='')
                print()
                print(''.rjust(80, '-'))
                F1_avg = sum(F.values()) / len(F)
                # The metrics on development set has not been improved for num_steps_without_improvement steps,
                # we thus reach to the cusp of fitting stage and finished training
                if F1_avg > F1_best_avg:
                    F1_best_avg = F1_avg
                    epoch_best = epoch
                    does_stop = False
                elif epoch - epoch_best >= num_epochs_without_improvement:
                    does_stop = True
                else:
                    does_stop = False
                if does_stop:
                    break
            else:
                if max_iter <= epoch:
                    break

        # Test stage for the i-th experiment
        print()
        print(''.rjust(80, '-'))
        print('-test'.ljust(79, '-') + '-')
        print(''.rjust(80, '-'))

        test_fold_id = i - 1
        # pos_id_seq_map_test = pos_id_seq_map_cv[test_fold_id]
        wrd_id_seq_map_test = wrd_id_seq_map_cv[test_fold_id]
        label_seq_map_test = label_seq_map_cv[test_fold_id]
        R, P, F = compute_metrics(sess,
                                  # pos_id_seq_map_test,
                                  wrd_id_seq_map_test,
                                  label_seq_map_test,
                                  tag_dict,
                                  batch_size,
                                  hidden_size,
                                  tags_map,
                                  tag_type_list
                                  )
        # print(''.rjust(80, '-'))
        for tag_type in tag_type_list:
            print(('-%s' % tag_type).ljust(80, '-'))
            print("-Recall Precision F1".ljust(80, '-'))
            print(("-%.4f %.4f %.4f" % (R[tag_type], P[tag_type], F[tag_type])).ljust(80, '-'))
            print(''.rjust(80, '-'))
        print()
        for tag_type in tag_types:
            Rs[tag_type].append(R[tag_type])
            Ps[tag_type].append(P[tag_type])
            Fs[tag_type].append(F[tag_type])
        sess.close()

    # Output the final metrics for cross validation on the given training data
    print(''.rjust(80, '-'))
    print('-Final Cross Validation Metrics'.ljust(80, '-'))
    for tag_type in tag_type_list:
        print(('-%s' % tag_type).ljust(80, '-'))
        print("-Recall Precision F1".ljust(80, '-'))
        print(("-%.4f %.4f %.4f" % (sum(Rs[tag_type]) / split_size,
                                    sum(Ps[tag_type]) / split_size,
                                    sum(Fs[tag_type]) / split_size)).ljust(80, '-'))
    print(''.rjust(80, '-'))

    elapsed_time = time.time() - begin
    metric_string = ''
    for tag_type in tag_type_list:
        metric_string += "-%s-r%.4f-p%.4f-f%.4f" % \
                         (tag_type,
                          sum(Rs[tag_type]) / split_size,
                          sum(Ps[tag_type]) / split_size,
                          sum(Fs[tag_type]) / split_size)
    metric_string += "-t%.4fs.txt" % elapsed_time
    cv_metrics_filepath = cv_metrics_filepath_prefix + metric_string
    f = open(cv_metrics_filepath, 'w', encoding=encoding)
    f.close()
    print("%d fold cross validation metrics are saved at %s\n" % (split_size, cv_metrics_filepath))
    print("Elapsed time: %.4fs\n" % elapsed_time)
    return


################################################################################
# Prediction                                                                   #
################################################################################
def predict(model_dir, hidden_size, cell_type, test_filepath):
    sess = restore_session(model_dir)

    tag_type_list = load_tag_type_list(model_dir)
    tag_dict = build_tag_dictionary(tag_type_list)
    # pos_dict = build_pos_dictionary()
    vocab = load_dictionary(model_dir)

    tags_map = [""] * len(tag_dict)
    for (tag, index) in tag_dict.items():
        tags_map[index] = tag

    exit_command_set = {":q", ":quit", "quit()", "exit()", '#exit#'}

    f = None
    if test_filepath:
        f = open(test_filepath, 'r', encoding=encoding)
    else:
        f = sys.stdin

    for query in f:
        query = query.strip()
        if not query:
            continue

        if query in exit_command_set:
            break

        wrd_id_seq, wrd_seq = build_sequence_from_a_query(query, vocab)
        length = len(wrd_id_seq)
        # pos_id_seq_batch = np.array([pos_id_seq])
        wrd_id_seq_batch = np.array([wrd_id_seq])
        tag_ids_pred_batch = np.zeros([1, length], np.int32)

        # Compute backward RNN outputs
        zero_matrix = np.zeros([1, hidden_size])
        feed_dict = {  # 'input_pos_ids:0': pos_id_seq_batch,
            'input_wrd_ids:0': wrd_id_seq_batch,
            'initial_hidden_bw:0': zero_matrix}
        hidden_bw_reverse = sess.run('hidden_bw_reverse:0', feed_dict=feed_dict)

        # Predict tags token by token using concatenated forward and backward hidden vectors
        if cell_type == "GRU":
            state_batch = zero_matrix
        elif cell_type == "LSTM":
            state_batch = [zero_matrix, zero_matrix]
        else:
            raise CustomException("Cell type %s is not supported." % cell_type)
        tag_id_batch = tag_dict[N] * np.ones(1)
        for t in range(length):
            if cell_type == "GRU":
                feed_dict = {'input_tag_id:0': tag_id_batch,
                             # 'input_pos_id:0': pos_id_seq_batch[:, t],
                             'input_wrd_id:0': wrd_id_seq_batch[:, t],
                             'input_state_fw:0': state_batch,
                             'input_hidden_bw:0': hidden_bw_reverse[:, t, :]}
                state_batch, logits_batch = sess.run(['Forward_1/GRUCell/add:0', 'logits_pred:0'], feed_dict=feed_dict)
            elif cell_type == "LSTM":
                feed_dict = {'input_tag_id:0': tag_id_batch,
                             # 'input_pos_id:0': pos_id_seq_batch[:, t],
                             'input_wrd_id:0': wrd_id_seq_batch[:, t],
                             'input_state_fw_c:0': state_batch[0],
                             'input_state_fw_h:0': state_batch[1],
                             'input_hidden_bw:0': hidden_bw_reverse[:, t, :]}
                # TypeError: 'tuple' object does not support item assignment
                state_batch[0], state_batch[1], logits_batch = sess.run(['Forward_1/LSTMCell/add_1:0',
                                                                         'Forward_1/LSTMCell/mul_2:0',
                                                                         'logits_pred:0'],
                                                                        feed_dict=feed_dict)
            else:
                raise CustomException("Cell type %s is not supported." % cell_type)
            tag_id_pred_batch = np.argmax(logits_batch, 1)
            tag_id_batch = tag_id_pred_batch
            tag_ids_pred_batch[:, t] = tag_id_batch

        tag_ids_pred = tag_ids_pred_batch[0, :]
        tags_pred = [tags_map[tag_id_pred] for tag_id_pred in tag_ids_pred]

        # Decode tags_pred using wrd_seq to get the target concepts in the query
        concepts = {tag_type: [] for tag_type in tag_type_list}
        tag_type_prev = ''
        concept = ''
        for i, tag_curr in enumerate(tags_pred):
            word = wrd_seq[i]
            if tag_curr.startswith('B'):
                tag_type_curr = tag_curr[2:]
                if concept:
                    concepts[tag_type_prev].append(concept)
                concept = word
            elif tag_curr.startswith('I'):
                tag_type_curr = tag_curr[2:]
                if tag_type_prev != tag_type_curr:
                    if concept:
                        concepts[tag_type_prev].append(concept)
                    concept = word
                else:
                    concept += ' ' + word
            else:
                tag_type_curr = ''
                if concept:
                    concepts[tag_type_prev].append(concept)
                concept = ''
            tag_type_prev = tag_type_curr
            # tag_prev = tag_curr
        if concept:
            concepts[tag_type_prev].append(concept)

        sys.stdout.write(query)
        for tag_type in tag_type_list:
            sys.stdout.write('\t')
            sys.stdout.write(','.join(concepts[tag_type]))
        sys.stdout.write('\n')

    f.close()
    sess.close()


def restore_session(model_dir):
    filepath = os.path.join(model_dir, 'tf_version.txt')
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding=encoding) as f:
            tf_version = f.readline().strip()
            if tf_version != tf.__version__:
                raise Exception('model was trained on TF %s, but the TF version on current machine is %s' %
                                (tf_version, tf.__version__))
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(model_dir)
    model_checkpoint_path = ckpt.model_checkpoint_path
    if not os.path.exists(model_checkpoint_path):
        model_checkpoint_path = os.path.join(model_dir, os.path.basename(model_checkpoint_path))
    saver = tf.train.import_meta_graph(model_checkpoint_path + '.meta')
    saver.restore(sess, model_checkpoint_path)
    return sess


def predict_interactive(model_dir, hidden_size, cell_type):
    # global max_word_length
    sess = restore_session(model_dir)

    tag_type_list = load_tag_type_list(model_dir)
    tag_dict = build_tag_dictionary(tag_type_list)
    # pos_dict = build_pos_dictionary()
    vocab = load_dictionary(model_dir)
    if use_chars:
        char_dict = build_char_dictionary()
    # load_max_word_length()

    tags_map = [""] * len(tag_dict)
    for (tag, index) in tag_dict.items():
        tags_map[index] = tag

    exit_command_set = {":q", ":quit", "quit()", "exit()", '#exit#'}

    sys.stdout.write("> ")
    sys.stdout.flush()
    query = sys.stdin.readline()
    while query:
        query = query.strip()
        if not query:
            continue

        if query in exit_command_set:
            break
        begin = time.time()
        wrd_id_seq, wrd_seq = build_sequence_from_a_query(query, vocab)
        # max_word_length = max([len(w) for w in wrd_seq])
        max_word_length = max(map(lambda w: len(w), wrd_seq))
        if use_chars:
            chr_id_mat = build_sequence_from_a_word_sequence_in_char_level(wrd_seq, char_dict, max_word_length)

        feed_dict = {  # 'input_tag_ids:0': tag_id_seqs[data_index:end_index],
            # 'input_pos_ids:0': pos_id_seqs[data_index:end_index],
            'input_wrd_ids:0': [wrd_id_seq],
            'input_chr_ids:0': [chr_id_mat],
            # 'labels:0': label_seqs[data_index:end_index],
            # 'weights:0': [[class_id_weight_dict[tag] for tag in label_seq]
            #               for label_seq in label_seqs[data_index:end_index]],
            # 'initial_hidden_bw:0': np.zeros([1, hidden_size]),
            'keep_prob:0': 1.0}

        # Predict the tag sequence
        logits, trans_params = sess.run(['logits:0', tf.get_collection('CRF')[0]], feed_dict=feed_dict)
        # iterate over the sentences because no batching in vitervi_decode
        labels_pred = []
        # scores_list = []
        for logit in logits:
            # logit = logit[:sequence_length] # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            # viterbi_logit = [logit[t, viterbi_seq[t]] for t in range(len(wrd_id_seq))]
            if extend_sequence:
                viterbi_seq = viterbi_seq[1:-1]
                # viterbi_logit = viterbi_logit[1:-1]
                # viterbi_score = viterbi_score[1:-1]
            labels_pred.append(viterbi_seq)
            # logits_list.append(viterbi_logit)
            # scores_list.append(viterbi_score)

        tag_ids_pred = labels_pred[0]
        tags_pred = [tags_map[tag_id_pred] for tag_id_pred in tag_ids_pred]

        # Decode tags_pred using wrd_seq to get the target concepts in the query
        concepts = {tag_type: [] for tag_type in tag_type_list}
        llhs = {tag_type: [] for tag_type in tag_type_list}
        tag_type_prev = ''
        concept = ''
        llh = 0.0
        phrase_size = 0
        for i, (tag_curr, logits) in enumerate(zip(tags_pred, logits[0])):
            word = wrd_seq[i]
            if tag_curr != O:
                tag_id = tag_dict[tag_curr]
                probs = softmax(logits)
                llh_curr = math.log(probs[0, tag_id])
            if tag_curr.startswith('B'):
                tag_type_curr = tag_curr[2:]
                if concept:
                    concepts[tag_type_prev].append(concept)
                    llhs[tag_type_prev].append(llh / phrase_size)
                concept = word
                llh = llh_curr
                phrase_size = 1
            elif tag_curr.startswith('I'):
                tag_type_curr = tag_curr[2:]
                if tag_type_prev != tag_type_curr:
                    if concept:
                        concepts[tag_type_prev].append(concept)
                        llhs[tag_type_prev].append(llh / phrase_size)
                    concept = word
                    llh = llh_curr
                    phrase_size = 1
                else:
                    concept += ' ' + word
                    llh += llh_curr
                    phrase_size += 1
            else:
                tag_type_curr = ''
                if concept:
                    concepts[tag_type_prev].append(concept)
                    llhs[tag_type_prev].append(llh / phrase_size)
                concept = ''
                llh = 0.0
                phrase_size = 0
            tag_type_prev = tag_type_curr
        if concept:
            concepts[tag_type_prev].append(concept)
            llhs[tag_type_prev].append(llh / phrase_size)
        elapsed_time = time.time() - begin
        for tag_type in tag_type_list:
            print('- %s:' % tag_type,
                  ','.join(concepts[tag_type]),
                  'loglikelihood:',
                  ','.join(['%.4f' % llh for llh in llhs[tag_type]]))

        print(": ", end="")
        print(tags_pred)
        print(": ", end="")
        print(wrd_seq)
        print(": Elapsed time: %.4fs\n" % elapsed_time)
        sys.stdout.flush()
        sys.stdout.write("> ")
        sys.stdout.flush()
        query = sys.stdin.readline()

    sess.close()


################################################################################
# Evaluation                                                                   #
################################################################################
def evaluate(model_dir, hidden_size, cell_type, eval_filepath, output_filepath):
    begin = time.time()
    if eval_filepath:
        print("Doing evaluation on %s" % eval_filepath)
    else:
        print("Doing evaluation from standard input")
    print('model directory:', model_dir)

    sess = restore_session(model_dir)

    tag_type_list = load_tag_type_list(model_dir)
    tag_dict = build_tag_dictionary(tag_type_list)
    # pos_dict = build_pos_dictionary()
    vocab = load_dictionary(model_dir)
    if use_chars:
        char_dict = build_char_dictionary()
    load_max_word_length()

    tags_map = [""] * len(tag_dict)
    for (tag, index) in tag_dict.items():
        tags_map[index] = tag

    if eval_filepath:
        f = open(eval_filepath, 'r', encoding=encoding)
    else:
        f = sys.stdin

    if output_filepath:
        parent_dir = os.path.dirname(output_filepath)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        fout = open(output_filepath, 'w')
    else:
        fout = None

    yield_eval_data = not True
    if yield_eval_data:
        data_test = TextDataset(None, annotation_scheme, extend_sequence, file=f)
        R, P, F, R_all, P_all, F_all, acc = \
            compute_metrics_memory_efficient(sess,
                                             data_test,
                                             vocab,
                                             tag_dict,
                                             use_chars,
                                             char_dict,
                                             batch_size,
                                             tags_map,
                                             tag_type_list,
                                             fout=None,
                                             measure_all=True
                                             )
    else:
        tag_id_seq_map_val = {}
        # pos_id_seq_map_val = {}
        wrd_id_seq_map_val = {}
        wrd_seq_map_val = {}
        chr_id_mat_map_val = {}
        label_seq_map_val = {}
        cnt = 0
        if annotation_scheme == 'CoNLL':
            split_pattern = re.compile(r'[ \t]')
            wrd_seq, tag_seq = [], []
            if extend_sequence:
                wrd_seq += [STT]
                tag_seq += [O]
            for line in f:
                line = line.strip()
                if not line or line.startswith('-DOCSTART-'):
                    if extend_sequence and len(wrd_seq) > 1 or not extend_sequence and wrd_seq:
                        if extend_sequence:
                            wrd_seq += [END]
                            tag_seq += [O]
                        wrd_id_seq = [vocab[w if w in vocab else NUM if is_number(w) else UNK] for w in wrd_seq]
                        label_seq = [tag_dict[tag] for tag in tag_seq]
                        if use_chars:
                            chr_id_mat = build_sequence_from_a_word_sequence_in_char_level(wrd_seq, char_dict,
                                                                                           max_word_length)
                        T = len(wrd_seq)
                        if T not in label_seq_map_val:
                            label_seq_map_val[T] = []
                            # tag_id_seq_map[T] = []
                            # pos_id_seq_map[T] = []
                            wrd_id_seq_map_val[T] = []
                            if use_chars:
                                chr_id_mat_map_val[T] = []
                        label_seq_map_val[T].append(label_seq)
                        # tag_id_seq_map[T].append(prev_tag_id_seq)
                        # pos_id_seq_map[T].append(pos_id_seq)
                        wrd_id_seq_map_val[T].append(wrd_id_seq)
                        if use_chars:
                            chr_id_mat_map_val[T].append(chr_id_mat)
                        cnt += 1
                        if cnt % 500 == 0:
                            print("  formatted %d tagged examples" % cnt, end="\r")
                        wrd_seq, tag_seq = [], []
                        if extend_sequence:
                            wrd_seq += [STT]
                            tag_seq += [O]
                    continue
                # container = line.split(' \t')
                container = split_pattern.split(line)
                token, tag = container[0], container[-1]
                token = token.lower()
                tag = tag.upper()
                wrd_seq.append(token)
                tag_seq.append(tag)
            print("  formatted %d tagged examples" % cnt)
        else:
            # open_tag = '{' if model_type == ModelType.ANSWER else '['
            # close_tag = '}' if model_type == ModelType.ANSWER else ']'

            for tagged_query in f:
                tagged_query = tagged_query.strip()
                if not tagged_query:
                    continue
                prev_tag_id_seq, wrd_id_seq, label_seq, wrd_seq = build_sequence_from_a_tagged_query(
                    tagged_query, tag_dict, vocab, update_vocab=False)
                if use_chars:
                    chr_id_mat = build_sequence_from_a_word_sequence_in_char_level(wrd_seq, char_dict, max_word_length)
                T = len(label_seq)
                if T not in label_seq_map_val:
                    label_seq_map_val[T] = []
                    # tag_id_seq_map_val[T] = []
                    # pos_id_seq_map_val[T] = []
                    wrd_id_seq_map_val[T] = []
                    wrd_seq_map_val[T] = []
                    if use_chars:
                        chr_id_mat_map_val[T] = []
                label_seq_map_val[T].append(label_seq)
                # tag_id_seq_map_val[T].append(prev_tag_id_seq)
                # pos_id_seq_map_val[T].append(pos_id_seq)
                wrd_id_seq_map_val[T].append(wrd_id_seq)
                wrd_seq_map_val[T].append(wrd_seq)
                if use_chars:
                    chr_id_mat_map_val[T].append(chr_id_mat)
                cnt += 1
                if cnt % 500 == 0:
                    print("  formatted %d tagged examples" % cnt, end="\r")
            print("  formatted %d tagged examples" % cnt)
        f.close()

        R, P, F, R_all, P_all, F_all, acc = \
            compute_metrics(sess,
                            # pos_id_seq_map_test,
                            wrd_id_seq_map_val,
                            chr_id_mat_map_val,
                            label_seq_map_val,
                            tag_dict,
                            batch_size,
                            hidden_size,
                            tags_map,
                            tag_type_list,
                            fout,
                            wrd_seq_map_val,
                            measure_all=True
                            )

    for tag_type in tag_type_list:
        print(''.rjust(80, '-'))
        print(('-%s' % tag_type).ljust(80, '-'))
        print("-Recall Precision F1".ljust(80, '-'))
        print(("-%.4f %.4f %.4f" % (R[tag_type], P[tag_type], F[tag_type])).ljust(80, '-'))
    print(''.rjust(80, '-'))
    print(('-%s' % 'All').ljust(80, '-'))
    print("-Recall Precision F1 Accuracy".ljust(80, '-'))
    print(("-%.4f %.4f %.4f %.4f" % (R_all, P_all, F_all, acc)).ljust(80, '-'))
    print(''.rjust(80, '-'))
    print()
    elapsed_time = time.time() - begin
    print("Elapsed time: %.4fs\n" % elapsed_time)
    if fout:
        fout.close()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A neural network model to recognize answer and context concepts for a query. "
                    "The type of recognized concepts depends on the specified model type.")
    parser.add_argument("-d", '--data_dir',
                        type=str,
                        help='training data directory path',
                        default='.'
                        )
    parser.add_argument("-t", '--training_filename',
                        type=str,
                        help='filename for the training data',
                        default='ManualLabeledConcepts.txt'
                        )
    parser.add_argument("-v", '--validation_filename',
                        type=str,
                        help='filename for the validation or development data',
                        default=''
                        )
    parser.add_argument("-T", '--task',
                        type=str,
                        help='task',
                        choices=['train', 'online', 'cv', 'predict', 'eval'],
                        default='train'
                        )
    parser.add_argument("-i", '--maxiter',
                        type=int,
                        help='maximal number of iterations',
                        default=-1
                        )
    # parser.add_argument("-m", '--model_dir_name',
    #                     type=str,
    #                     help='model directory name',
    #                     default='model'
    #                     )
    parser.add_argument("-l", '--learning_rate',
                        type=float,
                        help='learning rate',
                        default=0.001
                        )
    parser.add_argument("-e", '--embedding_size',
                        type=int,
                        help='embedding size',
                        default=64
                        )
    parser.add_argument("-H", '--hidden_size',
                        type=int,
                        help='hidden size',
                        default=64
                        )
    parser.add_argument('--hidden_size_char',
                        type=int,
                        help='hidden size of LSTM on characters',
                        default=100
                        )
    parser.add_argument("-b", '--batch_size',
                        type=int,
                        help='mini-batch size',
                        default=10
                        )
    parser.add_argument("-w", '--window_size',
                        type=int,
                        help='window size',
                        default=4
                        )
    # parser.add_argument("-M", '--model_type',
    #                     type=str,
    #                     help='model type',
    #                     choices=['answer', 'context'],
    #                     default="answer"
    #                     )
    parser.add_argument("-s", '--split_size',
                        type=int,
                        help='split size',
                        default=5
                        )
    parser.add_argument("-a", '--acc_stop',
                        type=float,
                        help='accuracy to stop training',
                        default=None
                        )
    parser.add_argument('-f', '--fix_word_embedding',
                        dest='fix_word_embedding',
                        action='store_true',
                        help='if word embedding is fixed',
                        default=False
                        )
    parser.add_argument("-p", '--pretrained_embedding_path',
                        type=str,
                        help='pretrained embedding path',
                        default=''
                        )
    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true',
                        help='if more information is going to be displayed',
                        default=False
                        )
    parser.add_argument('--resume',
                        dest='resume',
                        action='store_true',
                        help='if model training is resumed on top of last checkpoint',
                        default=False
                        )
    parser.add_argument('--use_senna',
                        dest='use_senna',
                        action='store_true',
                        help='if SENNA is used for POS tagging',
                        default=False
                        )
    parser.add_argument("-n", '--num_epochs_without_improvement',
                        type=int,
                        help='number of epochs without improvement',
                        default=25
                        )
    parser.add_argument("-k", '--keep_prob',
                        type=float,
                        help='number of epochs without improvement',
                        default=0.5
                        )
    parser.add_argument('-m', '--model_dir',
                        type=str,
                        help='model directory path',
                        default=''
                        )
    parser.add_argument('--test_filepath',
                        type=str,
                        help='test filepath',
                        default=''
                        )
    parser.add_argument('--eval_filepath',
                        type=str,
                        help='path to the evaluation file',
                        default=''
                        )
    parser.add_argument('--output_filepath',
                        type=str,
                        help='path to the evaluation file',
                        default=''
                        )
    parser.add_argument('--cv_dir',
                        type=str,
                        help='directory path to save cross validation results',
                        default=''
                        )
    parser.add_argument("-c", '--chopoff',
                        type=int,
                        help='minimal frequency a vocabulary term should have',
                        default=1
                        )
    parser.add_argument('-C', '--cell_type',
                        type=str,
                        help='Cell type for RNN',
                        choices=['GRU', 'LSTM'],
                        default='LSTM'
                        )
    parser.add_argument('-A', '--annotation_scheme',
                        type=str,
                        help='annotation scheme for tagged training data. \'<>\' for XML tags or \'[]\' '
                             'for named bracket tags',
                        choices=['<>', '[]', 'CoNLL'],
                        default='[]'
                        )
    parser.add_argument("-E", '--embedding_size_char',
                        type=int,
                        help='embedding size for characters',
                        default=100
                        )
    parser.add_argument('--use_chars',
                        dest='use_chars',
                        action='store_true',
                        help='if character embedding is used',
                        default=False
                        )
    parser.add_argument('--decay',
                        type=float,
                        help='decay for learning rate',
                        default=0.9
                        )
    # parser.add_argument('--shift_backward',
    #                     dest='shift_backward',
    #                     action='store_true',
    #                     help='if the last state vectors are not used for backward state sequence',
    #                     default=False
    #                     )
    parser.add_argument('--exhaust_backward',
                        dest='exhaust_backward',
                        action='store_true',
                        help='if all state vectors are used for backward state sequence',
                        default=False
                        )
    # parser.add_argument('--use_all_chars',
    #                     dest='use_all_chars',
    #                     action='store_true',
    #                     help='if all characters are used for character level embedding, i.e., no clipping',
    #                     default=False
    #                     )
    parser.add_argument('--max_gradient_norm',
                        type=float,
                        help='max gradient norm for clipping, if negative no clipping',
                        default=-1
                        )
    parser.add_argument('--extend_sequence',
                        dest='extend_sequence',
                        action='store_true',
                        help='if an input sequence is extended by extra tokens, e.g., "^start" or "end$"',
                        default=False
                        )
    parser.add_argument('--yield_data',
                        dest='yield_data',
                        action='store_true',
                        help='if fetch a mini-batch raw data each time from a text file to save memory',
                        default=False
                        )
    parser.add_argument('-B', '--brnn_type',
                        type=str,
                        help='BRNN architecture type',
                        choices=['vanilla', 'backward_shift', 'residual'],
                        default='vanilla'
                        )
    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        help='if in debug mode',
                        default=False
                        )
    args = parser.parse_args()

    data_dir = args.data_dir
    training_filename = args.training_filename
    task = args.task
    max_iter = args.maxiter

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    window_size = args.window_size
    # model_type = ModelType.ANSWER if args.model_type == "answer" else ModelType.CONTEXT
    # By default train_word_embedding = True
    fix_word_embedding = args.fix_word_embedding
    train_word_embedding = not args.fix_word_embedding
    pretrained_embedding_path = args.pretrained_embedding_path
    verbose = args.verbose
    validation_filename = args.validation_filename
    keep_prob = args.keep_prob
    chopoff = args.chopoff
    cell_type = args.cell_type
    annotation_scheme = args.annotation_scheme
    if annotation_scheme == '<>':
        extract_word = extract_word_xml
        parse_a_tagged_query = parse_a_tagged_query_xml
    elif annotation_scheme == '[]':
        extract_word = extract_word_bracket
        parse_a_tagged_query = parse_a_tagged_query_bracket
    embedding_size_char = args.embedding_size_char
    hidden_size_char = args.hidden_size_char
    use_chars = args.use_chars
    decay = args.decay
    exhaust_backward = args.exhaust_backward
    shift_backward = not exhaust_backward
    # use_all_chars = args.use_all_chars
    max_gradient_norm = args.max_gradient_norm
    extend_sequence = args.extend_sequence
    yield_data = args.yield_data
    brnn_type = args.brnn_type

    debug = args.debug
    if debug:
        data_dir = r"C:\Users\miqian\Data\conll2003\CoNLL2003-BIO"
        training_filename = 'train.ner.bio'
        validation_filename = 'testa.ner.bio'
        task = 'train'
        maxiter = 30
        learning_rate = 0.001
        embedding_size = 300
        hidden_size = 300
        embedding_size_char = 100
        hidden_size_char = 100
        batch_size = 20
        fix_word_embedding = True
        pretrained_embedding_path = r"C:\Users\miqian\Software\sequence_tagging\data\glove.6B\glove.6B.300d.txt"
        chopoff = 1
        annotation_scheme = 'CoNLL'
        use_chars = True
        decay = 0.9
        max_gradient_norm = -1
        extend_sequence = False
        yield_data = True
        brnn_type = 'vanilla'

    # By default initialized_by_pretrained_embedding = False
    initialized_by_pretrained_embedding = pretrained_embedding_path != ''
    zero_padded = yield_data

    # model_dir_name = args.model_dir_name
    model_dir = args.model_dir
    model_dir_name = os.path.basename(model_dir) if model_dir != '' else 'model-BRNNCRF-%s' % cell_type

    if args.use_senna:
        senna_path = r"C:\Users\miqian\Software\senna-v3.0"
        pos_tagger = SennaTagger(senna_path)
        print("SENNA is used for POS tagging.")

    # if debug:
        # model_dir_name = 'model-answer-cv-w4-e64-H64-l0.001-b10'

    if task == "train":
        resume = args.resume
        acc_stop = args.acc_stop
        if model_dir == '':

            if yield_data:
                model_dir_name += "-x"

            if extend_sequence:
                model_dir_name += "-^"

            # if not shift_backward:
            #     model_dir_name += "-$"
            if brnn_type == 'vanilla':
                model_dir_name += "-B1"
            elif brnn_type == 'backward_shift':
                model_dir_name += "-B2"
            elif brnn_type == 'residual':
                model_dir_name += "-B3"

            # if use_all_chars:
            #     model_dir_name += "-A"

            if max_gradient_norm > 0:
                model_dir_name += "-m%g" % max_gradient_norm

            if use_chars:
                model_dir_name += "-E%d-h%d" % (embedding_size_char, hidden_size_char)
            model_dir_name += "-e%d-H%d-k%s-l%s-b%d" % \
                              (embedding_size, hidden_size, keep_prob, learning_rate, batch_size)
            if acc_stop:
                model_dir_name += "-a%g" % acc_stop
            elif max_iter > 0:
                model_dir_name += "-i%d" % max_iter

            if chopoff > 1:
                model_dir_name += "-c%d" % chopoff

            if initialized_by_pretrained_embedding:
                model_dir_name += "-%s" % os.path.basename(pretrained_embedding_path)

            if fix_word_embedding:
                model_dir_name += "-fixed"

            model_dir_name += "-%s" % os.path.splitext(os.path.basename(training_filename))[0]

            if validation_filename:
                model_dir_name += "-%s" % os.path.splitext(os.path.basename(validation_filename))[0]

            model_dir = os.path.join(data_dir, model_dir_name)
        # if not os.path.exists(model_dir):
        #     os.makedirs(model_dir)
        parameters = {"l": learning_rate,
                      "e": embedding_size,
                      "H": hidden_size,
                      "k": keep_prob,
                      # "M": model_type,
                      "i": max_iter,
                      "a": acc_stop,
                      "c": chopoff,
                      "C": cell_type,
                      "b": batch_size}
        for (key, value) in parameters.items():
            print("Parameter %s: %s" % (key, value))
        print("Python script name: %s" % os.path.splitext(os.path.basename(__file__))[0])
        if yield_data:
            train_memory_efficient()
        else:
            train(data_dir, training_filename, validation_filename, model_dir,
                  embedding_size, hidden_size, keep_prob, learning_rate, batch_size, acc_stop,
                  chopoff, resume)
        print("Models are saved in %s" % os.path.abspath(model_dir))
    elif task == "predict":
        match = re.compile(r'-H([\d]+)-').findall(model_dir_name)
        if match:
            hidden_size = int(match[0])
        match = re.compile(r'model-([\w]+)-([\w]+)-').findall(model_dir_name)
        if match:
            cell_type = match[0][1]
        test_filepath = args.test_filepath
        predict(model_dir, hidden_size, cell_type, test_filepath)
    elif task == "eval":
        match = re.compile(r'-H([\d]+)-').findall(model_dir_name)
        if match:
            hidden_size = int(match[0])
        match = re.compile(r'model-([\w]+)-([\w]+)-').findall(model_dir_name)
        if match:
            cell_type = match[0][1]
        if model_dir_name.find('-^-') != -1:
            extend_sequence = True
        # if model_dir_name.find('-A-') != -1:
        #     use_all_chars = True
        match = re.compile(r'-E([\d]+)-').findall(model_dir_name)
        if match:
            use_chars = True
        eval_filepath = args.eval_filepath
        output_filepath = args.output_filepath
        evaluate(model_dir, hidden_size, cell_type, eval_filepath, output_filepath)
    elif task == "cv":
        cv_dir = args.cv_dir
        if cv_dir == '':
            cv_dir = os.path.join(data_dir, 'cv')
        if not os.path.exists(cv_dir):
            os.makedirs(cv_dir)
        model_dir = os.path.join(cv_dir, model_dir_name.replace('model-', ''))
        split_size = args.split_size
        num_epochs_without_improvement = args.num_epochs_without_improvement
        acc_stop = args.acc_stop
        # if not os.path.exists(model_dir):
        #     os.makedirs(model_dir)
        # for i in range(split_size):
        #     cv_model_dir = os.path.join(model_dir, "fold-%d" % i)
        #     if not os.path.exists(cv_model_dir):
        #         os.makedirs(cv_model_dir)
        # print("Models will be saved in %s" % os.path.abspath(model_dir))
        parameters = {"l": learning_rate,
                      "e": embedding_size,
                      "H": hidden_size,
                      "k": keep_prob,
                      # "M": model_type,
                      "s": split_size,
                      "i": max_iter,
                      "n": num_epochs_without_improvement,
                      "a": acc_stop,
                      "c": chopoff,
                      "C": cell_type,
                      "b": batch_size}
        for (key, value) in parameters.items():
            print("Parameter %s: %s" % (key, value))
        print("Python script name: %s" % os.path.splitext(os.path.basename(__file__))[0])
        cv_metrics_filepath_prefix = model_dir + "-cv%d-e%d-H%d-k%s-l%s-b%d" % \
                                     (split_size, embedding_size,
                                      hidden_size, keep_prob, learning_rate, batch_size)
        if acc_stop:
            cv_metrics_filepath_prefix += "-a%g" % acc_stop
        elif max_iter < 0:
            cv_metrics_filepath_prefix += "-n%d" % num_epochs_without_improvement
        else:
            cv_metrics_filepath_prefix += "-i%d" % max_iter

        if chopoff > 1:
            cv_metrics_filepath_prefix += "-c%d" % chopoff

        if args.use_senna:
            cv_metrics_filepath_prefix += "-senna"

        if initialized_by_pretrained_embedding:
            cv_metrics_filepath_prefix += "-%s" % os.path.basename(pretrained_embedding_path)

        if fix_word_embedding:
            cv_metrics_filepath_prefix += "-fixed"

        cv_metrics_filepath_prefix += "-%s" % os.path.splitext(os.path.basename(training_filename))[0]

        cross_validation(data_dir, training_filename, cv_metrics_filepath_prefix,
                         embedding_size, hidden_size, keep_prob, learning_rate, batch_size,
                         split_size, num_epochs_without_improvement, acc_stop, chopoff)
    elif task == "online":
        match = re.compile(r'-H([\d]+)-').findall(model_dir_name)
        if match:
            hidden_size = int(match[0])
        match = re.compile(r'model-([\w]+)-([\w]+)-').findall(model_dir_name)
        if match:
            cell_type = match[0][1]
        if model_dir_name.find('-^-') != -1:
            extend_sequence = True
        # if model_dir_name.find('-A-') != -1:
        #     use_all_chars = True
        match = re.compile(r'-E([\d]+)-').findall(model_dir_name)
        if match:
            use_chars = True
        predict_interactive(model_dir, hidden_size, cell_type)
