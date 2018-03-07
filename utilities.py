import re
import sys
import time
import logging
import numpy as np

__author__ = "Mingjie Qian"
__date__ = "January 28th, 2018"

NUM = 'NUM'
UNK = 'UNK'

encoding = 'utf-8'


class TextDataset(object):
    """Class that iterates over a text Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filepath, annotation_scheme=None, extend_sequence=False, max_iter=None, file=None):
        """
        Args:
            filepath: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filepath = filepath
        self.annotation_scheme = annotation_scheme
        self.extend_sequence = extend_sequence
        self.max_iter = max_iter
        self.file = file
        self.length = None

    def __iter__(self):
        niter = 0
        if self.file is not None:
            f = self.file
        else:
            f = open(self.filepath, 'r', encoding=encoding)
        if self.annotation_scheme == 'CoNLL':
            split_pattern = re.compile(r'[ \t]')
            # with open(self.filepath, 'r', encoding=encoding) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    # ls = line.split(' ')
                    ls = split_pattern.split(line)
                    word, tag = ls[0],ls[-1]
                    word = word.lower()
                    words += [word]
                    tags += [tag]
        else:
            from SequenceLabelingBRNNCRF import parse_a_tagged_query_xml
            from SequenceLabelingBRNNCRF import parse_a_tagged_query_bracket
            if self.annotation_scheme == '<>':
                parse_a_tagged_query = parse_a_tagged_query_xml
            elif self.annotation_scheme == '[]':
                parse_a_tagged_query = parse_a_tagged_query_bracket
            else:
                raise Exception("%s is not a supported annotation scheme." % self.annotation_scheme)
            # with open(self.filepath, "r", encoding=encoding) as f:
            for tagged_query in f:
                tagged_query = tagged_query.strip()
                if not tagged_query:
                    continue
                niter += 1
                if self.max_iter is not None and niter > self.max_iter:
                    break
                tag_seq, wrd_seq = parse_a_tagged_query(tagged_query, None, False, self.extend_sequence)
                yield wrd_seq, tag_seq
        f.close()

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def fetch_minibatches(data, minibatch_size):
    """
    Copied from https://github.com/guillaumegenthial/sequence_tagging

    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of word sequences, list of tag sequences

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        # if type(x[0]) == tuple:
        #     x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Copied from https://github.com/guillaumegenthial/sequence_tagging

    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_length_sentence)

    return sequence_padded, sequence_length


def get_data_input(x_batch, y_batch, vocab, tag_dict, use_chars=False, char_dict=None):
    from SequenceLabelingBRNNCRF import is_number
    wrd_id_seqs = []
    chr_id_mats = []
    label_seqs = []
    for wrd_seq, tag_seq in zip(x_batch, y_batch):
        wrd_id_seq = [vocab[w if w in vocab else NUM if is_number(w) else UNK] for w in wrd_seq]
        label_seq = [tag_dict[tag] for tag in tag_seq]
        wrd_id_seqs.append(wrd_id_seq)
        label_seqs.append(label_seq)
        if use_chars:
            chr_id_mat = []
            for word in wrd_seq:
                chr_id_seq = [char_dict[ch] for ch in word]
                chr_id_mat.append(chr_id_seq)
            chr_id_mats.append(chr_id_mat)
    wrd_id_seqs, _ = pad_sequences(wrd_id_seqs, 0)
    label_seqs, _ = pad_sequences(label_seqs, 0)
    if use_chars:
        chr_id_mats, _ = pad_sequences(chr_id_mats, pad_tok=0, nlevels=2)

    return wrd_id_seqs, chr_id_mats, label_seqs


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def get_logger(filename):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
        logger: (instance of logger)

    """
    logger = logging.getLogger('logger')
    # print(logger.handlers)  # A child logger initially doesn't have a handler

    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    # print(logging.getLogger().handlers)

    lhStdout = logging.getLogger().handlers[0]  # stdout is the only handler initially for the root logger
    # Remove the handler to stdout from the root's handlers list so that the logging information won't
    # display in the stdout.
    logging.getLogger().removeHandler(lhStdout)

    return logger


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)
