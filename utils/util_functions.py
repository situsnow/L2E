import torch
import re


def send_to_cuda(a_tensor):
    if torch.cuda.is_available():
        return a_tensor.cuda()
    else:
        return a_tensor


def read_plain_file(path):
    data = []
    with open(path) as f:
        for each in f:
            data.append(each.strip())
    return data


def write_plain_file(path, data):
    with open(path, 'w') as f:
        for each in data:
            f.write(each)


def clean_data(text, lower=True):
    non_word = re.compile(r'(\W+)|$').match
    text = [x for x in list(filter(None, re.split(r'(\W+)|$', clean_str(text, lower)))) if not non_word(x)]
    return text


def clean_str(string, lower=True):
    """
    Tokenization/string cleaning
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    # special handling for web documents
    string = re.sub(r"<br />", " ", string)

    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[^A-Za-z(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    if lower:
        return string.strip().lower()
    else:
        return string.strip()


def count_word_freq(dataset):
    word_freq = {}
    for each in dataset:
        # words = each.split()
        for w in split_sentence(each):
            try:
                previous_count = word_freq[w]
                word_freq[w] = previous_count + 1
            except KeyError:
                word_freq[w] = 1

    return word_freq


def split_sentence(x):
    if type(x) == list:
        return x
    else:
        return x.split(" ")


def join_sentence(x):
    if type(x) == list:
        return " ".join(x)
    else:
        return x


def convert_to_n_gram(placeholder1, placeholder2, x):
    from nltk.util import ngrams

    all_ngrams = []

    def split_sent(x):
        if isinstance(x, list):
            return x
        else:
            return x.split(" ")
    # up till 4-gram or len(x)
    for j in range(1, min(5, len(split_sent(x))+1)):
        ngram = ngrams(split_sent(x), j)
        j_gram = []
        for each in ngram:
            j_gram.append([" ".join(each)][0] if len(each) > 1 else list(each)[0])
        all_ngrams.append(j_gram)

    return all_ngrams


def n_gram_sim(a, b, logarithm=False):
    import math

    def intersect_over_union(a_ngram, b_ngram):
        # a small epsilon value to control no intersect case
        return max(len(set(a_ngram) & set(b_ngram)) / len(set(a_ngram) | set(b_ngram)), epsilon)

    sim = 0.0
    for j in range(min(len(a), len(b))):
        precision = intersect_over_union(a[j], b[j])
        # similarity result has no difference with/out logarithm
        sim += math.log2(precision) if logarithm else precision
    return sim / min(len(a), len(b))