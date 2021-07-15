import argparse, sys, torch, os
import numpy as np

from utils.constants import *
from evaluation.constants import *

from utils.utils_data import get_bbox_by_dataset, data_files, get_dataset_load_func
from utils.utils_model import load_blackbox, prediction_by_bert

epsilon = 1e-200


def set_model_parameters():
    parser = argparse.ArgumentParser(description='Find similar documents for consistency analysis')
    parser.add_argument('--data', type=str, help='path to data')
    parser.add_argument('--dataset', type=str, choices=[AGNEWS, SST, COLA])
    parser.add_argument('--sim-metric', type=str, choices=[BERT_COS, N_GRAM])
    parser.add_argument('--log-precision', default=False, action='store_true',
                        help='if logarithm is required in similarity metric, only availble in n-gram, '
                             'default No logarithm.')

    args, _ = parser.parse_known_args(sys.argv)
    args.bbox_path = get_bbox_by_dataset(args.data, args.dataset)
    args.saved_blackbox_model = args.data + args.dataset + '/' + args.bbox_path
    args.save_file = '%s/fairseq/%s/test_%s_similarity.txt' % (args.data, args.dataset, args.sim_metric)
    args.lower = True
    return args


# find the similar documents among test set
def main():
    max_len = 100
    nearest_neighbors = 10

    args = set_model_parameters()
    args.file_paths = data_files(args.data, args.dataset)
    (_, _), (_, _), (test_data, _) = get_dataset_load_func(
        args.dataset)(args)

    # load black-box
    tokenizer, bbox = load_blackbox(args.bbox_path)

    saved_test = []
    if os.path.exists(args.save_file):
        with open(args.save_file, 'r') as f:
            for each in f:
                saved_test.append(int(each.split(':::')[0]))

    label_data_map = dict()

    for i, x in enumerate(test_data):
        if len(x) > max_len or len(x) == 0:
            continue
        else:
            y_hat = prediction_by_bert(tokenizer, bbox, " ".join(x))
            max_y_hat = np.argmax(y_hat)
            try:
                data_indices = label_data_map[max_y_hat]
            except KeyError:
                data_indices = []
            data_indices.append(i)
            label_data_map[max_y_hat] = data_indices

    pretrained_bert, bert_tokenizer = None, None
    if args.sim_metric == BERT_COS:
        # load bert for similarity measure
        from transformers import BertModel, BertTokenizer
        # vectorize src according to BERT pre-training models
        pretrained_bert = BertModel.from_pretrained(args.data + '/transformers_models/')
        bert_tokenizer = BertTokenizer.from_pretrained(args.data + '/transformers_models/')
    # read from test file
    for i, x_test in enumerate(test_data):
        if len(x_test) > max_len or i in saved_test or len(x_test) == 0:
            continue
        y_hat_test = prediction_by_bert(tokenizer, bbox, " ".join(x_test))
        max_y_hat_test = np.argmax(y_hat_test)
        # load all test data [filter long documents] whose ground-truth label = test predicted label
        all_same_label_data = label_data_map[max_y_hat_test]

        func = convert_to_bert_hidden_states if args.sim_metric == BERT_COS else convert_to_n_gram
        sim_metric = bert_cos_sim if args.sim_metric == BERT_COS else n_gram_sim
        converted_x_test = func(bert_tokenizer, pretrained_bert, x_test)
        similarities = dict()
        for each in all_same_label_data:
            if i != each:   # find similar documents other than itself
                x = test_data[each]
                converted_x = func(bert_tokenizer, pretrained_bert, x)
                # calculate all similarity
                similarities[each] = sim_metric(converted_x_test, converted_x, args.log_precision)
        # sort similarity
        # save up to 10 documents index along with this exp test file
        sorted_similarity = dict(sorted(similarities.items(), key=lambda kv: kv[1], reverse=True))

        neighbors_indices = list(sorted_similarity.keys())[:nearest_neighbors]
        neighbors_indices = [str(i) for i in neighbors_indices]

        with open(args.save_file, 'a') as f:
            f.write(str(i) + ":::" + ','.join(neighbors_indices) + '\n')


def convert_to_bert_hidden_states(bert_tokenizer, pretrained_bert, x):
    with torch.no_grad():
        x = " ".join(x).replace("<", "[").replace(">", "]")
        input_ids = torch.tensor([bert_tokenizer.encode(x)])
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        hidden_states, _ = pretrained_bert(input_ids)[-2:]
        return torch.sum(hidden_states.squeeze(0), dim=0)


def bert_cos_sim(a, b, dummy):
    from torch.nn import CosineSimilarity
    cos = CosineSimilarity(dim=0, eps=1e-6)
    return cos(a, b).item()


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


if __name__ == "__main__":
    main()