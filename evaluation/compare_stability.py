import argparse, math, random, sys
import numpy as np

from utils.util_functions import read_plain_file
from evaluation.constants import *
from utils.constants import *

from evaluation.utils_functions import read_file, load_explainer_checkpoint, get_weights_from_explainer


def split_src_by_category(src):
    # split the train src according to predicted label
    src_map = dict()
    for i, x in enumerate(src):
        predicted_label = x[0][5:-1]
        try:
            previous_src = src_map[predicted_label]
            previous_src.append(i)

        except KeyError:
            src_map[predicted_label] = [i]

    return src_map


def remove_long_doc(src, tgt, max_length):
    new_src, new_tgt = [], []

    for i, doc in enumerate(src):
        if len(doc) <= max_length:
            new_src.append(doc)
            new_tgt.append(tgt[i])
    return np.array(new_src), np.array(new_tgt)


def set_model_parameters():
    parser = argparse.ArgumentParser(description='Compare Variance Parser')
    parser.add_argument('--data', type=str, default='path to data')
    parser.add_argument('--dataset', type=str, choices=[SST, COLA, AGNEWS, IMDB_R])
    parser.add_argument('--doc-num', type=int, help='number of documents selected for evaluation')
    parser.add_argument('--categorize-type', type=str, choices=[RANK, MULTILABEL],
                        help='the categorization type in explainer target')
    parser.add_argument('--original-explainer', nargs='+',
                        default=[KERNEL_SHAP, DEEP_SHAP, OCCLUSION, GRADIENT, LIME, LRP])
    parser.add_argument('--ratio', type=float, default=0.3, help='the ratio to mask when explainer type is RANK')
    parser.add_argument('--similarity-metric', type=str, choices=[BERT_COS, N_GRAM],
                        help='the similarity file collected by find_similar_documents')

    args, _ = parser.parse_known_args(sys.argv)

    args.similarity_file = '%s/fairseq/%s/%s' % (args.data, args.dataset,
                                                 'test_' + args.similarity_metric + '_similarity.txt')
    args.debug_log_file = '%s/fairseq/%s/%s_%s_%s_variance_log_file.txt' % (args.data, args.dataset, args.dataset,
                                                                            args.categorize_type, args.similarity_metric)
    args.explainer_parameters = []
    args.explainer_checkpoint_path = []
    args.test_src_file = []
    args.test_tgt_file = []

    args.test_index_mapping_file = []
    args.save_log_file = []
    for each in args.original_explainer:
        nsample = 'auto' if each in [OCCLUSION, GRADIENT, LRP] else '1000'
        args.explainer_parameters.append('%s/fairseq/%s/%s/%s/explainer_parameters_%s' %
                                         (args.data, args.dataset, each, nsample, args.categorize_type))
        args.explainer_checkpoint_path.append('%s/fairseq/%s/%s/%s/%s_%s/%s' %
                                              (args.data, args.dataset, each, nsample,
                                               'checkpoints', args.categorize_type, 'checkpoint_best.pt'))
        args.test_src_file.append('%s/fairseq/%s/%s/%s/test_src.exp' % (args.data, args.dataset, each, nsample))
        args.test_tgt_file.append('%s/fairseq/%s/%s/%s/test_tgt.exp' % (args.data, args.dataset, each, nsample))

        args.test_index_mapping_file.append('%s/fairseq/%s/%s/%s/test_index.txt' % (args.data, args.dataset, each, nsample))

        args.save_log_file.append('%s/fairseq/%s/%s/%s/%s_%s_exp_variance_log.txt' % (args.data, args.dataset, each,
                                                                                      nsample, args.categorize_type,
                                                                                      args.similarity_metric))
    return args


def get_similar_doc(args, nn):
    similar_docs = read_plain_file(args.similarity_file)
    test_similar_doc_map = dict()
    for each in similar_docs:
        seq = each.strip().split(":::")
        test_idx = int(seq[0])
        similar_train_docs = [int(i) for i in seq[1].split(',')]
        test_similar_doc_map[test_idx] = similar_train_docs[:nn]

    return test_similar_doc_map


def get_index_mapping(args, idx):
    # original ::: exp
    mappings = read_plain_file(args.test_index_mapping_file[idx])
    exp_to_original = dict()
    original_to_exp = dict()
    for each in mappings:
        seq = each.strip().split(":::")
        original_index = int(seq[0])
        exp_index = int(seq[1])

        exp_to_original[exp_index] = original_index
        original_to_exp[original_index] = exp_index
    return exp_to_original, original_to_exp


def filter_long_doc(src, max_len):
    avail_src = []
    for i in range(len(src)):
        if 3 < len(src[i]) <= max_len:  # must have at least one word and lower than what BERT allows
            avail_src.append(1)
        else:
            avail_src.append(0)
    return [i/sum(avail_src) for i in avail_src]


def print_log(log_file, doc, compare_tuple, formulation=MULTILABEL, nn=False):
    if not nn:
        log_file.write('-------------------------------------------------------------------\n')
        log_file.write("##### Current document:\n")
    else:
        log_file.write('------------------\n')
        log_file.write("##### Neighbor documents:\n")
    log_file.write(" ".join(doc) + "\n")
    log_file.write("##### Our Explanation (Pos):\n")
    log_file.write(" ".join(compare_tuple[OUR_EXP_POS]) + "\n")
    log_file.write("##### Base Explanation (Pos):\n")
    log_file.write(" ".join(compare_tuple[BASE_EXP_POS]) + "\n")
    log_file.write("##### Random Explanation (Pos):\n")
    log_file.write(" ".join(compare_tuple[RANDOM_EXP_POS]) + "\n")

    if formulation == MULTILABEL:
        log_file.write("##### Our Explanation (Neg):\n")
        log_file.write(" ".join(compare_tuple[OUR_EXP_NEG]) + "\n")
        log_file.write("##### Base Explanation (Neg):\n")
        log_file.write(" ".join(compare_tuple[BASE_EXP_NEG]) + "\n")
        log_file.write("##### Random Explanation (Neg):\n")
        log_file.write(" ".join(compare_tuple[RANDOM_EXP_NEG]) + "\n")


def compute_iou(current_compare_tuple, neighbor_compare_tuple, formulation=MULTILABEL):
    # UNION
    our_exp_pos_union = list(set(current_compare_tuple[OUR_EXP_POS] + neighbor_compare_tuple[OUR_EXP_POS]))
    our_exp_neg_union = list(set(current_compare_tuple[OUR_EXP_NEG] + neighbor_compare_tuple[OUR_EXP_NEG]))

    base_exp_pos_union = list(set(current_compare_tuple[BASE_EXP_POS] + neighbor_compare_tuple[BASE_EXP_POS]))
    base_exp_neg_union = list(set(current_compare_tuple[BASE_EXP_NEG] + neighbor_compare_tuple[BASE_EXP_NEG]))

    random_exp_pos_union = list(
        set(current_compare_tuple[RANDOM_EXP_POS] + neighbor_compare_tuple[RANDOM_EXP_POS]))
    random_exp_neg_union = list(
        set(current_compare_tuple[RANDOM_EXP_NEG] + neighbor_compare_tuple[RANDOM_EXP_NEG]))

    # INTERSECT
    our_exp_pos_intersect = list(set(current_compare_tuple[OUR_EXP_POS]) & set(neighbor_compare_tuple[OUR_EXP_POS]))
    our_exp_neg_intersect = list(set(current_compare_tuple[OUR_EXP_NEG]) & set(neighbor_compare_tuple[OUR_EXP_NEG]))

    base_exp_pos_intersect = list(set(current_compare_tuple[BASE_EXP_POS]) & set(neighbor_compare_tuple[BASE_EXP_POS]))
    base_exp_neg_intersect = list(set(current_compare_tuple[BASE_EXP_NEG]) & set(neighbor_compare_tuple[BASE_EXP_NEG]))

    random_exp_pos_intersect = list(
        set(current_compare_tuple[RANDOM_EXP_POS]) & set(neighbor_compare_tuple[RANDOM_EXP_POS]))
    random_exp_neg_intersect = list(
        set(current_compare_tuple[RANDOM_EXP_NEG]) & set(neighbor_compare_tuple[RANDOM_EXP_NEG]))

    # IOU: denominator
    our_exp_denominator = len(our_exp_pos_union) + len(our_exp_neg_union) \
        if formulation == MULTILABEL else len(our_exp_pos_union)
    base_exp_denominator = len(base_exp_pos_union) + len(base_exp_neg_union) \
        if formulation == MULTILABEL else len(base_exp_pos_union)
    random_exp_denominator = len(random_exp_pos_union) + len(random_exp_neg_union) \
        if formulation == MULTILABEL else len(random_exp_pos_union)

    # IOU : numerator
    our_exp_numerator = len(our_exp_pos_intersect) + len(our_exp_neg_intersect) \
        if formulation == MULTILABEL else len(our_exp_pos_intersect)
    base_exp_numerator = len(base_exp_pos_intersect) + len(base_exp_neg_intersect) \
        if formulation == MULTILABEL else len(base_exp_pos_intersect)
    random_exp_numerator = len(random_exp_pos_intersect) + len(random_exp_neg_intersect) \
        if formulation == MULTILABEL else len(random_exp_pos_intersect)

    our_iou = 0.0 if our_exp_denominator == 0 else our_exp_numerator / our_exp_denominator
    base_iou = 0.0 if base_exp_denominator == 0 else base_exp_numerator / base_exp_denominator
    random_iou = 0.0 if random_exp_denominator == 0 else random_exp_numerator / random_exp_denominator

    return our_iou, base_iou, random_iou


def main():
    args = set_model_parameters()
    max_len = 100
    nearest_neighbors = 3
    random.seed(1234)

    # read nearest neighbor from file: original test ::: original trains
    similar_doc_map = get_similar_doc(args, nearest_neighbors)

    debug_log_file = open(args.debug_log_file, 'w')

    # load data and base exp
    test_src, test_tgt = read_file(args.test_src_file[0], " "), read_file(args.test_tgt_file[0], ",")
    prob = filter_long_doc(test_src, max_len)
    selected = random.choices(range(len(test_src)), weights=prob, k=args.doc_num)

    for idx in range(len(args.original_explainer)):
        variance_log_file = open(args.save_log_file[idx], 'w')
        # Debug info
        debug_log_file.write('================' + args.original_explainer[idx] + '================' + "\n")
        exp_to_original, original_to_exp = get_index_mapping(args, idx)

        if idx != 0:
            test_src, test_tgt = read_file(args.test_src_file[idx], " "), read_file(args.test_tgt_file[idx], ",")

        # get weights from the trained explanation model
        trainer = load_explainer_checkpoint(args.explainer_parameters[idx], args.explainer_checkpoint_path[idx])

        # check explanation intersection / union in those documents
        for i in selected:

            doc = test_src[i]
            base_weight = test_tgt[i]
            current_compare_tuple = get_exp_tuple(args, doc, base_weight, trainer,
                                                  lrp=args.original_explainer[idx] == LRP)
            # find neighbors
            try:
                # in case the mappings are not finalised by find_neighbours.py yet
                neighbors_indices = [original_to_exp[each] for each in similar_doc_map[exp_to_original[i]]]
            except KeyError:
                continue

            # print debug info
            print_log(debug_log_file, doc, current_compare_tuple, formulation=args.categorize_type)

            our_exp_var = 0.0
            base_exp_var = 0.0
            random_exp_var = 0.0

            for neighbor in neighbors_indices:
                neighbor_src = test_src[neighbor]
                neighbor_tgt = test_tgt[neighbor]

                neighbor_compare_tuple = get_exp_tuple(args, neighbor_src, neighbor_tgt, trainer,
                                                       lrp=args.original_explainer[idx]==LRP)

                print_log(debug_log_file, neighbor_src, neighbor_compare_tuple,
                          formulation=args.categorize_type, nn=True)

                # accumulate IOU value
                our_iou, base_iou, random_iou = compute_iou(current_compare_tuple, neighbor_compare_tuple,
                                                            formulation=args.categorize_type)
                our_exp_var += our_iou
                base_exp_var += base_iou
                random_exp_var += random_iou

            variance_log_file.write("In %s -th instance: the average similarity of base explanation is %.2f%%, "
                                    "the random explanation is %.2f%%, "
                                    "our explanation is %.2f%%.\n" %
                                    (i, base_exp_var / len(neighbors_indices) * 100,
                                     random_exp_var / len(neighbors_indices) * 100,
                                     our_exp_var / len(neighbors_indices) * 100))

            debug_log_file.write("In %s -th instance: the average similarity of base explanation is %.2f%%, "
                                 "the random explanation is %.2f%%, "
                                 "our explanation is %.2f%%.\n" %
                                 (i, base_exp_var / len(neighbors_indices) * 100,
                                  random_exp_var / len(neighbors_indices) * 100,
                                  our_exp_var / len(neighbors_indices) * 100))
            debug_log_file.write('\n')


def get_exp_tuple(args, src, base_weight, explainer, lrp=False):

    our_weight = get_weights_from_explainer(explainer, src, output_dim=5 if args.categorize_type == MULTILABEL else 1).squeeze(0).tolist()

    def sort_base_weight(i):
        return base_weight[i]
    order = sorted(range(len(base_weight)), key=lambda i: sort_base_weight(i), reverse=True)

    if args.categorize_type == RANK:
        def sort_out_weight(i):
            return our_weight[i]
        if not isinstance(our_weight, list):
            print('stop.')
            our_weight = [our_weight]
        interval = math.ceil(len(our_weight) * args.ratio)
        sorted_our_weight = sorted(range(len(our_weight)), key=lambda i: sort_out_weight(i), reverse=True)

        our_exp_pos = [src[i + 1] for i in sorted_our_weight[:interval]]
        our_exp_neg = [src[i + 1] for i in sorted_our_weight[-interval:]]

        # when it's ranking, the base explanation does not need to care if it's > or < 0
        base_exp_pos = list(filter(None, [src[i + 1] if i in order[:interval] else None
                                          for i in range(len(base_weight))]))
        base_exp_neg = list(filter(None, [src[i + 1] if i in order[-interval:] else None
                                          for i in range(len(base_weight))]))

        pos_idx = random.sample(range(1, len(src)-1), interval)
        random_exp_pos = [src[i] for i in pos_idx]
        neg_idx = random.sample(range(1, len(src)-1), interval)
        random_exp_neg = [src[i] for i in neg_idx]
    else:
        # multilabel - non lrp
        if not lrp:
            # 0: high neg; 1: low neg; 2: neutral; 3: low pos; 4: high pos
            pos_interval = sum([1 if x == 3 or x == 4 else 0 for x in our_weight])
            neg_interval = sum([1 if x == 0 or x == 1 else 0 for x in our_weight])

            our_exp_pos = list(filter(None, [src[i + 1] if our_weight[i] == 3 or our_weight[i] == 4 else None
                                             for i in range(len(our_weight))]))
            our_exp_neg = list(filter(None, [src[i + 1] if our_weight[i] == 0 or our_weight[i] == 1 else None
                                             for i in range(len(our_weight))]))

            if pos_interval != 0:
                base_exp_pos = list(filter(None, [src[i + 1] if i in order[:pos_interval] and base_weight[i] > 0.0 else None
                                                  for i in range(len(base_weight))]))
                pos_idx = random.sample(range(1, len(src)-1), pos_interval)
                random_exp_pos = [src[i] for i in pos_idx]
            else:
                base_exp_pos = []
                random_exp_pos = []
            if neg_interval != 0:
                base_exp_neg = list(filter(None, [src[i + 1] if i in order[-neg_interval:] and base_weight[i] < 0.0
                                                  else None for i in range(len(base_weight))]))
                neg_idx = random.sample(range(1, len(src)-1), neg_interval)
                random_exp_neg = [src[i] for i in neg_idx]
            else:

                base_exp_neg = []
                random_exp_neg = []
        else:
            # multilabel - lrp (only pos)
            # 0: high pos; 1: medium pos; 2: low pos
            our_exp_pos, our_exp_neg, base_exp_pos, base_exp_neg, random_exp_pos, random_exp_neg = \
                [], [], [], [], [], []
            pos_interval = sum([1 if x == 0 or x == 1 else 0 for x in our_weight])
            our_exp_pos = list(filter(None, [src[i + 1] if our_weight[i] == 0 or our_weight[i] == 1 else None
                                             for i in range(len(our_weight))]))

            if pos_interval != 0:
                mean_base_weight = np.mean(base_weight)
                std_base_weight = np.std(base_weight)
                base_exp_pos = list(
                    filter(None, [src[i + 1] if i in order[:pos_interval]
                                  and base_weight[i] > mean_base_weight - std_base_weight
                                  else None for i in range(len(base_weight))]))
                pos_idx = random.sample(range(1, len(src) - 1), pos_interval)
                random_exp_pos = [src[i] for i in sorted(pos_idx)]
            else:
                base_exp_pos = []
                random_exp_pos = []

    compare_tuple = {OUR_EXP_POS: our_exp_pos, OUR_EXP_NEG: our_exp_neg,
                     BASE_EXP_POS: base_exp_pos, BASE_EXP_NEG:base_exp_neg,
                     RANDOM_EXP_POS: random_exp_pos, RANDOM_EXP_NEG: random_exp_neg}
    return compare_tuple


if __name__ == "__main__":
    main()
