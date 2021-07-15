import argparse, sys, os, random
from pathlib import Path
from nltk.corpus import stopwords
import numpy as np

from utils.constants import *
from evaluation.constants import *

from utils.util_functions import read_plain_file, split_sentence
from evaluation.utils_functions import mean_confidence_interval, paired_sign_test, \
    load_explainer_checkpoint, get_weights_from_explainer

STOPWORDS = stopwords.words('english')


def print_metrics(*argv):
    print(METRICS)
    i = 0
    all_m, all_h = [], []
    for arg in argv:
        m, h = mean_confidence_interval(arg, round_decimal=4)
        # print_str = str(m) + '$\pm$' + str(h)
        # print(METRICS[i] + ": " + print_str, end=', ')
        all_m.append(str(m))
        all_h.append(str(h))
        i += 1
    print(', '.join(all_m))
    print(','.join([all_m[i] + '$\pm$' + all_h[i] for i in range(len(all_m))]))


def set_model_parameters():
    parser = argparse.ArgumentParser(description='Compare IMDB_R rationale Parser')
    parser.add_argument('--data-dir', type=str, help='path to data')
    parser.add_argument('--doc-num', type=int, help='number of documents selected for evaluation')
    # [KERNEL_SHAP, DEEP_SHAP, LIME, OCCLUSION, GRADIENT, LRP]
    parser.add_argument('--original-explainer', nargs='+',
                        default=[OCCLUSION, GRADIENT, LRP, LIME, KERNEL_SHAP, DEEP_SHAP])
    parser.add_argument('--all-splits', nargs='+',
                        default=[TRAIN, VALID, TEST])
    parser.add_argument('--top-k', type=int, default=20)

    args, _ = parser.parse_known_args(sys.argv)

    args.l2e_folder = args.data_dir + '/' + IMDB_R + '/l2e/'
    args.longformer_acc_files = {TRAIN: args.l2e_folder + 'train_acc.txt',
                                 VALID: args.l2e_folder + 'valid_acc.txt',
                                 TEST: args.l2e_folder + 'test_acc.txt'}

    folder = args.data_dir + '/' + IMDB_R + '/original/'
    args.file_paths = [folder + 'withRats_neg', folder + 'withRats_pos',
                       folder + 'noRats_neg', folder + 'noRats_pos']

    args.src_files = []
    args.tgt_files = []
    args.index_mapping_file = []
    args.explainer_parameters = []
    args.explainer_checkpoint = []

    src_suffix = "_src.exp"
    tgt_suffix = "_tgt.exp"
    index_mapping_suffix = '_index.txt'
    explainer_parameters_file = "explainer_parameters_multilabel"
    explainer_checkpoint = "checkpoints_multilabel/checkpoint_best.pt"

    for ex in args.original_explainer:
        nsample = 'auto' if ex in [OCCLUSION, GRADIENT, LRP, GROUND_TRUTH] else '500'
        args.src_files.append({TRAIN: os.path.join(args.data_dir, FAIRSEQ, IMDB_R, ex, nsample, TRAIN + src_suffix),
                               VALID: os.path.join(args.data_dir, FAIRSEQ, IMDB_R, ex, nsample, VALID + src_suffix),
                               TEST: os.path.join(args.data_dir, FAIRSEQ, IMDB_R, ex, nsample, TEST + src_suffix)})
        args.tgt_files.append({TRAIN: os.path.join(args.data_dir, FAIRSEQ, IMDB_R, ex, nsample, TRAIN + tgt_suffix),
                               VALID: os.path.join(args.data_dir, FAIRSEQ, IMDB_R, ex, nsample, VALID + tgt_suffix),
                               TEST: os.path.join(args.data_dir, FAIRSEQ, IMDB_R, ex, nsample, TEST + tgt_suffix)})
        args.index_mapping_file.append({TRAIN: os.path.join(args.data_dir, FAIRSEQ, IMDB_R, ex, nsample,
                                                            TRAIN + index_mapping_suffix),
                                        VALID: os.path.join(args.data_dir, FAIRSEQ, IMDB_R, ex, nsample,
                                                            VALID + index_mapping_suffix),
                                        TEST: os.path.join(args.data_dir, FAIRSEQ, IMDB_R, ex, nsample,
                                                           TEST + index_mapping_suffix)})

        args.explainer_parameters.append(os.path.join(args.data_dir, FAIRSEQ, IMDB_R, ex, nsample,
                                                      explainer_parameters_file))
        args.explainer_checkpoint.append(os.path.join(args.data_dir, FAIRSEQ, IMDB_R, ex, nsample,
                                                      explainer_checkpoint))

    return args


def find_mapping_indices(filepath):
    all_mappings = [l.strip().split(":::") for l in read_plain_file(filepath)]
    id_to_exp_mapping_dict = {}
    exp_to_id_mapping_dict = {}
    for l in all_mappings:
        id_to_exp_mapping_dict[l[0]] = int(l[1])
        exp_to_id_mapping_dict[int(l[1])] = l[0]

    return id_to_exp_mapping_dict, exp_to_id_mapping_dict


def compute_cm_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0   # true positive rate, sensitivity
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    tn_rate = tn / (tn + fp) if (tn + fp) > 0 else 0 # specificity
    return accuracy, precision, recall, f1, fp_rate, tn_rate


class metrics:
    def __init__(self):
        self.tp, self.tn, self.fp, self.fn = [], [], [], []
        self.acc, self.pr, self.re, self.f1, self.fp_rate, self.tn_rate = [], [], [], [], [], []


def get_pairwise_cm(human_rationale, base_exp, l2e_exp, base_metrics, l2e_metrics):
    tn = sum([1 if human_rationale[w] == base_exp[w] == 0 else 0
              for w in range(len(selected_src))])
    tp = sum([1 if human_rationale[w] == base_exp[w] == 1 else 0
              for w in range(len(selected_src))])
    fp = sum([1 if human_rationale[w] == 0 and base_exp[w] == 1 else 0
              for w in range(len(selected_src))])
    fn = sum([1 if human_rationale[w] == 1 and base_exp[w] == 0 else 0
              for w in range(len(selected_src))])

    accuracy, precision, recall, f1, fp_rate, tn_rate = \
        compute_cm_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
    base_metrics.tn.append(tn)
    base_metrics.tp.append(tp)
    base_metrics.fp.append(fp)
    base_metrics.fn.append(fn)
    base_metrics.acc.append(accuracy)
    base_metrics.pr.append(precision)
    base_metrics.re.append(recall)
    base_metrics.f1.append(f1)
    base_metrics.fp_rate.append(fp_rate)
    base_metrics.tn_rate.append(tn_rate)

    tp = sum([1 if human_rationale[w] == l2e_exp[w] == 1 else 0
              for w in range(len(selected_src))])
    tn = sum([1 if human_rationale[w] == l2e_exp[w] == 0 else 0
              for w in range(len(selected_src))])
    fp = sum([1 if human_rationale[w] == 0 and l2e_exp[w] == 1 else 0
              for w in range(len(selected_src))])
    fn = sum([1 if human_rationale[w] == 1 and l2e_exp[w] == 0 else 0
              for w in range(len(selected_src))])

    accuracy, precision, recall, f1, fp_rate, tn_rate = \
        compute_cm_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
    l2e_metrics.tn.append(tn)
    l2e_metrics.tp.append(tp)
    l2e_metrics.fp.append(fp)
    l2e_metrics.fn.append(fn)
    l2e_metrics.acc.append(accuracy)
    l2e_metrics.pr.append(precision)
    l2e_metrics.re.append(recall)
    l2e_metrics.f1.append(f1)
    l2e_metrics.fp_rate.append(fp_rate)
    l2e_metrics.tn_rate.append(tn_rate)


def count_rationale_word_freq(word_dict, list_of_words):
    for w in list_of_words:
        # if w in stopwords.words('english'):
        #     continue
        try:
            freq = word_dict[w]
            word_dict[w] = freq + 1
        except KeyError:
            word_dict[w] = 1
    return word_dict


if __name__ == "__main__":
    args = set_model_parameters()

    args.randomness = False if args.doc_num is None or args.doc_num < 1 else True
    observe_cat = POS

    for split in range(len(args.all_splits)):
        print("================ " + args.all_splits[split] + " ================")
        # get file with ground-truth rationale (also being predicted correctly by longformer)
        # (original_src, _, original_indices), rationales, (_, _, _) = load_original_IMDB_R(args)
        correct_instances = read_plain_file(args.longformer_acc_files[args.all_splits[split]])[0].strip().split(",")

        selected_instances = list(filter(None, [i if observe_cat in i else None for i in correct_instances]))
        if args.randomness:
            random.seed(1234)
            selected_instances = random.sample(correct_instances, min(args.doc_num, len(correct_instances)))

        # e.g. {posR_218.txt: [0, 0, ....1, 1,...]} where position of 1 represents rationale
        selected_instances_rationales = {}
        for idx in selected_instances:

            # get the neutral src
            folder = args.all_splits[split]

            neutral_src = Path(os.path.join(args.l2e_folder, 'dev' if folder == VALID else folder,
                                            'neu_' + idx)).read_text()
            # filter empty in case double space
            selected_instances_rationales[idx] = [1 if w == MASK else 0
                                                  for w in list(filter(None, neutral_src.split(" ")))]

        all_p_values_pr = []
        all_p_values_re = []
        all_p_values_f1 = []
        for ex in range(len(args.original_explainer)):
            print("================ " + args.original_explainer[ex] + " ================")
            id_to_exp, exp_to_id = find_mapping_indices(args.index_mapping_file[ex][args.all_splits[split]])
            # read src from current explainer folder
            # and get corresponding explanations of base - tgt.exp
            src_exp = read_plain_file(args.src_files[ex][args.all_splits[split]])
            tgt_exp = read_plain_file(args.tgt_files[ex][args.all_splits[split]])

            # 3. load explainer, get explanations from explainer
            explainer = load_explainer_checkpoint(args.explainer_parameters[ex], args.explainer_checkpoint[ex])

            base_metrics = metrics()
            l2e_metrics = metrics()

            pos_rationale_words, neg_rationale_words = {}, {}
            pos_base_words, neg_base_words = {}, {}
            pos_l2e_words, neg_l2e_words = {}, {}

            skip_count = 0
            for idx in selected_instances:
                selected_src = src_exp[id_to_exp[idx]].strip()
                selected_tgt = tgt_exp[id_to_exp[idx]].strip().split(",")  # base

                pure_l2e_exp = get_weights_from_explainer(explainer, selected_src).squeeze(0).tolist()
                if len(pure_l2e_exp) > len(selected_tgt):
                    # the explainer is a longformer, needs to trim the padding positions
                    pure_l2e_exp = pure_l2e_exp[:len(selected_tgt)]

                pure_base_exp = [float(w) for w in selected_tgt]

                # since we already have the l2e explanation, remove the label tag
                selected_src = split_sentence(selected_src)[1:-1]

                def sort_weight(i):
                    return pure_base_exp[i]

                ordered_base_exp = sorted(range(len(pure_base_exp)), key=lambda i: sort_weight(i), reverse=True)

                if args.original_explainer[ex] == LRP:
                    # lrp - 0: high pos; 1: medium pos; 2: low pos
                    l2e_exp = [1 if pure_l2e_exp[w] == 0 and selected_src[w] not in STOPWORDS
                               else 0 for w in range(len(pure_l2e_exp))]
                    if sum(l2e_exp) == 0:
                        l2e_exp = [1 if pure_l2e_exp[w] == 1 and selected_src[w] not in STOPWORDS
                                   else 0 for w in range(len(pure_l2e_exp))]
                elif args.original_explainer[ex] == GROUND_TRUTH:
                    l2e_exp = [1 if pure_l2e_exp[w] == 1 and selected_src[w] not in STOPWORDS
                               else 0 for w in range(len(pure_l2e_exp))]
                else:
                    # non lrp - 0: high neg; 1: low neg; 2: neutral; 3: low pos; 4: high pos
                    l2e_exp = [1 if pure_l2e_exp[w] == 4 and selected_src[w] not in STOPWORDS
                               else 0 for w in range(len(pure_l2e_exp))]
                    if sum(l2e_exp) == 0:
                        l2e_exp = [1 if pure_l2e_exp[w] == 3 and selected_src[w] not in STOPWORDS
                                   else 0 for w in range(len(pure_l2e_exp))]

                if sum(l2e_exp) == 0 or sum(l2e_exp) == len(l2e_exp):
                    skip_count += 1
                    continue

                # ensure base will have same number of words in rationale
                base_exp = [1 if w in ordered_base_exp[:sum(l2e_exp)] and selected_src[w] not in STOPWORDS
                            else 0 for w in range(len(pure_base_exp))]

                # 4. find all number for confusion matrix
                human_rationale = selected_instances_rationales[idx]
                # remove stop_words
                human_rationale = [1 if human_rationale[w] == 1 and selected_src[w] not in STOPWORDS else 0
                                   for w in range(len(human_rationale))]

                get_pairwise_cm(human_rationale, base_exp, l2e_exp, base_metrics, l2e_metrics)


            print("Skip %d of instances." % skip_count)
            # 5. calculate the macro/micro metrics
            METRICS = [ACC, PR, RE, F1, FP_RATE, TN_RATE]
            print_metrics(base_metrics.acc, base_metrics.pr, base_metrics.re, base_metrics.f1,
                          base_metrics.fp_rate, base_metrics.tn_rate)
            print("Base Micro-average precision: " + str(round(np.mean(base_metrics.pr), 4)))
            print("Base Macro-average precision: " + str(round(
                sum(base_metrics.tp) / (sum(base_metrics.tp) + sum(base_metrics.fp)), 4)
            ))
            print("Base Micro-average recall: " + str(round(np.mean(base_metrics.re), 4)))
            print("Base Macro-average recall:" + str(round(
                sum(base_metrics.tp) / (sum(base_metrics.tp) + sum(base_metrics.fn)), 4)
            ))

            print("======================")
            print_metrics(l2e_metrics.acc, l2e_metrics.pr, l2e_metrics.re, l2e_metrics.f1,
                          l2e_metrics.fp_rate, l2e_metrics.tn_rate)
            print("L2E Micro-average precision: " + str(round(np.mean(l2e_metrics.pr), 4)))
            print("L2E Macro-average precision: " + str(round(
                sum(l2e_metrics.tp) / (sum(l2e_metrics.tp) + sum(l2e_metrics.fp)), 4)
            ))
            print("L2E Micro-average recall: " + str(round(np.mean(l2e_metrics.re), 4)))
            print("L2E Macro-average recall:" + str(round(
                sum(l2e_metrics.tp) / (sum(l2e_metrics.tp) + sum(l2e_metrics.fn)), 4)
            ))

            all_p_values_pr.append(paired_sign_test(base_metrics.pr, l2e_metrics.pr) / 2)
            all_p_values_re.append(paired_sign_test(base_metrics.re, l2e_metrics.re) / 2)
            all_p_values_f1.append(paired_sign_test(base_metrics.f1, l2e_metrics.f1) / 2)

        print(all_p_values_pr)
        print(all_p_values_re)
        print(all_p_values_f1)
