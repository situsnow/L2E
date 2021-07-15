import random, argparse, sys
import numpy as np
from utils.constants import *
from evaluation.constants import *

from utils.utils_data import get_bbox_by_dataset
from utils.utils_model import load_blackbox, prediction_by_bert

from evaluation.utils_functions import read_file, load_explainer_checkpoint, get_weights_from_explainer, \
    proportions_z_test
from evaluation.compare_faithfulness import get_interval_by_type, \
    masked_top_weight_tokens, masked_random_tokens, masked_explainer_tokens


def set_model_parameters():
    parser = argparse.ArgumentParser(description='Log odds Parser')
    # parser.add_argument("--preprocess", action='store_true')
    parser.add_argument('--dataset', type=str, choices=[SST, COLA, AGNEWS])
    parser.add_argument('--cat', type=str, default=POS, choices=[POS, NEG], help='the category of weights')
    parser.add_argument('--data', type=str, help='path to data')
    parser.add_argument('--doc-num', type=int, help='number of documents selected for evaluation')
    parser.add_argument('--original-explainer', nargs='+', default=[KERNEL_SHAP, DEEP_SHAP, LIME, OCCLUSION, GRADIENT, LRP])
    parser.add_argument('--explainer-type', type=str, choices=[RANK, MULTILABEL])
    parser.add_argument('--ratio', type=float, default=0.3, help='the ratio to mask when explainer type is RANK')

    args, _ = parser.parse_known_args(sys.argv)

    args.src_file = []
    args.tgt_file = []
    args.explainer_parameters = []
    args.explainer_checkpoint_path = []
    for each in args.original_explainer:
        nsample = 'auto' if each in [OCCLUSION, GRADIENT, LRP] else '1000'

        args.src_file.append('%s/fairseq/%s/%s/%s/test_src.exp' % (args.data, args.dataset, each, nsample))
        args.tgt_file.append('%s/fairseq/%s/%s/%s/test_tgt.exp' % (args.data, args.dataset, each, nsample))
        args.explainer_parameters.append('%s/fairseq/%s/%s/%s/explainer_parameters_%s' %
                                         (args.data, args.dataset, each, nsample, args.explainer_type))
        args.explainer_checkpoint_path.append('%s/fairseq/%s/%s/%s/%s/%s' %
                                              (args.data, args.dataset, each, nsample,
                                               'checkpoints_' + args.explainer_type, 'checkpoint_best.pt'))

    args.blackbox_checkpoint_path = get_bbox_by_dataset(args.data, args.dataset)
    args.plot_item = '%s/fairseq/%s/log_odds_%s_%s' % (args.data, args.dataset, args.cat, args.explainer_type)

    return args


def extract_random_tokens(src, interval, masked_token):
    new_srcs = []
    for _ in range(5):
        extract_indices = random.sample(range(1, len(src) - 1), interval)
        new_src = [src[i] if i in extract_indices else masked_token for i in range(1, len(src)-1)]
        new_srcs.append(new_src)
    return new_srcs


def extract_explainer_tokens(src, weights, masked_token, args, interval, flag=False, lrp=False):
    if args.explainer_type == RANK:
        def sort_weight(i):
            return weights[i]

        if args.cat == "pos":
            order = sorted(range(len(weights)), key=lambda i: sort_weight(i), reverse=True)
            weights = [0 if i in order[:interval] else 1 for i in range(len(weights))]
        else:
            order = sorted(range(len(weights)), key=lambda i: sort_weight(i))
            weights = [0 if i in order[:interval] else 1 for i in range(len(weights))]

        new_src = [src[i] if weights[i - 1] == 0 else masked_token for i in range(1, len(src) - 1)]
    else:
        if args.cat == "pos" and not lrp:
            if not flag:
                # 3: low pos; 4: high pos
                new_src = [src[i] if weights[i-1] == 3 or weights[i-1] == 4 else masked_token for i in range(1, len(src)-1)]
            else:
                high_pos_count = sum(1 if weights[i-1] == 4 else 0 for i in range(1, len(src) - 1))
                if high_pos_count == 0:
                    new_src = [src[i] if weights[i - 1] == 3 else masked_token for i in range(1, len(src) - 1)]
                else:
                    new_src = [src[i] if weights[i - 1] == 4 else masked_token for i in range(1, len(src) - 1)]
        elif lrp:
            # lrp:
            # 0: high pos; 1: medium pos; 2: low pos
            all_pos_count = sum([1 if w == 0 or w == 1 else 0 for w in weights])
            if all_pos_count == len(weights):
                # only get high pos
                new_src = [src[i] if weights[i - 1] == 0 else masked_token
                           for i in range(1, len(src) - 1)]
            else:

                new_src = [src[i] if weights[i - 1] == 0 or weights[i - 1] == 1 else masked_token
                           for i in range(1, len(src) - 1)]
        else:
            # neg
            # 0: high neg; 1: low neg;
            new_src = [src[i] if weights[i-1] == 0 or weights[i-1] == 1 else masked_token for i in range(1, len(src)-1)]

    return new_src


def extract_top_weight_tokens(args, src, weights, masked_token, interval=0):
    def sort_weight(i):
        return weights[i]

    if args.cat == POS:
        order = sorted(range(len(weights)), key=lambda i: sort_weight(i), reverse=True)
        if args.explainer_type == RANK:
            # when it's ranking, the base explanation does not need to care if it's > or < 0
            weights = [0 if i in order[:interval] else 1 for i in range(len(weights))]
        else:
            weights = [0 if i in order[:interval] and weights[i] > 0 else 1 for i in range(len(weights))]
    else:
        order = sorted(range(len(weights)), key=lambda i: sort_weight(i))
        if args.explainer_type == RANK:
            # when it's ranking, the base explanation does not need to care if it's > or < 0
            weights = [0 if i in order[:interval] else 1 for i in range(len(weights))]
        else:
            weights = [0 if i in order[:interval] and weights[i] < 0 else 1 for i in range(len(weights))]

    new_src = [src[i] if weights[i-1] == 0 else masked_token for i in range(1, len(src)-1)]

    return new_src


def main():
    args = set_model_parameters()
    # load black-box
    tokenizer, bbox = load_blackbox(args.blackbox_checkpoint_path)
    masked_token = tokenizer.mask_token

    format = EXTRACT_FORMAT    # MASK_FORMAT

    seed = 1000
    random.seed(seed)

    base_p_values = {}    # the p_value while comparing proportions between base and random agreement
    our_p_values = {}
    for idx in range(len(args.original_explainer)):

        print("===============" + args.original_explainer[idx])
        # if idx != 0:
        src, tgt = read_file(args.src_file[idx], " "), read_file(args.tgt_file[idx], ",")
        selected = random.sample(range(len(src)), args.doc_num)
        src_selected, tgt_selected = src[selected], tgt[selected]
        # get weights from the trained explanation model
        trainer = load_explainer_checkpoint(args.explainer_parameters[idx], args.explainer_checkpoint_path[idx])

        base_agreement = 0
        our_agreement = 0
        our_agreement_with_base = 0
        random_agreement = 0
        total = 0

        for i, each in enumerate(src_selected):
            # exclude doc with only one word + front/back label tag
            if len(each) == 3:
                continue
            explainer_tgt = get_weights_from_explainer(trainer, each, output_dim=5 if args.explainer_type == MULTILABEL else 1).squeeze(0).tolist()

            # check interval to decide how to mask src
            interval = -1  # -1: match the number of pos/neg in our result
            interval = get_interval_by_type(args, explainer_tgt, flag=True,
                                            lrp=args.original_explainer[idx] == LRP)   # choose only high pos
            # print(interval)
            if interval == 0 or interval == len(explainer_tgt):  # we cannot select all words
                continue

            # masked tokens according to cat and their weights in both sets (base and our model)
            base_src = select_mask_or_extract(format, BASE)\
                (args, each, tgt_selected[i], masked_token, interval=interval)

            # get only top pos
            our_src = select_mask_or_extract(format, OUR)(each, explainer_tgt, masked_token, args, interval, flag=True,
                                                          lrp=args.original_explainer[idx] == LRP)

            random_srcs = select_mask_or_extract(format, RANDOM)(each, interval, masked_token)

            original_pred = prediction_by_bert(tokenizer, bbox, " ".join(each[1:len(each)-1]))
            base_pred = prediction_by_bert(tokenizer, bbox, " ".join(base_src))
            our_pred = prediction_by_bert(tokenizer, bbox, " ".join(our_src))
            random_pred = []
            for k in range(len(random_srcs)):
                random_pred.append(prediction_by_bert(tokenizer, bbox, " ".join(random_srcs[k])))

            original_y_hat = np.argmax(original_pred)
            base_y_hat = np.argmax(base_pred)
            our_y_hat = np.argmax(our_pred)

            random_y_hat = []
            for k in range(len(random_pred)):
                random_y_hat.append(np.argmax(random_pred[k]))

            base_agreement += 1 if base_y_hat == original_y_hat else 0
            our_agreement += 1 if our_y_hat == original_y_hat else 0
            our_agreement_with_base += 1 if our_y_hat == base_y_hat else 0
            # random_agreement += 1 if random_y_hat == original_y_hat else 0
            # random_ag_count = sum([1 if i == original_y_hat else 0 for i in random_y_hat])
            random_disag_count = sum([1 if i != original_y_hat else 0 for i in random_y_hat])
            random_agreement += 0 if random_disag_count > 0 else 1
            total += 1

        print(total)
        print('Acc_B: %.2f' % (base_agreement / total), end=',')
        print('Acc_O: %.2f' % (our_agreement / total), end=',')
        print('Acc_O_B: %.2f' % (our_agreement_with_base / total))
        print('Acc_Random: %.2f' % (random_agreement / total))
        count, nobs = [base_agreement, random_agreement], [total, total]
        base_p_values[args.original_explainer[idx]] = proportions_z_test(count, nobs)
        count = [our_agreement, random_agreement]
        our_p_values[args.original_explainer[idx]] = proportions_z_test(count, nobs)

    new_base_p_values, new_our_p_values = [], []
    for exp in args.original_explainer:
        new_base_p_values.append(base_p_values[exp])
        new_our_p_values.append(our_p_values[exp])
    print('Base with random p_value: ', end='')
    print(new_base_p_values)
    print('Our with random p_value:', end='')
    print(new_our_p_values)


def select_mask_or_extract(format, model):
    all_functions={MASK_FORMAT: {BASE: masked_top_weight_tokens,
                                 OUR: masked_explainer_tokens,
                                 RANDOM: masked_random_tokens},
                   EXTRACT_FORMAT: {BASE: extract_top_weight_tokens,
                                    OUR: extract_explainer_tokens,
                                    RANDOM: extract_random_tokens}}
    return all_functions[format][model]


if __name__ == "__main__":
    main()