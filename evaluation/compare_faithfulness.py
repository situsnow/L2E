import random, math, argparse, sys
import numpy as np
from utils.constants import *
from utils.utils_model import load_blackbox, prediction_by_bert
from utils.utils_data import get_bbox_by_dataset

from evaluation.utils_functions import read_file, get_weights_from_explainer, load_explainer_checkpoint, \
    paired_sign_test, mean_confidence_interval


def masked_random_tokens(src, interval, masked_token, times=5):
    new_srcs = []
    for _ in range(times):
        masked_indices = random.sample(range(1, len(src) - 1), interval)
        new_src = [masked_token if i in masked_indices else src[i] for i in range(1, len(src) - 1)]
        new_srcs.append(new_src)

    return new_srcs


def masked_explainer_tokens(src, weights, masked_token, args, interval, flag=False, lrp=False):
    if args.explainer_type == RANK:
        def sort_weight(i):
            return weights[i]

        if args.cat.lower() == POS:
            order = sorted(range(len(weights)), key=lambda i: sort_weight(i), reverse=True)
            weights = [0 if i in order[:interval] else 1 for i in range(len(weights))]
        else:
            order = sorted(range(len(weights)), key=lambda i: sort_weight(i))
            weights = [0 if i in order[:interval] else 1 for i in range(len(weights))]

        new_src = [masked_token if weights[i - 1] == 0 else src[i] for i in range(1, len(src) - 1)]
    else:
        # multilabel, not lrp
        if args.cat.lower() == POS and not lrp:
            # 3: low pos; 4: high pos
            if not flag:
                new_src = [masked_token if weights[i - 1] == 3 or weights[i - 1] == 4 else src[i] for i in
                           range(1, len(src) - 1)]
            else:
                # only get top pos
                high_pos_count = sum([1 if w == 4 else 0 for w in weights])
                if high_pos_count == 0:
                    new_src = [masked_token if weights[i - 1] == 3 else src[i] for i in range(1, len(src) - 1)]
                else:
                    new_src = [masked_token if weights[i - 1] == 4 else src[i] for i in range(1, len(src) - 1)]
        elif lrp:
            # multilabel, lrp
            # 0: high pos; 1: medium pos; 2: low pos
            pos_count = sum([1 if w == 0 or w == 1 else 0 for w in weights])
            if pos_count == len(weights):
                new_src = [masked_token if weights[i-1] == 0 else src[i] for i in
                           range(1, len(src) - 1)]
            else:
                new_src = [masked_token if weights[i - 1] == 0 or weights[i - 1] == 1 else src[i] for i in
                           range(1, len(src) - 1)]
        else:
            # 0: high neg; 1: low neg;
            new_src = [masked_token if weights[i - 1] == 0 or weights[i - 1] == 1 else src[i] for i in
                       range(1, len(src) - 1)]

            # only get top neg
            # new_src = [masked_token if weights[i - 1] == 0 else src[i] for i in
            #            range(1, len(src) - 1)]
    return new_src


def masked_top_weight_tokens(args, src, weights, masked_token, interval=0):
    def sort_weight(i):
        return weights[i]

    if args.cat.lower() == POS:
        order = sorted(range(len(weights)), key=lambda i: sort_weight(i), reverse=True)
        if args.explainer_type == RANK:
            # when it's ranking, the base explanation does not need to care if it's > or < 0
            weights = [0 if i in order[:interval] else 1 for i in range(len(weights))]
        else:
            # multilabel
            weights = [0 if i in order[:interval] and weights[i] > 0 else 1 for i in range(len(weights))]
    else:
        order = sorted(range(len(weights)), key=lambda i: sort_weight(i))
        if args.explainer_type == RANK:
            # when it's ranking, the base explanation does not need to care if it's > or < 0
            weights = [0 if i in order[:interval] else 1 for i in range(len(weights))]
        else:
            weights = [0 if i in order[:interval] and weights[i] < 0 else 1 for i in range(len(weights))]

    new_src = [masked_token if weights[i - 1] == 0 else src[i] for i in range(1, len(src) - 1)]

    return new_src


def set_model_parameters():
    parser = argparse.ArgumentParser(description='Log odds Parser')
    # parser.add_argument("--preprocess", action='store_true')
    parser.add_argument('--dataset', type=str, choices=[SST, COLA, AGNEWS])
    parser.add_argument('--cat', type=str, choices=[POS, NEG], help='the category of weights')
    parser.add_argument('--data', type=str, help='path to the data')
    parser.add_argument('--doc-num', type=int, help='number of documents selected for evaluation')
    parser.add_argument('--original-explainer', nargs='+', default=[KERNEL_SHAP, DEEP_SHAP, LIME, OCCLUSION, GRADIENT, LRP])
    parser.add_argument('--explainer-type', type=str, choices=[RANK, MULTILABEL])
    parser.add_argument('--ratio', type=float, default=0.3, help='the ratio to mask when explainer type is RANK')

    args, _ = parser.parse_known_args(sys.argv)

    args.src_file = []
    args.tgt_file = []
    args.explainer_parameters = []
    args.explainer_checkpoint_path = []
    args.continuous_explainer_parameters = []
    args.continuous_explainer_checkpoint_path = []
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


def get_interval_by_type(args, explainer_tgt, flag=False, lrp=False):
    if args.explainer_type == RANK:
        if isinstance(explainer_tgt, list):
            return math.ceil(len(explainer_tgt) * args.ratio)
        else:
            return 1
    elif args.explainer_type == MULTILABEL and not lrp:
        # multilabel - non lrp
        if args.cat == 'pos':
            # 3: low pos; 4: high pos
            if not flag:  # the flag to choose all pos or only top pos
                interval = sum([1 if x == 3 or x == 4 else 0 for x in explainer_tgt])
            # only get top pos
            else:
                interval = sum([1 if x == 4 else 0 for x in explainer_tgt])
                if interval == 0:
                    interval = sum([1 if x == 3 else 0 for x in explainer_tgt])
        else:
            # 0: high neg; 1: low neg;
            interval = sum([1 if x == 0 or x == 1 else 0 for x in explainer_tgt])
            # only get top neg
            # interval = sum([1 if x == 0 else 0 for x in explainer_tgt])
    else:
        # multilabel - lrp, there could only be args.cat == pos
        # 0: high pos; 1: medium pos; 2: low pos
        interval = sum([1 if x == 0 or x == 1 else 0 for x in explainer_tgt])
        if interval == len(explainer_tgt):
            interval = sum([1 if x == 0 else 0 for x in explainer_tgt])

    return interval


def main():
    args = set_model_parameters()
    # load black-box
    tokenizer, bbox = load_blackbox(args.blackbox_checkpoint_path)
    masked_token = tokenizer.mask_token

    seed = 1234
    random.seed(seed)
    # load original data and ground-truth from kernel_shap (which has lowest number of ground-truths)
    src, tgt = read_file(args.src_file[0], " "), read_file(args.tgt_file[0], ",")
    selected = random.sample(range(len(src)), args.doc_num)

    # for statistical test
    base_our_p_values = {}
    base_reported_log_odds = {}
    our_reported_log_odds = {}

    random_delta_log_odds = []
    for idx in range(len(args.original_explainer)):
        skip_count = 0
        if idx != 0:
            src, tgt = read_file(args.src_file[idx], " "), read_file(args.tgt_file[idx], ",")
        src_selected, tgt_selected = src[selected], tgt[selected]
        # get weights from the trained explanation model
        trainer = load_explainer_checkpoint(args.explainer_parameters[idx], args.explainer_checkpoint_path[idx])

        base_delta_log_odds, our_continuous_delta_log_odds, our_delta_log_odds = [], [], []
        for i, each in enumerate(src_selected):
            # exclude doc with only one word + front/back label tag
            if len(each) == 3:
                continue
            explainer_tgt = \
                get_weights_from_explainer(trainer, each,
                                           output_dim=5 if args.explainer_type == MULTILABEL else 1).squeeze(0).tolist()

            # check interval to decide how to mask src
            interval = get_interval_by_type(args, explainer_tgt, lrp=args.original_explainer[idx] == LRP)
            if interval == 0 or interval == len(explainer_tgt):
                skip_count += 1
                continue

            # masked tokens according to cat and their weights in both sets (base and our model)
            random_srcs = masked_random_tokens(each, interval, masked_token)
            base_src = masked_top_weight_tokens(args, each, tgt_selected[i], masked_token, interval=interval)

            our_src = masked_explainer_tokens(each, explainer_tgt, masked_token, args, interval,
                                              lrp=args.original_explainer[idx] == LRP)

            # get prediction
            original_pred = prediction_by_bert(tokenizer, bbox, " ".join(each[1:len(each) - 1]))
            random_pred = []
            for k in range(len(random_srcs)):
                random_pred.append(prediction_by_bert(tokenizer, bbox, " ".join(random_srcs[i])))
            base_masked_pred = prediction_by_bert(tokenizer, bbox, " ".join(base_src))
            our_masked_pred = prediction_by_bert(tokenizer, bbox, " ".join(our_src))

            predicted_index = np.argmax(original_pred)

            original_log_odds = log_odds(original_pred[predicted_index])
            random_log_odds = []
            for k in range(len(random_srcs)):
                random_log_odds.append(log_odds(random_pred[i][predicted_index]))
            base_log_odds = log_odds(base_masked_pred[predicted_index])
            our_log_odds = log_odds(our_masked_pred[predicted_index])

            random_delta_log_odds.append(sum([original_log_odds - l for l in random_log_odds]) / len(random_srcs))
            base_delta_log_odds.append(original_log_odds - base_log_odds)
            our_delta_log_odds.append(original_log_odds - our_log_odds)

        print(skip_count)
        # for log odd table
        base_str, ours_str = log_odds_table(our_delta_log_odds, base_delta_log_odds, args.original_explainer[idx])

        base_reported_log_odds[args.original_explainer[idx]] = base_str
        our_reported_log_odds[args.original_explainer[idx]] = ours_str

        # for plot
        # all_log_odds_base.append(base_delta_log_odds)
        # all_log_odds_our.append(our_delta_log_odds)

        # for statistical test
        p_value = paired_sign_test(base_delta_log_odds, our_delta_log_odds)
        base_our_p_values[args.original_explainer[idx]] = p_value

    # for statistical test
    # holm_bonferroni(p_values, 0.05, args.original_explainer)
    base_row = 'Base'
    our_row = "Our"

    explainer_row = "Methods"
    base_our_stat_p_value = []
    for i in args.original_explainer:
        explainer_row += " & " + i
        base_row += " & " + base_reported_log_odds[i]
        our_row += " & " + our_reported_log_odds[i]
        base_our_stat_p_value.append(base_our_p_values[i])
    m, h = mean_confidence_interval(random_delta_log_odds)
    print("Random logodds: " + str(m) + '$\pm$' + str(h))
    print(explainer_row)
    print(base_row)
    print(our_row)
    print('Base:::Our--> ', end='')
    print(base_our_stat_p_value)

    # plot boxplot; If pos words are masked, higher delta log odds is better; and neg words, lower is better
    # box_plot(args, all_log_odds_base, all_log_odds_our)


def log_odds_table(our_delta_log_odds, base_delta_log_odds, explainer):
    print("=======" + explainer)
    m, h = mean_confidence_interval(base_delta_log_odds)
    base_str = str(m) + '$\pm$' + str(h)

    m, h = mean_confidence_interval(our_delta_log_odds)
    our_str = str(m) + '$\pm$' + str(h)
    return base_str, our_str


def box_plot(args, all_log_odds_base, all_log_odds_our):
    import matplotlib.pyplot as plt

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure()
    bp_base = plt.boxplot(all_log_odds_base, positions=np.array(range(len(all_log_odds_base))) * 2.0 - 0.4, sym='',
                          widths=0.6)
    bp_our = plt.boxplot(all_log_odds_our, positions=np.array(range(len(all_log_odds_our))) * 2.0 + 0.4, sym='',
                         widths=0.6)
    set_box_color(bp_base, '#D7191C')
    set_box_color(bp_our, '#2C7BB6')
    plt.plot([], c='#D7191C', label='Baseline')
    plt.plot([], c='#2C7BB6', label='Our')
    plt.legend()
    plt.xticks(range(0, len(args.original_explainer) * 2, 2), args.original_explainer)
    plt.xlim(-2, len(args.original_explainer) * 2)
    # plt.ylim(0, 8)
    plt.tight_layout()
    label = 'Positive' if args.cat == 'pos' else 'Negative'
    plt.title("Log odds by masking " + label + " words")
    plt.savefig(args.plot_item + '.png')
    plt.show()


def log_odds(prob):
    return math.log(prob / (1 - prob)) if prob < 1.0 else 0.0


if __name__ == "__main__":
    main()