import argparse, sys, os
from pathlib import Path
from nltk.corpus import stopwords

from utils.constants import *
from evaluation.constants import *

from utils.util_functions import read_plain_file, split_sentence
from evaluation.utils_functions import get_weights_from_explainer, load_explainer_checkpoint, paired_sign_test
from evaluation.compare_imdbr_faithfulness import find_mapping_indices

STOPWORDS = stopwords.words('english')


def set_model_parameters():
    parser = argparse.ArgumentParser(description='Compare IMDB_R rationale Parser')
    parser.add_argument('--data-dir', type=str, help='path to data')
    parser.add_argument('--doc-num', type=int, help='number of documents selected for evaluation')
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
    args.similarity_file = args.l2e_folder + 'similarity.txt'
    args.split_file = args.l2e_folder + 'split.txt'

    folder = args.data_dir + '/' + IMDB_R + '/original/'
    args.file_paths = [folder + 'withRats_neg', folder + 'withRats_pos',
                       folder + 'noRats_neg', folder + 'noRats_pos']

    args.src_files = []
    args.tgt_files = []
    args.index_mapping_file = []
    args.explainer_parameters = []
    args.explainer_checkpoint = []
    args.explainer_folder = []

    src_suffix = "_src.exp"
    tgt_suffix = "_tgt.exp"
    index_mapping_suffix = '_index.txt'
    explainer_parameters_file = "explainer_parameters_multilabel"
    explainer_checkpoint = "checkpoints_multilabel/checkpoint_best.pt"

    for ex in args.original_explainer:
        nsample = 'auto' if ex in [OCCLUSION, GRADIENT, LRP] else '500'
        args.explainer_folder.append(os.path.join(args.data_dir, FAIRSEQ, IMDB_R, ex, nsample))
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


def get_similar_doc(args, nn):
    similar_docs = read_plain_file(args.similarity_file)
    test_similar_doc_map = dict()
    for each in similar_docs:
        seq = each.strip().split(":::")
        test_idx = seq[0]
        similar_train_docs = [i for i in seq[1].split(',')]
        test_similar_doc_map[test_idx] = similar_train_docs[:nn]

    return test_similar_doc_map


def get_doc_split(args):
    split_file = read_plain_file(args.split_file)
    doc_split_map = {}
    for each in split_file:
        seq = each.strip().split(":::")
        doc_split_map[seq[0]] = seq[1]
    return doc_split_map


def get_all_explanations_without_stopwords(args, selected_src, explainer, selected_tgt, ex, selected_instances_rationales, idx):
    human_rationale = selected_instances_rationales[idx]
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
        l2e_exp_count = [1 if pure_l2e_exp[w] == 0 and selected_src[w] else 0 for w in range(len(pure_l2e_exp))]
        l2e_exp = list(filter(None, [selected_src[w] if pure_l2e_exp[w] == 0 and selected_src[w] not in STOPWORDS else None
                                     for w in range(len(pure_l2e_exp))]))
        if len(l2e_exp) == 0:
            l2e_exp_count = [1 if pure_l2e_exp[w] == 1 else 0 for w in range(len(pure_l2e_exp))]
            l2e_exp = list(filter(None, [selected_src[w] if pure_l2e_exp[w] == 1 and selected_src[w] not in STOPWORDS else None
                                         for w in range(len(pure_l2e_exp))]))
    else:
        # non lrp - 0: high neg; 1: low neg; 2: neutral; 3: low pos; 4: high pos
        l2e_exp_count = [1 if pure_l2e_exp[w] == 4 else 0 for w in range(len(pure_l2e_exp))]
        l2e_exp = list(filter(None, [selected_src[w] if pure_l2e_exp[w] == 4 and selected_src[w] not in STOPWORDS else None
                                     for w in range(len(pure_l2e_exp))]))
        if len(l2e_exp) == 0:
            l2e_exp_count = [1 if pure_l2e_exp[w] == 3 else 0 for w in range(len(pure_l2e_exp))]
            l2e_exp = list(filter(None, [selected_src[w] if pure_l2e_exp[w] == 3 and selected_src[w] not in STOPWORDS else None
                                         for w in range(len(pure_l2e_exp))]))

    if len(l2e_exp) == 0 or sum(l2e_exp_count) == len(pure_l2e_exp):
        return (False, )

    # ensure base will have same number of words in rationale
    base_exp = list(filter(None, [selected_src[w]
                                  if w in ordered_base_exp[:sum(l2e_exp_count)] and selected_src[w] not in STOPWORDS else None
                                  for w in range(len(pure_base_exp))]))

    human_rationale = list(filter(lambda x: x!=MASK, human_rationale))
    human_rationale = list(filter(None, [x if x not in STOPWORDS else None for x in human_rationale]))
    return (l2e_exp, base_exp, human_rationale)


def get_all_explanations_all_words(args, selected_src, explainer, selected_tgt, ex, selected_instances_rationales, idx):
    human_rationale = selected_instances_rationales[idx]
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
        l2e_exp_count = [1 if pure_l2e_exp[w] == 0 else 0 for w in range(len(pure_l2e_exp))]
        l2e_exp = list(filter(None, [selected_src[w] if pure_l2e_exp[w] == 0 else None
                                     for w in range(len(pure_l2e_exp))]))
        if len(l2e_exp) == 0:
            l2e_exp_count = [1 if pure_l2e_exp[w] == 1 else 0 for w in range(len(pure_l2e_exp))]
            l2e_exp = list(filter(None, [selected_src[w] if pure_l2e_exp[w] == 1 else None
                                         for w in range(len(pure_l2e_exp))]))
    else:
        # non lrp - 0: high neg; 1: low neg; 2: neutral; 3: low pos; 4: high pos
        l2e_exp_count = [1 if pure_l2e_exp[w] == 4 else 0 for w in range(len(pure_l2e_exp))]
        l2e_exp = list(filter(None, [selected_src[w] if pure_l2e_exp[w] == 4 else None
                                     for w in range(len(pure_l2e_exp))]))
        if len(l2e_exp) == 0:
            l2e_exp_count = [1 if pure_l2e_exp[w] == 3 else 0 for w in range(len(pure_l2e_exp))]
            l2e_exp = list(filter(None, [selected_src[w] if pure_l2e_exp[w] == 3 else None
                                         for w in range(len(pure_l2e_exp))]))

    if len(l2e_exp) == 0 or sum(l2e_exp_count) == len(pure_l2e_exp):
        return (False, )

    # ensure base will have same number of words in rationale
    base_exp = list(filter(None, [selected_src[w]
                                  if w in ordered_base_exp[:sum(l2e_exp_count)] else None
                                  for w in range(len(pure_base_exp))]))

    human_rationale = list(filter(lambda x: x!=MASK, human_rationale))
    return (l2e_exp, base_exp, human_rationale)


# just consider rationales
def get_all_explanations_only_rationales(args, selected_src, explainer, selected_tgt, ex, selected_instances_rationales, idx):
    human_rationale = selected_instances_rationales[idx]
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
        l2e_exp_count = [1 if pure_l2e_exp[w] == 0 else 0 for w in range(len(pure_l2e_exp))]
        l2e_exp = list(filter(None, [selected_src[w] if pure_l2e_exp[w] == 0 and human_rationale[w] != MASK else None
                                     for w in range(len(pure_l2e_exp))]))
        if len(l2e_exp) == 0:
            l2e_exp_count = [1 if pure_l2e_exp[w] == 1 else 0 for w in range(len(pure_l2e_exp))]
            l2e_exp = list(filter(None, [selected_src[w] if pure_l2e_exp[w] == 1 and human_rationale[w] != MASK else None
                                         for w in range(len(pure_l2e_exp))]))
    else:
        # non lrp - 0: high neg; 1: low neg; 2: neutral; 3: low pos; 4: high pos
        l2e_exp_count = [1 if pure_l2e_exp[w] == 4 else 0 for w in range(len(pure_l2e_exp))]
        l2e_exp = list(filter(None, [selected_src[w] if pure_l2e_exp[w] == 4 and human_rationale[w] != MASK else None
                                     for w in range(len(pure_l2e_exp))]))
        if len(l2e_exp) == 0:
            l2e_exp_count = [1 if pure_l2e_exp[w] == 3 else 0 for w in range(len(pure_l2e_exp))]
            l2e_exp = list(filter(None, [selected_src[w] if pure_l2e_exp[w] == 3 and human_rationale[w] != MASK else None
                                         for w in range(len(pure_l2e_exp))]))

    if len(l2e_exp) == 0 or sum(l2e_exp_count) == len(pure_l2e_exp):
        return (False, )

    # ensure base will have same number of words in rationale
    base_exp = list(filter(None, [selected_src[w]
                                  if w in ordered_base_exp[:sum(l2e_exp_count)] and human_rationale[w] != MASK else None
                                  for w in range(len(pure_base_exp))]))

    human_rationale = list(filter(lambda x: x!=MASK, human_rationale))
    return (l2e_exp, base_exp, human_rationale)


def extract_original_rationales(args, selected_instances, split):
    selected_instances_rationales = {}
    for idx in selected_instances:
        # get the neutral src
        folder = split

        neutral_src = Path(os.path.join(args.l2e_folder, 'dev' if folder == VALID else folder,
                                        'neu_' + idx)).read_text()
        # filter empty in case double space
        neutral_src = list(filter(None, neutral_src.split(" ")))
        no_masking_src = Path(os.path.join(args.l2e_folder, 'dev' if folder == VALID else folder, idx)).read_text()
        no_masking_src = no_masking_src.strip().split(" ")

        # we reverse the neutral_src where rationales are revealed and mask all the rest
        selected_instances_rationales[idx] = [no_masking_src[w] if neutral_src[w] == MASK else MASK
                                              for w in range(len(neutral_src))]
    return selected_instances_rationales


if __name__ == "__main__":
    args = set_model_parameters()
    nearest_neighbors = 3

    observe_cat = POS
    strategy = ONLY_RATIONALE  # EXCLUDE_STOPWORD
    if strategy == ALL_WORDS:
        func_get_explanation = get_all_explanations_all_words
    elif strategy == ONLY_RATIONALE:
        func_get_explanation = get_all_explanations_only_rationales
    else:
        func_get_explanation = get_all_explanations_without_stopwords

    similar_doc_map = get_similar_doc(args, nearest_neighbors)
    doc_split_map = get_doc_split(args)

    # we will only examine test set
    train_correct_instances = read_plain_file(args.longformer_acc_files[TRAIN])[0].strip().split(",")
    valid_correct_instances = read_plain_file(args.longformer_acc_files[VALID])[0].strip().split(",")
    test_correct_instances = read_plain_file(args.longformer_acc_files[TEST])[0].strip().split(",")

    train_selected_instances = list(filter(None, [i if observe_cat in i else None for i in train_correct_instances]))
    valid_selected_instances = list(filter(None, [i if observe_cat in i else None for i in valid_correct_instances]))
    test_selected_instances = list(filter(None, [i if observe_cat in i else None for i in test_correct_instances]))

    train_selected_rationales = extract_original_rationales(args, train_selected_instances, TRAIN)
    valid_selected_rationales = extract_original_rationales(args, valid_selected_instances, VALID)
    test_selected_rationales = extract_original_rationales(args, test_selected_instances, TEST)

    p_values = []
    # p_value_base = []
    for ex in range(len(args.original_explainer)):
        print("================ " + args.original_explainer[ex] + " ================")

        train_id_to_exp, train_exp_to_id = find_mapping_indices(args.index_mapping_file[ex][TRAIN])
        valid_id_to_exp, valid_exp_to_id = find_mapping_indices(args.index_mapping_file[ex][VALID])
        test_id_to_exp, test_exp_to_id = find_mapping_indices(args.index_mapping_file[ex][TEST])
        # read src from current explainer folder
        # and get corresponding explanations of base - tgt.exp
        train_src_exp = read_plain_file(args.src_files[ex][TRAIN])
        train_tgt_exp = read_plain_file(args.tgt_files[ex][TRAIN])

        valid_src_exp = read_plain_file(args.src_files[ex][VALID])
        valid_tgt_exp = read_plain_file(args.tgt_files[ex][VALID])

        test_src_exp = read_plain_file(args.src_files[ex][TEST])
        test_tgt_exp = read_plain_file(args.tgt_files[ex][TEST])

        # 3. load explainer, get explanations from explainer
        explainer = load_explainer_checkpoint(args.explainer_parameters[ex], args.explainer_checkpoint[ex])

        skip_count = 0

        plot_l2e = []
        plot_base = []
        plot_human = []
        for idx in test_selected_rationales:
            selected_src = test_src_exp[test_id_to_exp[idx]].strip()
            selected_tgt = test_tgt_exp[test_id_to_exp[idx]].strip().split(",")  # base

            all_exp = func_get_explanation(args, selected_src, explainer, selected_tgt, ex,
                                           test_selected_rationales, idx)
            if len(all_exp) < 2:
                skip_count += 1
                continue

            l2e_exp, base_exp, human_rationale = all_exp
            # print("Current example explanation %d %d %d" % (len(l2e_exp), len(base_exp), len(human_rationale)))

            all_l2e_int = []
            all_l2e_union = []
            all_base_int = []
            all_base_union = []
            all_human_int = []
            all_human_union = []
            l2e_var = 0.0
            base_var = 0.0
            human_var = 0.0
            # get neighbours
            nn_count = 0
            for nn in similar_doc_map[idx]:
                nn_split = doc_split_map[nn]
                if nn_split == TRAIN:
                    nn_selected_src = train_src_exp[train_id_to_exp[nn]].strip()
                    nn_selected_tgt = train_tgt_exp[train_id_to_exp[nn]].strip().split(",")  # base
                    nn_all_exp = func_get_explanation(args, nn_selected_src, explainer, nn_selected_tgt, ex,
                                                      train_selected_rationales, nn)
                elif nn_split == DEV or nn_split == VALID:
                    nn_selected_src = valid_src_exp[valid_id_to_exp[nn]].strip()
                    nn_selected_tgt = valid_tgt_exp[valid_id_to_exp[nn]].strip().split(",")  # base
                    nn_all_exp = func_get_explanation(args, nn_selected_src, explainer, nn_selected_tgt, ex,
                                                      valid_selected_rationales, nn)
                else:
                    nn_selected_src = test_src_exp[test_id_to_exp[nn]].strip()
                    nn_selected_tgt = test_tgt_exp[test_id_to_exp[nn]].strip().split(",")  # base
                    nn_all_exp = func_get_explanation(args, nn_selected_src, explainer, nn_selected_tgt, ex,
                                                      test_selected_rationales, nn)

                if len(nn_all_exp) < 2:
                    continue
                nn_count += 1
                nn_l2e_exp, nn_base_exp, nn_human_rationales = nn_all_exp
                # print("Neighbour explanation %d %d %d" % (len(nn_l2e_exp), len(nn_base_exp), len(nn_human_rationales)))

                # intersect
                l2e_int = list(set(l2e_exp) & set(nn_l2e_exp))
                base_int = list(set(base_exp) & set(nn_base_exp))
                human_int = list(set(human_rationale) & set(nn_human_rationales))

                print("Example %s and %s have common explanation in L2E: %s" % (idx, nn, ",".join(l2e_int)))
                print("Example %s and %s have common explanation in Base: %s" % (idx, nn, ",".join(base_int)))
                if strategy == ONLY_RATIONALE:
                    print("Example %s and %s have common rationales: %s" % (idx, nn, ",".join(human_int)))
                # print("Intersection %d %d %d" % (len(l2e_int), len(base_int), len(human_int)))
                # union
                l2e_union = list(set(l2e_exp + nn_l2e_exp))
                base_union = list(set(base_exp + nn_base_exp))
                human_union = list(set(human_rationale + nn_human_rationales))
                # print("Union %d %d %d" % (len(l2e_union), len(base_union), len(human_union)))
                # var
                all_l2e_int.append(len(l2e_int))
                all_l2e_union.append(len(l2e_union))
                all_base_int.append(len(base_int))
                all_base_union.append(len(base_union))
                all_human_int.append(len(human_int))
                all_human_union.append(len(human_union))

            if nn_count != 0:
                l2e_var = 0.0 if sum(all_l2e_union) == 0 else sum(all_l2e_int) / sum(all_l2e_union)
                base_var = 0.0 if sum(all_base_union) == 0 else sum(all_base_int) / sum(all_base_union)
                human_var = sum(all_human_int) / sum(all_human_union)
                l2e_iou = l2e_var/ nn_count
                base_iou = base_var / nn_count
                human_iou = human_var / nn_count
                if strategy == ONLY_RATIONALE:
                    print("In example %s, the average IoU of L2E is %.2f%%, Base is %.2f%%, original Human is %.2f%%"
                          % (idx, l2e_iou * 100, base_iou * 100, human_iou * 100))
                else:
                    print("In example %s, the average IoU of L2E is %.2f%%, Base is %.2f%%"
                          % (idx, l2e_iou * 100, base_iou * 100))
                # if l2e_iou != 0:
                plot_l2e.append(round(l2e_iou * 100, 2))
                plot_base.append(round(base_iou * 100, 2))
                plot_human.append(round(human_iou * 100, 2))
                print("==============================================================")

        # order according to human rationales
        def sort_rationale_iou(i):
            return plot_human[i]
        sorted_plot_human = sorted(range(len(plot_human)), key=lambda i:sort_rationale_iou(i), reverse=True)

        plot_l2e = [plot_l2e[i] for i in sorted_plot_human]
        plot_base = [plot_base[i] for i in sorted_plot_human]
        plot_human = [plot_human[i] for i in sorted_plot_human]

        p_values.append(paired_sign_test(plot_base, plot_l2e))

    print(p_values)