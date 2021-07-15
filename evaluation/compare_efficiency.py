import random, time, argparse, sys, torch
import numpy as np
from pathlib import Path
from sklearn.utils import check_random_state

from utils.constants import *
from evaluation.constants import *
from utils.util_functions import write_plain_file, read_plain_file, send_to_cuda, join_sentence

from utils.utils_data import get_bbox_by_dataset, get_dataset_categories, get_dataset_load_func, data_files
from utils.utils_model import load_blackbox
from evaluation.utils_functions import get_weights_from_explainer, load_explainer_checkpoint
from utils.utils_algo import get_teacher_explanations


def set_model_parameters():
    parser = argparse.ArgumentParser(description='Compare efficiency Parser')
    # parser.add_argument("--preprocess", action='store_true')
    parser.add_argument('--dataset', type=str, choices=[SST, COLA, AGNEWS, IMDB_R])
    parser.add_argument('--data-dir', type=str, help='path to data')
    parser.add_argument('--doc-num', type=int, help='number of documents selected for evaluation')
    parser.add_argument('--record', type=str, choices=[BASE, OUR])
    parser.add_argument('--original-explainer', type=str,
                        choices=[KERNEL_SHAP, DEEP_SHAP, LIME, OCCLUSION, GRADIENT, LRP])
    parser.add_argument('--mini-batch', type=int, help='the mini batch size to get the output from BERT.')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='the size in generating pertubed neighbor samples')
    parser.add_argument('--suffix', type=str, default='')

    args, _ = parser.parse_known_args(sys.argv)

    args.fairseq_path = '%s/%s/%s/' % (args.data_dir, 'fairseq', args.dataset)
    args.selected_docs = args.fairseq_path + 'efficiency_selected_docs.txt'

    file_suffix = args.original_explainer + args.suffix + '.txt' if args.record == BASE else 'our.txt'
    args.efficiency_file = args.fairseq_path + 'efficiency_' + file_suffix

    args.blackbox_checkpoint_path = get_bbox_by_dataset(args.data_dir, args.dataset)

    args.seed = 1234
    args.random_state = check_random_state(args.seed)  # for LIME
    # args.mini_batch = 200   # for kernel_shap
    args.norm = 'no_norm'  # a placeholder, not useful in this pgm
    args.categories = get_dataset_categories(args.dataset)
    args.class_num = len(args.categories)

    if 'cased' in args.blackbox_checkpoint_path:
        args.lower = False
    else:
        args.lower = True

    return args


def main():
    args = set_model_parameters()

    args.file_paths = data_files(args.data_dir, args.dataset)
    (train_data, train_labels), (_, _), (_, _) = get_dataset_load_func(
        args.dataset)(args)

    if not Path(args.selected_docs).exists():
        selected_idx = random.sample(range(len(train_data)), args.doc_num)
        write_plain_file(args.selected_docs, ",".join([str(i) for i in selected_idx]))
    else:
        selected_idx = read_plain_file(args.selected_docs)
        selected_idx = [int(i) for i in selected_idx[0].strip().split(',')]

    selected_x = np.array(train_data)[selected_idx]
    selected_y = np.array(train_labels)[selected_idx]

    write_file = open(args.efficiency_file, 'w')
    if args.record == BASE:
        # get base time
        tokenizer, model = load_blackbox(args.blackbox_checkpoint_path, args.original_explainer == LRP)

        for i, x in enumerate(selected_x):
            y = selected_y[i]
            input_id = send_to_cuda(torch.tensor([tokenizer.encode(join_sentence(x))]))

            logits = model(input_id)[0]
            y_hat = torch.argmax(logits)

            args.explain_label = y_hat.item()

            start_time = time.time()

            weights = get_teacher_explanations(args.original_explainer)\
                (args, tokenizer, model, x, y, input_id, logits, train_split=True, i=i, train_data=train_data)

            end_time = time.time()
            print_log = str(i) + ":::" + str(end_time - start_time) + "\n"
            write_file.write(print_log)
            write_file.flush()

    else:

        for i, each in enumerate(selected_x):
            # get our time, default, get the one from occlusion, rank
            explainer_parameters = '%s/fairseq/%s/gradient/auto/explainer_parameters_multilabel' % (args.data_dir, args.dataset)
            explainer_checkpoint_path = '%s/fairseq/%s/gradient/auto/checkpoints_multilabel/checkpoint_best.pt' % \
                                        (args.data_dir, args.dataset)
            # start the counting time
            trainer = load_explainer_checkpoint(explainer_parameters, explainer_checkpoint_path)
            start_time = time.time()
            explainer_tgt = get_weights_from_explainer(trainer, each, output_dim=1).squeeze(0).tolist()
            end_time = time.time()

            print_log = str(i)+":::"+str(end_time - start_time)+"\n"
            write_file.write(print_log)

    write_file.close()


if __name__ == "__main__":
    main()