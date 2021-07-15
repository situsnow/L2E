import argparse
import os.path
import sys
from pathlib import Path

from sklearn.utils import check_random_state
from utils.utils_data import *
from utils.utils_model import load_blackbox
from utils.constants import LRP
from utils.util_functions import join_sentence
from utils.utils_algo import get_teacher_explanations


def set_model_parameters():
    parser = argparse.ArgumentParser(description='Preprocess Parser')

    parser.add_argument('--data-dir', type=str, help="Directory to the data")
    parser.add_argument('--dataset', type=str, help="Dataset name")

    parser.add_argument('--seed', type=int)

    parser.add_argument("--teacher-explainer", type=str, help="The base explanation algorithm, e.g., LIME")
    parser.add_argument('--sample-size', type=int, default=None,
                        help='the size in generating pertubed neighbor samples')
    parser.add_argument('--mini-batch', type=int, help='the mini batch size to get the output from BERT.')
    parser.add_argument('--norm', type=str, help='the way to normalize the attribution (explanation) of each feature.')

    args, _ = parser.parse_known_args(sys.argv)

    # ----------------- for dataset  ----------------- #
    args.categories = get_dataset_categories(args.dataset)
    args.class_num = len(args.categories)

    args.save_path = "%s/%s/" % (args.data_dir, args.dataset)

    # ----------------- for general purpose ----------------- #
    args.valid_ratio = 0.1
    args.label_token = "lbl"
    args.label_format = "<{}-{}>"
    torch.manual_seed(args.seed)

    # ----------------- for original data path ----------------- #
    args.saved_blackbox_model = get_bbox_by_dataset(args.data_dir, args.dataset)

    # ----------------- for fairseq data path ----------------- #
    if args.sample_size == 0 or args.sample_size is None:
        args.sample_size = "auto"
    args.fairseq_path = os.path.join(args.data_dir, 'fairseq', args.dataset, args.teacher_explainer, args.sample_size)

    train_data_fairseq = os.path.join(args.fairseq_path, "train_src.exp")
    train_weights_fairseq = os.path.join(args.fairseq_path, "train_tgt.exp")

    valid_data_fairseq = os.path.join(args.fairseq_path, "valid_src.exp")
    valid_weights_fairseq = os.path.join(args.fairseq_path, "valid_tgt.exp")

    test_data_fairseq = os.path.join(args.fairseq_path, "test_src.exp")
    test_weights_fairseq = os.path.join(args.fairseq_path, "test_tgt.exp")

    args.fairseq_files_dict = {'train': (train_data_fairseq, train_weights_fairseq),
                               'valid': (valid_data_fairseq, valid_weights_fairseq),
                               'test': (test_data_fairseq, test_weights_fairseq)}

    # create directory inside fairseq if not exitst
    if not os.path.exists(args.fairseq_path):
        from pathlib import Path
        Path(args.fairseq_path).mkdir(parents=True, exist_ok=True)
    # dict
    args.dict_fairseq = args.fairseq_path + "dict.txt"
    # ----------------- for LIME perturbation ----------------- #
    args.random_state = check_random_state(args.seed)

    if 'cased' in args.saved_blackbox_model:
        args.lower = False
    else:
        args.lower = True

    return args


def gen_dict(args, train_data, train_labels):
    with open(args.dict_fairseq, 'w') as d:
        # get frequencies of labels
        labels = {}
        for i, each in enumerate(args.categories):
            labels[args.label_format.format(args.label_token, args.categories[i])] = sum(
                [1 if i == j else 0 for j in train_labels])
        labels = sorted(labels.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)  # labels is a list of tuples
        written_index = 0

        word_freq = count_word_freq(train_data)
        sorted_word_freq = sorted(word_freq.items(), key=lambda kv: kv[1], reverse=True)

        for each in sorted_word_freq:

            while written_index < len(labels) and \
                    each[1] < labels[written_index][1]:
                # write label token frequencies into dictionary
                d.write("{} {}{}".format(labels[written_index][0], labels[written_index][1], "\n"))
                written_index += 1
            d.write("{} {}{}".format(each[0], each[1], "\n"))


def gen_explanations(args, tokenizer, bbox, data, labels, split, train_data):

    data_file = open(args.fairseq_files_dict[split][0], 'w')
    explanation_file = open(args.fairseq_files_dict[split][1], 'w')

    for i in range(len(data)):
        x = data[i]
        y = labels[i]

        input_id = send_to_cuda(torch.tensor([tokenizer.encode(join_sentence(x), truncation=True)]))
        logits = bbox(input_id)[0]
        y_hat = torch.argmax(logits)

        # will use it for L2E training
        label_token = args.label_format.format(args.label_token, args.categories[int(y_hat)])

        # only explain the predicted label
        args.explain_label = y_hat.item()
        explanation = get_teacher_explanations(args.teacher_explainer)(args, tokenizer, bbox, x, y, input_id, logits,
                                                                       train_split=(split == 'train'), i=i,
                                                                       train_data=train_data)

        # in case there's no explanation generated
        if any(w != 0.0 for w in explanation):
            # write data, with predicted label in front and back
            data_file.write("{} {} {}{}".format(label_token, join_sentence(x), label_token, '\n'))
            data_file.flush()

            explanation.write("{}{}".format(",".join(map(str, explanation)), "\n"))
            explanation_file.flush()

        else:
            print('Cannot find explanation in document - %s' % (join_sentence(x)))

        del input_id
        del y

    data_file.close()
    explanation_file.close()


def main():
    # collect all hyper-parameters
    args = set_model_parameters()

    # load blackbox models
    tokenizer, model = load_blackbox(args.saved_blackbox_model, lrp=args.teacher_explainer == LRP)

    # load original data
    args.file_paths = data_files(args.data_dir, args.dataset)
    (train_data, train_labels), (valid_data, valid_labels), (test_data, test_labels) = get_dataset_load_func(
        args.dataset)(args)

    # get dictionary file, for training L2E explainer in fairseq framework
    if not Path(args.dict_fairseq).exists():
        gen_dict(args, train_data, train_labels)

    # get train files
    gen_explanations(args, tokenizer, model, train_data, train_labels, split='train', train_data=train_data)

    # get validation files
    gen_explanations(args, tokenizer, model, valid_data, valid_labels, split='valid', train_data=train_data)

    # get test files
    gen_explanations(args, tokenizer, model, test_data, test_labels, split='test', train_data=train_data)


if __name__ == "__main__":
    main()