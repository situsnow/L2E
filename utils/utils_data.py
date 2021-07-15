import os
from utils.constants import IMDB_R, AGNEWS, SST, COLA, POS, NEG
from utils.util_functions import *


def get_dataset_categories(dataset):
    dataset_categories = dict()
    dataset_categories[IMDB_R] = ["neg", "neu", "pos"]
    dataset_categories[AGNEWS] = ["World", "Sports", "Business", "Sci/Tech"]
    dataset_categories[SST] = ['neg', 'pos']  # merge very.negative to negative, and very.positive to positive
    dataset_categories[COLA] = ['unacceptable', 'acceptable']

    return dataset_categories[dataset]


def get_bbox_by_dataset(data_dir, dataset):
    file_path = data_dir + '/' + dataset + '/'
    files_dict = dict()

    files_dict[SST] = file_path + 'distilbert_models/'
    files_dict[COLA] = file_path + 'albert_base_models/'
    files_dict[AGNEWS] = file_path + 'bert_base_models/'
    files_dict[IMDB_R] = get_imdb_r_latest_checkpoint(file_path)
    return files_dict[dataset]


def get_imdb_r_latest_checkpoint(file_path):
    from pathlib import Path
    max_chkpt = 0
    if os.path.exists(Path(file_path + '/fine_tune_longformer')):
        for file in Path(file_path + 'fine_tune_longformer').iterdir():
            if 'checkpoint' in os.path.basename(file):
                file_name = os.path.basename(file)
                max_chkpt = max(max_chkpt, int(file_name[file_name.index("-")+1:]))
    else:
        return file_path + '/pretrained_longformer'

    return file_path + '/fine_tune_longformer/checkpoint-' + str(max_chkpt)


def data_files(data_dir, dataset):

    file_path = data_dir + '/' + dataset + '/'
    files_dict = dict()

    files_dict[AGNEWS] = [file_path + 'train.txt', file_path + 'dev.txt', file_path + 'test.txt']
    files_dict[SST] = [file_path+'sst_train.txt', file_path+'sst_dev.txt', file_path+'sst_test.txt']
    files_dict[COLA] = [file_path+'cola_public/raw/'+'in_domain_train.tsv',
                        file_path+'cola_public/raw/'+'in_domain_dev.tsv',
                        file_path+'cola_public/raw/'+'out_of_domain_dev.tsv']
    files_dict[IMDB_R] = [file_path+'/l2e/train', file_path +'l2e/dev', file_path+'l2e/test']

    return files_dict[dataset]


def get_dataset_load_func(dataset):
    load_func = dict()
    load_func[IMDB_R] = load_imdbr
    load_func[AGNEWS] = load_agnews
    load_func[SST] = load_sst
    load_func[COLA] = load_cola

    return load_func[dataset]


def load_imdbr(args, clean=True):
    import re
    # args.file_path: withRats_neg, withRats_pos, noRats_neg, noRats_pos
    # will only have train/test set
    train_x, train_y, test_x, test_y = [], [], [], []
    train_indices, test_indices = [], []  # so that we can retrieve the original document
    train_rationales = []

    def read_directory(filepath, cat, filter=False):
        tag_s = "<%s>" % cat
        tag_e = "</%s>" % cat

        rationale_pattern = "%s (.*?) %s" % (tag_s, tag_e)
        all_files_content, all_files_indices, all_rationales = [], [], []
        for filename in os.listdir(filepath):
            file_index = int(filename[filename.index("_")+1:filename.index(".txt")])
            if filter and file_index < 900:
                continue

            content = read_plain_file(os.path.join(filepath, filename))[0]   # one instance per file
            rationales = re.findall(rationale_pattern, content)

            if rationales is not None:
                rationales = [" ".join(clean_data(each)) for each in rationales]
                all_rationales.append(rationales)  # list of list

            content = content.replace(tag_s, "").replace(tag_e, "")

            if clean:
                content = " ".join(clean_data(content))
            all_files_content.append(content)
            all_files_indices.append(os.path.basename(filename))
        return all_files_content, all_files_indices, all_rationales

    withRats_neg, withRats_neg_indices, withRats_neg_rationales = read_directory(args.file_paths[0], NEG.upper())
    train_x.extend(withRats_neg)
    train_y.extend([0 for _ in range(len(withRats_neg))])
    train_indices.extend(withRats_neg_indices)
    train_rationales.extend(withRats_neg_rationales)

    withRats_pos, withRats_pos_indices, withRats_pos_rationales = read_directory(args.file_paths[1], POS.upper())
    train_x.extend(withRats_pos)
    train_y.extend([1 for _ in range(len(withRats_pos))])
    train_indices.extend(withRats_pos_indices)
    train_rationales.extend(withRats_pos_rationales)

    noRats_neg, noRats_neg_indices, _ = read_directory(args.file_paths[2], NEG.upper(), filter=True)
    test_x.extend(noRats_neg)
    test_y.extend([0 for _ in range(len(noRats_neg))])
    test_indices.extend(["neg" + str(each) for each in noRats_neg_indices])

    noRats_pos, noRats_pos_indices, _ = read_directory(args.file_paths[3], POS.upper(), filter=True)
    test_x.extend(noRats_pos)
    test_y.extend(([1 for _ in range(len(noRats_pos))]))
    test_indices.extend(["pos" + str(each) for each in noRats_pos_indices])

    return (train_x, train_y, train_indices), train_rationales, (test_x, test_y, test_indices)


def load_agnews(args, clean=True):
    return get_nlp_datasets(args, 'ag_news', clean)


def load_sst(args, clean=True):
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = [], [], [], [], [], []

    def split_data(lines, text_list, label_list):
        for line in lines:
            data = line.split('\t')
            # ignore class 3 (neutral) and merge very.negative to negative, and very.positive to positive
            label = int(data[0][-1:])
            if label < 3:
                label = 0   # neg
            elif label > 3:
                label = 1   # pos
            else:
                continue
            label_list.append(label)
            text_list.append(clean_data(data[1]) if clean else data[1])

    for i, filename in enumerate(args.file_paths):
        all_data = read_plain_file(filename)
        if i == 0:
            # load train data
            split_data(all_data, train_data, train_labels)
        elif i == 1:
            # load dev data
            split_data(all_data, valid_data, valid_labels)
        else:
            split_data(all_data, test_data, test_labels)

    return (train_data, train_labels), (valid_data, valid_labels), (test_data, test_labels)


def load_cola(args, clean=True):
    import pandas as pd
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = [], [], [], [], [], []

    for i, filename in enumerate(args.file_paths):

        df = pd.read_csv(filename, delimiter='\t', header=None,
                         names=['sentence_source', 'label', 'label_notes', 'sentence'])
        if i == 0:
            train_labels = df.label.values.tolist()
            if clean:
                train_data = [clean_data(each, args.lower) for each in df.sentence.values]
            else:
                train_data = df.sentence.values
        elif i == 1:
            valid_labels = df.label.values.tolist()
            if clean:
                valid_data = [clean_data(each, args.lower) for each in df.sentence.values]
            else:
                valid_data = df.sentence.values
        else:
            test_labels = df.label.values.tolist()
            if clean:
                test_data = [clean_data(each, args.lower) for each in df.sentence.values]
            else:
                test_data = df.sentence.values

    return (train_data, train_labels), (valid_data, valid_labels), (test_data, test_labels)


def get_nlp_datasets(args, dataset, clean):
    import random, os
    from nlp import load_dataset

    train_text, train_labels, dev_text, dev_labels, test_text, test_labels = [], [], [], [], [], []
    # check if files exist
    if (not os.path.exists(args.file_paths[0])) or \
            (not os.path.exists(args.file_paths[1])) or (not os.path.exists(args.file_paths[2])):
        ds = load_dataset(dataset)
        train_examples = list(ds['train'])
        test_examples = list(ds['test'])
        train_saved, dev_saved, test_saved = [], [], []

        def collect_text_label(examples, split):
            for each in examples:
                original_text = each['text']
                clean_text = clean_data(original_text)
                if len(clean_text) > 400:   # filter long documents (the max length allowed in Bert is 512)
                    continue
                label = each['label']
                if split == 'train':
                    if random.uniform(0,1) < args.valid_ratio:  # split 10% to dev set
                        dev_text.append(original_text)
                        dev_labels.append(label)
                        # dev_saved.append(str(label) + ":::" + ' '.join(text) + '\n')
                        dev_saved.append(str(label) + ":::" + original_text + '\n')
                    else:
                        train_text.append(original_text)
                        train_labels.append(label)
                        # train_saved.append(str(label) + ":::" + ' '.join(text) + '\n')
                        train_saved.append(str(label) + ":::" + original_text + '\n')
                else:
                    test_text.append(original_text)
                    test_labels.append(label)
                    # test_saved.append(str(label) + ":::" + ' '.join(text) + '\n')
                    test_saved.append(str(label) + ":::" + original_text + '\n')

        collect_text_label(train_examples, 'train')
        collect_text_label(test_examples, 'test')

        # save files
        write_plain_file(args.file_paths[0], train_saved)
        write_plain_file(args.file_paths[1], dev_saved)
        write_plain_file(args.file_paths[2], test_saved)
    else:
        # load from files
        train = read_plain_file(args.file_paths[0])
        dev = read_plain_file(args.file_paths[1])
        test = read_plain_file(args.file_paths[2])

        def split_text_label(examples, split):
            for each in examples:
                # text = each.strip().split(':::')[1].split(" ")
                text = clean_data(each.strip().split(':::')[1]) if clean else each.strip().split(':::')[1]
                label = int(each.split(":::")[0])
                if split == 'train':
                    train_text.append(text)
                    train_labels.append(label)
                elif split == 'dev':
                    dev_text.append(text)
                    dev_labels.append(label)
                else:
                    test_text.append(text)
                    test_labels.append(label)
        split_text_label(train, 'train')
        split_text_label(dev, 'dev')
        split_text_label(test, 'test')

    return (train_text, train_labels), (dev_text, dev_labels), (test_text, test_labels)


def load_IMDB_R_pure(args, clean=True):
    # args.file_path: withRats_neg, withRats_pos, noRats_neg, noRats_pos
    # will only have train/test set
    train_x, train_y = [], []
    train_indices = []  # so that we can retrieve the original document

    def read_directory(filepath):

        all_files_content, all_files_indices = [], []
        for filename in os.listdir(filepath):
            content = read_plain_file(os.path.join(filepath, filename))[0]  # one instance per file
            if clean:
                content = " ".join(clean_data(content))
            all_files_content.append(content)
            all_files_indices.append(os.path.basename(filename))
        return all_files_content, all_files_indices

    withRats_neg, withRats_neg_indices = read_directory(args.file_paths[0])
    train_x.extend(withRats_neg)
    train_y.extend([0 for _ in range(len(withRats_neg))])
    train_indices.extend(withRats_neg_indices)

    withRats_pos, withRats_pos_indices = read_directory(args.file_paths[1])
    train_x.extend(withRats_pos)
    train_y.extend([1 for _ in range(len(withRats_pos))])
    train_indices.extend(withRats_pos_indices)

    return train_x, train_y, train_indices