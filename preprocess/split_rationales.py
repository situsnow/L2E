
# mask all the rationales in an example of dataset IMDB-R so that we can generate a new example

import argparse, sys, os
from utils.constants import MASKING, IMDB_R
from utils.utils_data import data_files, load_IMDB_R_pure, clean_data
from utils.util_functions import write_plain_file, convert_to_n_gram, n_gram_sim
import random


def gen_mask_rationale_example(example, tag_s, tag_e):
    new_example = ''
    while len(example) > 0:
        if example[0:5] == tag_s:
            # clean data and replace with MASKING
            rationale = clean_data(example[6:example.find(tag_e)])  # exclude the tag
            # the longformer use <mask> as the mask token
            mask_rationale = " ".join([MASKING] * len(rationale))
            new_example += " " + mask_rationale if len(new_example) != 0 else mask_rationale
            new_start_pos = example.find(tag_e)
            example = example[new_start_pos + 6:]
        else:
            # clean data
            end_pos = example.find(tag_s)
            non_rantionale = clean_data(example[:len(example) if end_pos == -1 else end_pos])
            non_rantionale = " ".join(non_rantionale)
            new_example += " " + non_rantionale if len(new_example) != 0 else non_rantionale
            example = "" if end_pos == -1 else example[end_pos:]
    return new_example


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test IMDB rationales')
    parser.add_argument('--data', type=str, help='the path to the data')
    parser.add_argument('--dataset', type=str, default=IMDB_R)

    args, _ = parser.parse_known_args(sys.argv)

    args.file_paths = data_files(args.data, args.dataset)

    train_x, train_y, train_indices = load_IMDB_R_pure(args, clean=False)

    random.seed(1234)
    num_of_i = 50  # randomly select 50 instances: 25 pos, 25 neg
    num_of_neighbor = 3  # for each instance, find nearest num_of_neighbor in same category (pos/neg)
    new_cat = 'neu'

    pos_count, neg_count = 0, 0   # ensure we select only 25 pos and 25 neg
    count = num_of_i
    while count > 0:
        i = random.randint(0, len(train_x))
        if (train_y[i] == 0 and neg_count > num_of_i / 2) or (train_y[i] == 1 and pos_count > num_of_i / 2):
            continue

        x = train_x[i]
        y = train_y[i]

        cat = 'neg' if y == 0 else 'pos'
        tag_s = "<%s>" % cat.upper()
        tag_e = "</%s>" % cat.upper()

        # remove the tags and clean the data
        x = x.replace(tag_s, "").replace(tag_e, "")
        x = " ".join(clean_data(x))
        x_converted = convert_to_n_gram(None, x)

        # find num_of_neighbors according to n-gram similarity
        x_neighbours = list(
            filter(None, [nn_idx if nn == y and nn_idx != i else None for nn_idx, nn in enumerate(train_y)]))
        similarity = {}
        for nn in x_neighbours:
            neighbor = train_x[nn]
            neighbor = neighbor.replace(tag_s, "").replace(tag_e, "")
            neighbor = " ".join(clean_data(neighbor))
            nn_converted = convert_to_n_gram(None, neighbor)
            similarity[nn] = n_gram_sim(x_converted, nn_converted)

        sorted_similarity = dict(sorted(similarity.items(), key=lambda kv: kv[1], reverse=True))
        nn_idx = list(sorted_similarity.keys())[:num_of_neighbor]

        print("Current document: %s" % train_indices[i])
        print("Top %d documents are: %s" % (num_of_neighbor, ",".join([str(train_indices[idx]) for idx in nn_idx])))
        print("=======================")

        # cp current instance and neighbor files to folder l2e/test
        for index in [i] + nn_idx:
            example = train_x[index]
            example = example.replace(tag_s, "").replace(tag_e, "")
            example = " ".join(clean_data(example))

            new_file = '%s/%s/l2e/test/%s.txt' % (args.data, args.dataset, train_indices[index])
            write_plain_file(new_file, example + '\n')

            # we don't need the mask file for test set
            # new_file = '%s/%s/l2e/test/%s%s_masked.txt' % (args.data, args.dataset, new_cat, train_indices[index][3:])
            # example = train_x[index]
            # mask_rationale_example = gen_mask_rationale_example(example, tag_s, tag_e)
            # write_plain_file(new_file, mask_rationale_example)

        # remove i and nn_idx from train_x/train_y/train_indices
        for index in sorted([i] + nn_idx, reverse=True):
            del train_x[index]
            del train_y[index]
            del train_indices[index]

        count -= 1

    # cp rest of the file to folder
    for i in range(len(train_x)):
        x = train_x[i]
        y = train_y[i]

        cat = 'neg' if y == 0 else 'pos'
        tag_s = "<%s>" % cat.upper()
        tag_e = "</%s>" % cat.upper()
        x = x.replace(tag_s, "").replace(tag_e, "")
        x = " ".join(clean_data(x))

        new_file = "%s/%s/l2e/train/%s.txt" % (args.data, args.dataset, train_indices[i])
        write_plain_file(new_file, x + '\n')

        example = train_x[i]
        mask_rationale_example = gen_mask_rationale_example(example, tag_s, tag_e)
        # in case the index number overlaps between pos and neg, we append the original cat in the new file name
        new_file = "%s/%s/l2e/train/%s_%s.txt" % (args.data, args.dataset, new_cat, train_indices[i])
        write_plain_file(new_file, mask_rationale_example + '\n')
