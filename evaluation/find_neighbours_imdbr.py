import argparse, sys
from pathlib import Path
from nltk.corpus import stopwords

from utils.constants import *
from evaluation.constants import *
from utils.utils_data import load_imdbr

epsilon = 1e-200
STOPWORDS = stopwords.words('english')


def convert_to_n_gram(x):
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

    if min(len(a), len(b)) == 0:
        print('stop.')
    return sim / min(len(a), len(b))


# find the similar documents among test set
def main():
    max_len = 100
    nearest_neighbors = 10

    parser = argparse.ArgumentParser(description='Compare IMDB_R rationale Parser')
    parser.add_argument('--data-dir', type=str, help='path to data')
    args, _ = parser.parse_known_args(sys.argv)
    folder = args.data_dir + '/' + IMDB_R + '/original/'
    args.l2e_folder = args.data_dir + '/' + IMDB_R + '/l2e/'
    args.file_paths = [folder + 'withRats_neg', folder + 'withRats_pos',
                       folder + 'noRats_neg', folder + 'noRats_pos']
    args.longformer_acc_files = {TRAIN: args.l2e_folder + 'train_acc.txt',
                                 VALID: args.l2e_folder + 'valid_acc.txt',
                                 TEST: args.l2e_folder + 'test_acc.txt'}

    (_, _, train_indices), train_rationales, (_, _, _) = load_imdbr(args, clean=False)

    # filter incorrectly predicted instances
    train_acc = Path(args.longformer_acc_files[TRAIN]).read_text().strip().split(",")
    valid_acc = Path(args.longformer_acc_files[VALID]).read_text().strip().split(",")
    test_acc = Path(args.longformer_acc_files[TEST]).read_text().strip().split(",")

    acc = train_acc + valid_acc + test_acc

    filter_train_indices, filter_train_rationales = [], []
    for idx in range(len(train_indices)):
        if train_indices[idx] in acc:
            filter_train_indices.append(train_indices[idx])
            filter_train_rationales.append(train_rationales[idx])

    # convert all rationales to n-gram (1-4)
    merge_rationales = []
    for i, x_rationales in enumerate(filter_train_rationales):
        all_ngram = []
        for r in x_rationales:
            extracted_ngram = convert_to_n_gram(r)
            for idx, ngram in enumerate(extracted_ngram):
                if len(all_ngram) <= idx:
                    all_ngram.append(ngram)
                else:
                    all_ngram[idx].extend(ngram)
        if len(all_ngram) == 0:
            print(filter_train_indices[i])
            continue
        merge_rationales.append(all_ngram)

    similarity = {}
    # loop each x to find neighbours
    for i, i_rationales in enumerate(merge_rationales):
        i_similarity = {}
        for j, j_rationales in enumerate(merge_rationales):
            if i == j or \
                    (NEG in filter_train_indices[i] and POS in filter_train_indices[j]) or \
                    (POS in filter_train_indices[i] and NEG in filter_train_indices[j]):   # two neighbors has to be have same label
                continue
            score = n_gram_sim(i_rationales, j_rationales)
            i_similarity[filter_train_indices[j]] = score

        sorted_neighbours = sorted(i_similarity.items(), key=lambda x: x[1], reverse=True)
        similarity[filter_train_indices[i]] = [k for k, v in sorted_neighbours[:nearest_neighbors]]

    with open(args.data_dir + '/' + IMDB_R + '/l2e/similarity.txt', 'w') as f:
        for kv in similarity.items():
            f.write(kv[0] + ":::" + ",".join(kv[1]))
            f.write("\n")


if __name__ == "__main__":
    main()