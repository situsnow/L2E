# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset


def collate(
        samples, src_dict, tgt_dict, left_pad_source=True, left_pad_target=False,
        max_tokens=-1
):
    if len(samples) == 0:
        return {}

    def merge(key, dict, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            dict.pad(), src_dict.eos(), left_pad, move_eos_to_beginning, max_tokens=max_tokens
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', dict=src_dict, left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', dict=tgt_dict, left_pad=left_pad_target)
        target = target.index_select(0, sort_order)

    ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'src_text': None
        },
        'target': target,
    }
    # if prev_output_tokens is not None:
    #     batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class ExplanationPairDataset(FairseqDataset):
    """
    A torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None,  tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024,  # max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False
    ):

        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        # self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        return {
            'id': index,
            'source': src_item,   # this includes unique words only
            'original_source': None,
            'target': tgt_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples, max_tokens=-1):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            max_tokens: the max number of tokens in case we want to fix the size, default -1, no max tokens applies

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        # return collate(
        #     samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
        #     left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
        #     input_feeding=self.input_feeding,
        # )
        return collate(
            samples, src_dict=self.src_dict, tgt_dict=self.tgt_dict,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target, max_tokens=max_tokens
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        # return self.src_sizes[index], self.src_sizes[index]
        return self.src_sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        # if self.tgt_sizes is not None:
        #     indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)


class ExplanationPureWeightDataset:
    def __init__(self, path, tgt_padding, append_eos=True):
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        # to load data from file
        self.read_data(path, tgt_padding)
        self.size = len(self.lines)
        self.sizes = np.array(self.sizes)

    def read_tgt(self, idx, line, tgt_padding):

        line = torch.FloatTensor(line)
        # append the padding for label tag (front/back) and eos to match the same length as the input
        padding_tensor = torch.FloatTensor([tgt_padding])
        if self.append_eos:
            line = torch.cat((padding_tensor, line, padding_tensor, padding_tensor))
        else:
            line = torch.cat((padding_tensor, line, padding_tensor))
        self.lines.append(line)
        self.sizes.append(len(line))

    def read_data(self, path, tgt_padding):
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                line = [float(each) for each in line.strip('\n').split(",")]
                self.read_tgt(i, line, tgt_padding)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    def __getitem__(self, index):
        self.check_index(index)
        return self.lines[index]

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]
