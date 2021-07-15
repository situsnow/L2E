
import itertools
import os

import torch

from fairseq.fairseq import utils
from fairseq.fairseq.data import (
    ConcatDataset,
    Dictionary,
    indexed_dataset,
    ExplanationPairDataset,
    ExplanationPureWeightDataset
)
from fairseq.fairseq.tasks import FairseqTask, register_task

CATEGORICAL_TGT_PADDING = -100
CONTINUOUS_TGT_PADDING = 0.0
MEMORY = 'mem'
SIM = 'sim'
RANDOM = 'random'


def load_explpair_dataset(
        data_path, split,
        src_prefix, src_suffix, src_dict, tgt_prefix, tgt_suffix, tgt_dict,
        combine, dataset_impl, upsample_primary,
        left_pad_source, left_pad_target, max_source_positions, remove_eos_from_source, tgt_padding):

    def split_exists(split, prefix, suffix, data_path):
        filename = os.path.join(data_path, '{}{}{}'.format(prefix, split, suffix))   # e.g., train_src.exp
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        if split_exists(split_k, src_prefix, src_suffix, data_path) and \
                split_exists(split_k, tgt_prefix, tgt_suffix, data_path):
            prefix = os.path.join(data_path, '{}{}{}'.format(src_prefix, split_k, src_suffix))

            src = indexed_dataset.make_dataset(prefix, impl=dataset_impl, dictionary=src_dict,
                                               append_eos=not remove_eos_from_source)
            src_datasets.append(src)

            prefix = os.path.join(data_path, '{}{}{}'.format(tgt_prefix, split_k, tgt_suffix))
            tgt_datasets.append(ExplanationPureWeightDataset(prefix, tgt_padding,
                                                             append_eos=not remove_eos_from_source))
        elif k > 0:
            break
        else:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        print('| {} {} {}-{} {} examples'.format(data_path, split_k,
                                                 '{}{}{}'.format(src_prefix, split, src_suffix),
                                                 '{}{}{}'.format(tgt_prefix, split, tgt_suffix), len(src_datasets[-1])))

        if not combine:
            break
    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        # if we need to upsample the src to generate more versions of masking neutral weights in calculating loss
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    return ExplanationPairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dict=tgt_dict, left_pad_source=left_pad_source,
        left_pad_target=left_pad_target, max_source_positions=max_source_positions,
        remove_eos_from_source=remove_eos_from_source
    )


def load_sim_encoder(device_id, encoder_path):
    # load bert for similarity measure
    from transformers import BertModel, BertTokenizer
    # vectorize src according to BERT pre-training models
    pretrained_bert = BertModel.from_pretrained(encoder_path)
    bert_tokenizer = BertTokenizer.from_pretrained(encoder_path)
    if torch.cuda.is_available():
        pretrained_bert = pretrained_bert.to(device_id)
    return bert_tokenizer, pretrained_bert


@register_task('explanation')
class ExplanationTask(FairseqTask):
    """
    Generate the weight of each word from the document to explain their importance in the document classification task.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')

        parser.add_argument('--tokens-per-sample', default=1024, type=int,
                            help='max number of tokens per sample for dataset')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')

        parser.add_argument('--add-bos-token', action='store_true',
                            help='prepend beginning of sentence token (<s>)')

        # copy from translation
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--left-pad-source', default=False, action='store_true',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default=False, action='store_true',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')

        parser.add_argument('--remove-eos-from-source', default=False, action='store_true',
                            help='remove the eos from source sentence, default is False')

        parser.add_argument('--input-prefix', default='', help='the prefix for the input file')
        parser.add_argument('--output-prefix', default='', help='the prefix for the output file')

        parser.add_argument('--input-suffix', default='_src.exp', help='the suffix for the input file')
        parser.add_argument('--output-suffix', default='_tgt.exp', help='the suffix for the output file')

        # Added by Snow 17 Mar 2021
        parser.add_argument('--dict-file', type=str, default="dict.txt", help='the vocabulary file')
        # Added by Snow 17 Mar 2021
        # fmt: on

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        from copy import deepcopy
        out_dict = deepcopy(dictionary)
        self.tgt_padding = CATEGORICAL_TGT_PADDING \
            if args.criterion != 'explanation_mse_loss' else CONTINUOUS_TGT_PADDING
        out_dict.pad_index = self.tgt_padding
        self.output_dictionary = out_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        dictionary = None
        # output_dictionary = None
        if args.data:
            paths = args.data.split(':')
            assert len(paths) > 0
            dictionary = Dictionary.load(os.path.join(paths[0], args.dict_file))
            # print('| dictionary: {} types'.format(len(dictionary)))

        return cls(args, dictionary)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        self.datasets[split] = load_explpair_dataset(
            data_path, split, self.args.input_prefix, self.args.input_suffix, self.source_dictionary,
            self.args.output_prefix, self.args.output_suffix, self.output_dictionary,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary, left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target, max_source_positions=self.args.max_source_positions,
            remove_eos_from_source=self.args.remove_eos_from_source, tgt_padding=self.tgt_padding)


    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary
