import math, warnings
import torch
import torch.nn.functional as F
import numpy as np

from fairseq.fairseq import utils

from . import FairseqCriterion, register_criterion


INTERVAL = 'interval'
CATEGORIZE = 'categorize'
INTE_QUANTILE = 'inte_quantile'
INDI_QUANTILE = 'indi_quantile'
LRP = 'lrp'
GROUND_TRUTH = 'ground_truth'


@register_criterion('explanation_cross_entropy')
class ExplanationCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.discretize_method = args.discretize_method
        self.ranking_label_types = args.ranking_label_types
        if self.discretize_method == CATEGORIZE and not self.ranking_label_types == args.encoder_output_dim:
            raise RuntimeError('The --encoder-output-dim has to be aligned with the --ranking-label-types.')
        self.interval = args.interval

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--discretize-method', default=INTERVAL,
                            choices=[INTERVAL, CATEGORIZE, LRP, GROUND_TRUTH],
                            # PREFER_LABEL
                            help='the method to discretize the weights')
        parser.add_argument('--interval', default=5, type=int, metavar='N',
                            help='the N interval to discretize for each category.')
        parser.add_argument('--ranking-label-types', default=3, type=int, metavar='N',
                            help='discretize the ordering of the weights into N categories')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        # loss, metric_dict = self.compute_loss(model, net_output, sample)
        loss = self.compute_loss(model, net_output, sample)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample):
        lprobs = F.log_softmax(net_output.float(), dim=-1)   # T * B * C

        target = model.get_targets(sample, net_output)   # B * T

        # debug_info
        # reserve_target = torch.tensor(target.detach())

        bsz = target.shape[0]  # B
        wps = target.shape[1]  # T
        src_lengths = sample['net_input']['src_lengths']
        if self.discretize_method is not None and self.discretize_method != GROUND_TRUTH:
            target = self.discrete_target(target, bsz, wps, src_lengths)

        _, pred = torch.max(lprobs.transpose(1, 0), 2)  # B * T

        if lprobs.size()[0] != target.size()[1]:
            # the model is exp_longformer, we need to pad extra in target
            target = F.pad(target, pad=(0, lprobs.size()[0] - target.size()[1], 0, 0), mode='constant', value=self.padding_idx)

        target = target.type(torch.LongTensor)
        if torch.cuda.is_available():
            target = target.cuda()
        # lprobs = lprobs.view(-1, lprobs.size(-1))  # T * B * C ==> (T * B) * C
        lprobs = lprobs.transpose(1, 0).reshape(-1, lprobs.size(-1))  # B * T * C ==> (B*T) * C

        target = target.reshape(-1)  # B * T ==> (B*T)

        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='none',
        )

        return torch.sum(loss)

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2) if sample_size != 0 else 0.0,
            'ntokens': ntokens if isinstance(ntokens, int) else ntokens.item(),
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            ntokens = ntokens if isinstance(ntokens, int) else ntokens.item()
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output

    def discrete_target(self, target, bsz, wps, src_lengths):
        # convert the target according to the hyper-parameters
        def sort_taget_weight(i, n):
            return target[i][n]
        discretized_target = []

        # discretize the target for each example, exclude the padding index
        # 0: pos; 1: neutral; 2: neg
        for i in range(bsz):
            if self.discretize_method == CATEGORIZE and self.ranking_label_types == 3:
                discretized_target.append([0 if w > 0 else 1 if w == 0 else int(w) if w == self.padding_idx
                                          else 2 for w in target[i]])
            elif self.discretize_method == LRP:
                # 0: high pos; 1: medium pos; 2: low pos
                # only for LRP
                copy_target = np.array(target[i][:].cpu())
                miu = np.mean(copy_target[1:src_lengths[i]-1])
                sigma = np.std(copy_target[1:src_lengths[i]-1])
                high_pos = miu + sigma
                low_pos = miu - sigma
                discretized_target.append([int(w) if w == self.padding_idx
                                           else 0 if w >= high_pos else 1 if low_pos <= w < high_pos else 2
                                           for w in target[i]])

            elif self.discretize_method == CATEGORIZE and self.ranking_label_types == 5:
                y = target[i]
                T = src_lengths[i]  # includes the front/back label tag
                # 0: high neg; 1: low neg; 2: neutral; 3: low pos; 4: high pos
                all_pos = list(filter(None, [w if w > 0.0 else None for w in y[1:T-1]]))
                all_neg = list(filter(None, [w if w < 0.0 else None for w in y[1:T-1]]))
                mean_pos = sum(all_pos) / len(all_pos) if len(all_pos) > 0 else 0.0
                mean_neg = sum(all_neg) / len(all_neg) if len(all_neg) > 0 else 0.0
                discretized_target.append([2 if y[j] == 0.0 or j == 0 or j == T - 1
                                           else int(y[j]) if int(y[j]) == self.padding_idx
                                           else 0 if y[j] < mean_neg
                                           else 1 if mean_neg <= y[j] < 0.0 else 3 if 0.0 < y[j] <= mean_pos else 4
                                           for j in range(len(y))])

            elif self.discretize_method == INTERVAL:
                # identify the start/end of the sentence
                # + 1 to make sure the index correct
                # stop can be also like this: target[i][1:].tolist().index(self.padding_idx) + 1
                start, stop = 1, src_lengths[i] - 1
                sorted_target = sorted(range(start, stop),
                                       key=lambda n: sort_taget_weight(i, n), reverse=True)

                discretized_target.append([1 if target[i][j] == 0.0 or j == 0 or j == src_lengths[i] - 1
                                           else int(target[i][j]) if target[i][j] == self.padding_idx
                                           else 0 if j in sorted_target[:self.interval] and target[i][j] > 0
                                           else 2 if j in sorted_target[-self.interval:] and target[i][j] < 0 else 1
                                           for j in range(wps)])

        discretized_target = torch.tensor(discretized_target)
        if torch.cuda.is_available():
            discretized_target = discretized_target.cuda()
        return discretized_target
