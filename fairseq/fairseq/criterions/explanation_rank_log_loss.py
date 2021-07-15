import math
import torch

from fairseq.fairseq import utils

from . import FairseqCriterion, register_criterion

@register_criterion('explanation_rank_log_loss')
class ExplanationRankLogLoss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.fp16 = args.fp16

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # print("DEBUG INFO: before model()", flush=True)
        # print(sample, flush=True)
        # print("DEBUG info: before model() --> sample['net_input']", flush=True)
        # print(sample['net_input'], flush=True)

        net_output = model(**sample['net_input'])
        # print("DEBUG INFO: after model()", flush=True)
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
        # encoder_out = torch.exp(net_output)   # T * B * C
        encoder_out = net_output
        target = model.get_targets(sample, net_output).transpose(0, 1).unsqueeze(2)
        batch = target.shape[1]

        def compute_instance_loss(idx):
            # compute the pairwise loss according to their rank (target) at sample idx
            sent_len = sample['net_input']['src_lengths'][idx].item()
            y_hat = encoder_out[:, idx, :]   # T * C
            y = target[:, idx, :]
            base = torch.log(torch.tensor(0.5))
            loss = torch.tensor([0.0])
            if torch.cuda.is_available():
                base = base.cuda()
                loss = loss.cuda()
            if self.fp16:
                base = base.half()
                loss = loss.half()
            for i in range(sent_len):
                for j in range(i + 1, sent_len):
                    if i == 0 and j == sent_len - 1:
                        # ignore
                        continue
                    if i == 0 or abs(y[i]) < abs(y[j]):
                        # it's the label tag in front of sentence
                        loss += pairwise_loss(j, i, y_hat)
                    elif abs(y[i]) > abs(y[j]) or j == sent_len - 1:
                        # it's the label tag at the end of sentence
                        loss += pairwise_loss(i, j, y_hat)
                    else:
                        # the two true weights are equal, their output should be same
                        loss += pairwise_loss(i, j, y_hat) + base
            return loss / sent_len

        def pairwise_loss(higher_idx, lower_idx, y_hat):
            # the calculation below is the same is
            # -torch.log(torch.exp(y_hat[higher_idx]) / (torch.exp(y_hat[higher_idx]) + torch.exp(y_hat[lower_idx])))
            # but without loss overflow
            a = max(y_hat[higher_idx], y_hat[lower_idx])
            b = min(y_hat[higher_idx], y_hat[lower_idx])
            return - y_hat[higher_idx] + (a + torch.log(1 + torch.exp(b - a)))

        loss = sum(list(map(compute_instance_loss, range(batch))))

        return loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
