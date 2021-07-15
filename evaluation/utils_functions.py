import numpy as np
import os, sys, torch
from scipy import stats

from utils.util_functions import join_sentence, send_to_cuda


def read_file(path, delimiter):
    result = []
    with open(path) as f:
        for each in f:
            each = each.strip().split(delimiter)
            if delimiter == ",":
                each = [float(x) for x in each]
            result.append(each)
    return np.array(result)


def load_fairseq_args(explainer_parameters):

    from fairseq.fairseq import options, utils

    load_explainer_parameters(explainer_parameters)
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    return args


def load_explainer_parameters(configuration_path):
    sys.argv = ""

    with open(configuration_path, 'r') as config:
        arguments = []
        for line in config:
            arguments.extend(line.split())
        sys.argv = arguments


def load_fairseq_trainer(args, checkpoint_path):
    from fairseq.fairseq import checkpoint_utils, tasks
    from fairseq.fairseq.trainer import Trainer
    task = tasks.setup_task(args)

    model = task.build_model(args)
    criterion = task.build_criterion(args)
    trainer = Trainer(args, task, model, criterion)

    if os.path.exists(checkpoint_path):
        state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)
        trainer.get_model().load_state_dict(state['model'], strict=True)

    return trainer


def load_explainer_checkpoint(explainer_parameters, checkpoint_path):

    args = load_fairseq_args(explainer_parameters)

    trainer = load_fairseq_trainer(args, checkpoint_path)

    return trainer


def get_weights_from_explainer(trainer, test_example, output_dim=3):
    # import torch.nn.functional as F
    # convert the test_example according to the dictionary
    tokens = trainer.task.dictionary.encode_line(
        join_sentence(test_example), add_if_not_exist=False,
        append_eos=False, reverse_order=False,
    ).long().unsqueeze(0)

    # make the dict for the model input
    input_dict = dict()
    input_dict['src_tokens'] = send_to_cuda(tokens)
    # tokens.cuda() if torch.cuda.is_available() else tokens.clone().detach()
    input_dict['src_lengths'] = torch.tensor([len(test_example)])
    input_dict['src_text'] = None

    output = trainer.get_model()(**input_dict)[1:-1]   # exclude the front/back label tag
    if output_dim > 1:
        _, pred = torch.max(output.transpose(1, 0), 2)  # B * T
    else:
        pred = output.squeeze(2).squeeze(1)
    # transformer_output = F.log_softmax(output.float(), dim=-1)
    return pred


def paired_sign_test(dist_one, dist_two, test='t'):
    # the p value is from a two-sided t test;
    # if it should be a one-sided t test, p needs to divide half in later stage
    if test == 't':
        _, p_value = stats.ttest_rel(dist_one, dist_two)
    else:
        # wilcoxon signed ranked test
        _, p_value = stats.wilcoxon(dist_one, dist_two)
    return p_value


def proportions_z_test(count, nobs):
    from statsmodels.stats.proportion import proportions_ztest
    count = np.array(count)
    nobs = np.array(nobs)
    _, p_value = proportions_ztest(count, nobs)
    return p_value


def mean_confidence_interval(data, confidence=0.95, round_decimal=2):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m, round_decimal), round(h, round_decimal)