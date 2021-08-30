import torch
import numpy as np
from math import floor
import torch.nn.functional as F
from utils.constants import LIME, LRP, OCCLUSION, KERNEL_SHAP, DEEP_SHAP, GRADIENT
from utils.util_functions import split_sentence, join_sentence, send_to_cuda
from utils.utils_model import prediction_by_bert_bulk, convert_src_to_input_ids, construct_weights_after_backward


def get_teacher_explanations(original_explainer):

    teacher_algorithms = {LIME: get_lime_weights, LRP: get_lrp_weights, OCCLUSION: get_occlusion_weights,
                          KERNEL_SHAP: get_kernel_shap_weights, DEEP_SHAP: get_deep_shap_weights,
                          GRADIENT: get_gradient_weights}
    return teacher_algorithms[original_explainer]


def get_lime_weights(args, tokenizer, model, x, y, input_id, logits, train_split, index, train_data):
    from lime.lime_text import LimeTextExplainer

    def bert_predict(inputs):
        return get_output_as_mini_batch(args, tokenizer, model, src=inputs).cpu().detach().numpy()

    with torch.no_grad():
        num_features = len(split_sentence(x))
        lime_explainer = LimeTextExplainer(class_names=args.categories, random_state=args.random_state,
                                           mask_string=tokenizer.mask_token,
                                           bow=False)   # bow must set as False

        exp = lime_explainer.explain_instance(join_sentence(x), bert_predict, num_samples=args.sample_size,
                                              num_features=num_features, labels=[args.explain_label])

        dict_exp = dict(exp.as_list(label=args.explain_label))

        weights = []
        for each in split_sentence(x):
            try:
                w = dict_exp[each]
            except KeyError:
                w = 0.0
            weights.append(w)
        # normalize the weight
        lime_exp = list(map(lambda x: round(x, 8), normalize(weights, args.norm)))

    return lime_exp


def get_deep_shap_weights(args, tokenizer, bbox, x, y, input_id, logits, train_split=False, index=0, train_data=None):
    from shap.explainers.deep import DeepExplainer
    import torch.nn as nn

    class FakeModel(nn.Module):
        def __init__(self, tokenizer, bert_model):
            super(FakeModel, self).__init__()
            self.tokenizer = tokenizer
            self.bert_model = bert_model

        def forward(self, *input):
            input_ids = send_to_cuda(input[0])   # the input is a nested list with all samples now
            return get_output_as_mini_batch(args, self.tokenizer, self.bert_model, input_ids=input_ids)

        def embed(self, input):
            return bbox.base_model.embeddings.word_embeddings(input)

        def eval(self):
            self.bert_model.eval()
            return self

    x = split_sentence(x)
    np.random.seed(args.seed)
    if train_split:
        # in case we are generating the explanation of an instance training data,
        # make sure we don't use the same instance in the background
        p = [1 / (len(train_data) - 1) if i != index else 0.0 for i in range(len(train_data))]

        inds = np.random.choice(len(train_data), args.sample_size, replace=False, p=p)
    else:
        inds = np.random.choice(len(train_data), args.sample_size, replace=False)

    selected_references = np.array(train_data)[inds].tolist()
    selected_references.append(x)

    background = convert_src_to_input_ids(selected_references, tokenizer)

    test_example = background[-1].unsqueeze(0)
    background = background[:-1]

    fake_model = FakeModel(tokenizer, bbox)
    deepShapExplainer = DeepExplainer(fake_model, background)
    ''' Remember to change the shap interface as well
        Changed files: 
            shap.explainers.deep.__init__.py
            shap.explainers.deep.deep_pytorch.py
            
        Refer to the version in this repository:
        utils.deepshap_init.py
        utils.deepshap_pytorch.py        
    '''
    shap_values = deepShapExplainer.shap_values(test_example, get_max_output=args.explain_label,
                                                mini_batch=args.mini_batch)
    torch.cuda.empty_cache()
    shap_values = construct_weights_after_backward(shap_values[0], input_id, tokenizer, x)
    return [round(each, 8) for each in normalize(shap_values, args.norm)]


def get_kernel_shap_weights(args, tokenizer, bbox, x, y, logits, input_id, train_split, index, train_data):

    from shap.explainers.kernel import KernelExplainer

    x = split_sentence(x)

    def f(z):
        if np.all(z == 0):
            return np.array([0.0])
            # convert all z back to text
        all_samples = []
        for sample_index in range(z.shape[0]):
            s = z[sample_index].tolist()
            words = [x[w] if s[w] == 1 else tokenizer.mask_token for w in range(len(s))]

            all_samples.append(words)
        return get_output_as_mini_batch(args, tokenizer, bbox, src=all_samples).cpu().detach().numpy()[:, args.explain_label]

    with torch.no_grad():
        kernel_shap_explainer = KernelExplainer(f, np.zeros((1, len(x))))
        shap_values = kernel_shap_explainer.shap_values(np.ones((1, len(x))), nsamples=args.sample_size)

        shap_exp = list(map(lambda x: round(x, 8), normalize(shap_values[0].tolist(), args.norm)))
    return shap_exp


def get_lrp_weights(args, tokenizer, bbox, x, y, input_id, logits, train_split, index, train_data):

    bbox.train()

    lrp_out_mask = send_to_cuda(torch.zeros((input_id.shape[0], len(args.categories))))
    lrp_out_mask[:, args.explain_label] = 1.0
    relevance_score = logits * lrp_out_mask
    lrp_score = bbox.backward_lrp(relevance_score)
    lrp_score = lrp_score.cpu().detach().data
    lrp_score = torch.abs(lrp_score).sum(dim=-1)

    lrp_score = construct_weights_after_backward(lrp_score[0], input_id, tokenizer, x)

    lrp_score = list(map(lambda x: round(x, 8), lrp_score))

    del lrp_out_mask

    return lrp_score


def get_gradient_weights(args, tokenizer, bbox, x, y, input_id, logits, train_split, index, train_data):
    # use the dot product of input and gradient as the weight
    y = send_to_cuda(torch.tensor([y]))
    embeddings = send_to_cuda(bbox.base_model.embeddings.word_embeddings(input_id))
    output = bbox(inputs_embeds=embeddings)
    loss = F.nll_loss(output[0], y)
    gradient = torch.autograd.grad(loss, embeddings, retain_graph=True, create_graph=True)

    weights = torch.sum(torch.mul(embeddings.squeeze(0), gradient[0].squeeze(0)), dim=1)

    del embeddings
    del output
    del gradient

    new_weights = construct_weights_after_backward(weights, input_id, tokenizer, x)
    del weights
    return list(map(lambda x: round(x, 8), normalize(new_weights, args.norm)))


def get_occlusion_weights(args, tokenizer, bbox, x, y, input_id, logits, train_split, index, train_data):
    # use the same method as described in https://arxiv.org/pdf/1910.12336.pdf
    weights = []

    with torch.no_grad():
        y = send_to_cuda(torch.tensor([y]))
        loss_full = F.nll_loss(F.log_softmax(logits, dim=1), y)

        x = split_sentence(x)
        for i in range(len(x)):
            new_text = x[:i]
            new_text.append(tokenizer.mask_token)
            new_text.extend(x[i + 1:])

            input_id = send_to_cuda(torch.tensor([tokenizer.encode(join_sentence(new_text), truncation=True)]))
            y_hat = bbox(input_id)[0]

            loss_partial = F.nll_loss(F.log_softmax(y_hat, dim=1), y)
            weights.append((loss_partial - loss_full).cpu().detach().item())

            del input_id
            del y_hat

    return list(map(lambda x: round(x, 8), normalize(weights, args.norm)))


def get_output_as_mini_batch(args, tokenizer, model, src=None, input_ids=None):
    if src is not None and input_ids is not None:
        raise ValueError("You cannot specify both input_ids and src at the same time")
    flag = True if src is not None else False
    # split the inputs into mini-batch with size 200
    nsamples = len(src) if flag else len(input_ids)
    all_outputs = None
    i = 0

    defined_mini_batch = args.mini_batch
    while nsamples - defined_mini_batch * i > 0:
        try:
            if flag:
                mini_batch = src[defined_mini_batch * i:min(defined_mini_batch * (i + 1), nsamples)]
                output = prediction_by_bert_bulk(tokenizer, model, src=mini_batch)
                del mini_batch
            else:
                mini_batch = input_ids[defined_mini_batch * i:min(defined_mini_batch * (i + 1), nsamples)]

                output = prediction_by_bert_bulk(tokenizer, model, input_ids=mini_batch)
                del mini_batch
            if all_outputs is None:
                all_outputs = output
            else:
                all_outputs = torch.cat([all_outputs, output], dim=0)
            del output
            i += 1
        except RuntimeError:
            # reduce the mini batch size and try the same program again
            if defined_mini_batch >= 2:
                defined_mini_batch = floor(defined_mini_batch / 2)
                print(f'Adjust mini batch to {defined_mini_batch}')
                continue
            else:
                raise RuntimeError(f'Cannot process current long document (not even with mini-batch =1)')

    return all_outputs


def normalize(items, norm, a=-1, b=1):
    denom = 1.0  # no normalization by default
    if norm is not None:
        denom = sum(items)  # otherwise default denominator is sum
    if norm == 'max':
        denom = max(items)
    elif norm == 'sum_square':
        denom = sum([x ** 2 for x in items])
    elif norm == 'range':
        min_x = min(items)
        max_x = max(items)
        items = [(b - a) * ((x - min_x) / (max_x - min_x)) + a for x in items]
        return items
    elif norm == 'tanh':
        # convert items into tensor
        items = torch.tanh(torch.tensor(items))
        items = items.detach().cpu().tolist()
        return items

    items = [x / denom if denom != 0.0 else x for x in items]
    return items