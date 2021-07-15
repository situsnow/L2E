import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from utils.util_functions import send_to_cuda, split_sentence, join_sentence
from utils.constants import ALBERT, DISTILBERT, LONGFORMER


def load_blackbox(blackbox_path, lrp=False):
    tokenizer = AutoTokenizer.from_pretrained(blackbox_path)
    # default option, for SST/AGNews -Distilbert/Bert - occlusion/gradient/lime/kernel_shap/deep_shap
    classifier = AutoModelForSequenceClassification

    if ALBERT.lower() in blackbox_path and not lrp:
        # CoLA - Albert - occlusion/gradient/lime/kernel_shap/deep_shap
        from transformers import AlbertForSequenceClassification
        classifier = AlbertForSequenceClassification

    elif ALBERT.lower() in blackbox_path.lower() and lrp:
        # CoLA - Albert - lrp
        from models.huggingface_albert import MyAlbertForSequenceClassification
        classifier = MyAlbertForSequenceClassification

    elif DISTILBERT.lower() in blackbox_path.lower() and lrp:
        # SST - Distilbert - lrp
        from models.huggingface_distilbert import MyDistilBertForSequenceClassification
        classifier = MyDistilBertForSequenceClassification

    elif LONGFORMER.lower() not in blackbox_path and lrp:
        # AGNews - Bert - lrp
        from models.huggingface_bert import MyBertForSequenceClassification
        classifier = MyBertForSequenceClassification

    elif LONGFORMER.lower() in blackbox_path:
        # IMDB-R - Longformer - occlusion/gradient/lrp/lime/kernel_shap/deep_shap
        from models.huggingface_longformer import MyLongformerForSequenceClassification
        classifier = MyLongformerForSequenceClassification

    model = classifier.from_pretrained(blackbox_path)
    return tokenizer, send_to_cuda(model)


def prediction_by_bert_bulk(tokenizer, model, src=None, input_ids=None):
    import torch.nn.functional as F
    if src is not None:
        input_ids = convert_src_to_input_ids(src, tokenizer)
    if input_ids is not None and len(input_ids.shape) == 3:
        # the input_ids are embeddings
        outputs = F.log_softmax(model(inputs_embeds=input_ids)[0], dim=1)
    else:
        # for normal input_ids (after tokenizer)
        outputs = F.log_softmax(model(input_ids)[0], dim=1)
    del input_ids   # free memory
    return outputs


def prediction_by_bert(tokenizer, bbox, src):
    import torch.nn.functional as F
    with torch.no_grad():
        input_id = send_to_cuda(torch.tensor([tokenizer.encode(src)]))
        y_hat = F.softmax(bbox(input_id)[0][0], dim=0).cpu().detach().numpy()
    return y_hat


def convert_src_to_input_ids(src, tokenizer):
    input_id = [tokenizer.encode(' '.join(x) if type(x) == list else x) for x in src]

    max_len = len(max(input_id, key=len))
    new_input_id = []
    for x in input_id:
        while len(x) < max_len:
            x.append(tokenizer.pad_token_id)
        new_input_id.append(x)
    del input_id

    new_input_id = send_to_cuda(torch.tensor(new_input_id))
    return new_input_id


def construct_weights_after_backward(weights, input_id, tokenizer, text):

    from transformers.tokenization_roberta import RobertaTokenizer
    from transformers.tokenization_roberta_fast import RobertaTokenizerFast
    from transformers.tokenization_distilbert import DistilBertTokenizer
    from transformers.tokenization_distilbert_fast import DistilBertTokenizerFast

    def is_roberta():
        return issubclass(type(tokenizer), (RobertaTokenizer, RobertaTokenizerFast))

    def is_distilbert():
        return issubclass(type(tokenizer), (DistilBertTokenizer, DistilBertTokenizerFast))

    def convert_ids_to_tokens():
        if hasattr(tokenizer, 'ids_to_tokens'):
            return [tokenizer.ids_to_tokens[input_id[0][idx].item()] for idx in range(len(input_id[0]))]
        else:
            tokens = [tokenizer.convert_ids_to_tokens(input_id[0][idx].item()) for idx in range(len(input_id[0]))]

            if is_distilbert():
                return tokens
            for i in range(1, len(tokens)-1):  # exclude [CLS] and [SEP]
                if is_roberta():
                    if tokens[i][0] != 'Ġ' and i != 1:
                        tokens[i] = '##' + tokens[i]
                    else:
                        tokens[i] = tokens[i].replace('Ġ', '')
                else:
                    if (tokens[i] == '▁') or (tokens[i][0] != '▁' and tokens[i-1] != '##▁'):
                        # if current token not start with _ and
                        # previous token is not '##▁' ('▁' originally, but we updated)
                        # then it belongs to previous token
                        tokens[i] = '##' + tokens[i]
                    else:
                        # otherwise, it is the start of a valid token
                        tokens[i] = tokens[i].replace('▁', '')
        return tokens

    tokens = convert_ids_to_tokens()
    if '##' not in tokens[1]:  # need to filter the case like 'ions ...' where ions considers as '-ions'
        valid_token_start = 1
    else:
        valid_token_start = 2
    new_weights = [weights[valid_token_start].item()]  # the token must be valid and does not start with ##
    for i in range(valid_token_start + 1, len(tokens)-1):  # exclude [CLS], the first valid token and [SEP]

        if '##' not in tokens[i]:
            # this is a valid token or start of a split token
            # append the weight to new_weights
            new_weights.append(weights[i].item())
        else:
            # this is part of a previous token
            new_weights[-1] += weights[i].item()
    try:
        text = split_sentence(text)
        assert len(new_weights) == len(text)
    except AssertionError:
        if len(new_weights) > len(text):
            new_weights = new_weights[:len(text)]   # exclude the padding
        else:
            # the text had been truncated, those tokens will have zero weight
            while len(new_weights) != len(text):
                new_weights.append(0.0)

    return new_weights


def evaluate_bert(dataset, label, model, tokenizer):
    acc = 0

    for i, each in enumerate(dataset):
        input_id = send_to_cuda(torch.tensor([tokenizer.encode(join_sentence(each), truncation=True)]))

        y = model(input_id)[0]

        if torch.argmax(y) == label[i]:
           acc += 1

        del input_id
        del y

    return acc / len(dataset)
