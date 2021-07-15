import random
import numpy as np
import torch

from sklearn.cluster import KMeans

from fairseq.fairseq import utils


def map_wordpiece_to_weight(tokens, weight):
    # len(weight) != tokens
    new_weight = []
    i = 0
    j = 0
    while i < len(tokens):
        if "##" not in tokens[i]:
            # it's not a segmented word piece or the start of a wordpiece
            new_weight.append(weight[j].item())
            j += 1
        else:
            # it's part of the wordpiece with previous one(s)
            new_weight.append(weight[max(0, j-1)].item())    # in case the start token of sentence is segmented
        i += 1
    return new_weight


def convert_to_bert_hidden_states(args, bert_tokenizer, pretrained_bert, x, tgt):
    with torch.no_grad():
        x = ' '.join(x.split(' ')[1:-1])    # exclude the front/back label tag
        input_ids = torch.tensor([bert_tokenizer.encode(x)])

        # find the wordpiece token and extend their weight
        tokens = [bert_tokenizer.ids_to_tokens[input_ids[0][idx].item()] for idx in range(1, len(input_ids[0])-1)]
        wordpiece_tgt = torch.tensor(map_wordpiece_to_weight(tokens, tgt[1:-1]))  # exclude the front/back label tag

        if torch.cuda.is_available():
            input_ids = input_ids.to(args.device_id)
        hidden_states, _ = pretrained_bert(input_ids)[-2:]

        del _
        hidden_dim = hidden_states.shape[2]

        # (hidden_states: [1:T + 2:H]; wordpiece_tgt: [T]) ==> [H]
        # exclude the sos/eos
        x_exp_emb = torch.sum(hidden_states.squeeze(0)[1:-1].cpu().detach().clone() *
                              wordpiece_tgt.unsqueeze(1).repeat(1, hidden_dim), dim=0)
        del hidden_states

        return x_exp_emb.numpy()


def cluster_memory(args, tokenizer, encoder, memory, candidates):

    def slicing_memory_by_candidates(obj):
        return [obj[i] for i in candidates]

    converted_memory = [convert_to_bert_hidden_states(args, tokenizer, encoder, x, tgt)
                        for x, tgt in zip(slicing_memory_by_candidates(memory.src.lines),
                                          slicing_memory_by_candidates(memory.tgt.lines))]

    kmeans = KMeans(n_clusters=min(args.memory_sample_size, len(converted_memory)),
                    random_state=args.seed).fit(converted_memory)
    selected_mem = [np.random.choice(np.where(kmeans.labels_ == i)[0], 1).tolist()[0] for i in np.unique(kmeans.labels_)]

    return np.array(candidates)[selected_mem]


def sample_from_memory(args, task, max_tokens):

    # select select those shorter than the max_tokens
    candidates = np.where(task.dataset(args.memory_split).src_sizes <= max_tokens)[0].tolist()
    if args.memory_replay_strategy == 'random':
        selected_mem = random.sample(candidates, min(len(candidates), args.memory_sample_size))
    else:
        # to boost the training speed, randomly select some examples first
        candidates = random.sample(candidates, min(len(candidates), args.memory_sample_size * 5))
        # based on similarity with current mini-batch (train_samples)
        selected_mem = cluster_memory(args, task.bert_tokenizer, task.pretrained_bert, task.dataset(args.memory_split),
                                      candidates)
    mini_batch = [task.dataset(args.memory_split)[i] for i in selected_mem]

    mem_samples = task.dataset(args.memory_split).collater(mini_batch, max_tokens)
    return mem_samples


def concat_train_mem(train_samples, mem_samples):
    train_samples['id'] = torch.cat((train_samples['id'], mem_samples['id']), 0)
    train_samples['nsentences'] = train_samples['nsentences'] + mem_samples['nsentences']
    train_samples['ntokens'] = train_samples['ntokens'] + mem_samples['ntokens']
    # dict
    new_net_input = dict()
    new_net_input['src_tokens'] = torch.cat((train_samples['net_input']['src_tokens'],
                                             mem_samples['net_input']['src_tokens']), 0)
    new_net_input['src_lengths'] = torch.cat((train_samples['net_input']['src_lengths'],
                                              mem_samples['net_input']['src_lengths']), 0)
    new_net_input['src_text'] = None
    train_samples['net_input'] = new_net_input

    train_samples['target'] = torch.cat((train_samples['target'], mem_samples['target']), 0)

    if torch.cuda.is_available():
        train_samples = utils.move_to_cuda(train_samples)

    return [train_samples]