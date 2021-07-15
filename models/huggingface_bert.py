import collections, torch, copy, math
import torch.nn as nn
from transformers.modeling_bert import BertForSequenceClassification, BertModel, BertPooler, BertEncoder, BertLayer, \
    BertAttention, BertSelfAttention, BertSelfOutput, BertIntermediate, BertOutput
from models.lrp import backprop_lrp_fc, backprop_lrp_jacobian

# access global vars here
global func_inputs
global func_activations
func_inputs = collections.defaultdict(list)
func_activations = collections.defaultdict(list)


def get_inputivation(name):
    def hook(model, input, output):
        func_inputs[name] = [_in for _in in input]

    return hook


def get_activation(name):
    def hook(model, input, output):
        func_activations[name] = output

    return hook


def get_activation_multi(name):
    def hook(model, input, output):
        func_activations[name] = [_out for _out in output]

    return hook


def init_hooks_lrp(model):
    """
    Initialize all the hooks required for full lrp for BERT model.
    """
    # in order to backout all the lrp through layers
    # you need to register hooks here.

    model.classifier.register_forward_hook(
        get_inputivation('model.classifier'))
    model.classifier.register_forward_hook(
        get_activation('model.classifier'))
    model.bert.pooler.dense.register_forward_hook(
        get_inputivation('model.bert.pooler.dense'))
    model.bert.pooler.dense.register_forward_hook(
        get_activation('model.bert.pooler.dense'))
    model.bert.pooler.register_forward_hook(
        get_inputivation('model.bert.pooler'))
    model.bert.pooler.register_forward_hook(
        get_activation('model.bert.pooler'))

    model.bert.embeddings.word_embeddings.register_forward_hook(
        get_activation('model.bert.embeddings.word_embeddings'))
    model.bert.embeddings.register_forward_hook(
        get_activation('model.bert.embeddings'))

    layer_module_index = 0
    for module_layer in model.bert.encoder.layer:

        ## Encoder Output Layer
        layer_name_output_layernorm = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.output.LayerNorm'
        module_layer.output.LayerNorm.register_forward_hook(
            get_inputivation(layer_name_output_layernorm))

        layer_name_dense = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.output.dense'
        module_layer.output.dense.register_forward_hook(
            get_inputivation(layer_name_dense))
        module_layer.output.dense.register_forward_hook(
            get_activation(layer_name_dense))

        layer_name_output = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.output'
        module_layer.output.register_forward_hook(
            get_inputivation(layer_name_output))
        module_layer.output.register_forward_hook(
            get_activation(layer_name_output))

        ## Encoder Intermediate Layer
        layer_name_inter = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.intermediate.dense'
        module_layer.intermediate.dense.register_forward_hook(
            get_inputivation(layer_name_inter))
        module_layer.intermediate.dense.register_forward_hook(
            get_activation(layer_name_inter))

        layer_name_attn_layernorm = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.output.LayerNorm'
        module_layer.attention.output.LayerNorm.register_forward_hook(
            get_inputivation(layer_name_attn_layernorm))

        layer_name_attn = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.output.dense'
        module_layer.attention.output.dense.register_forward_hook(
            get_inputivation(layer_name_attn))
        module_layer.attention.output.dense.register_forward_hook(
            get_activation(layer_name_attn))

        layer_name_attn_output = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.output'
        module_layer.attention.output.register_forward_hook(
            get_inputivation(layer_name_attn_output))
        module_layer.attention.output.register_forward_hook(
            get_activation(layer_name_attn_output))

        layer_name_self = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.self'
        module_layer.attention.self.register_forward_hook(
            get_inputivation(layer_name_self))
        module_layer.attention.self.register_forward_hook(
            get_activation_multi(layer_name_self))

        layer_name_value = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.self.value'
        module_layer.attention.self.value.register_forward_hook(
            get_inputivation(layer_name_value))
        module_layer.attention.self.value.register_forward_hook(
            get_activation(layer_name_value))

        layer_name_query = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.self.query'
        module_layer.attention.self.query.register_forward_hook(
            get_inputivation(layer_name_query))
        module_layer.attention.self.query.register_forward_hook(
            get_activation(layer_name_query))

        layer_name_key = 'model.bert.encoder.' + str(layer_module_index) + \
                                '.attention.self.key'
        module_layer.attention.self.key.register_forward_hook(
            get_inputivation(layer_name_key))
        module_layer.attention.self.key.register_forward_hook(
            get_activation(layer_name_key))

        layer_module_index += 1


class MyBertSelfAttention(BertSelfAttention):
    def attention_core(self, query_layer, key_layer, value_layer, attention_mask):
        """
        This is the core self-attention layer.
        """
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        return attention_probs

    def jacobian(self, tensor_out, tensor_in, debug=False):
        """
        This is super slow. You can simply write out the full
        jacboian by hand, and it would be so much faster.
        PyTorch team is working on a fastor impl which is still
        in progress.
        """
        import time
        start = time.time()
        jacobian_full = []
        for i in range(tensor_out.shape[2]):
            jac_mask = torch.zeros_like(tensor_out)
            jac_mask[:, :, i] = 1.
            jacobian_partial = torch.autograd.grad(tensor_out, tensor_in,
                                                   grad_outputs=jac_mask,
                                                   retain_graph=True)[0]
            jacobian_full.append(jacobian_partial)
        jacobian_full = torch.stack(jacobian_full, dim=2)
        end = time.time()
        if debug:
            print(jacobian_full.shape)
            print("Time Elapse for 1 Jacobian Full: ", end - start)
        return jacobian_full

    def _attn_head_jacobian(self, q, k, v, attn_mask):
        """
        same as jacobian above, but faster
        referene code:
        https://github.com/lena-voita/the-story-of-heads/blob/master/lib/layers/attn_lrp.py
        """
        # input shapes: (q, k, v) - [batch_size, n_q or n_kv, dim per head]
        # attn_head_mask: [batch_size, n_q, n_kv]
        assert len(q.shape) == 3 and len(attn_mask.shape) == 3

        ATTN_BIAS_VALUE = -1e9
        key_depth_per_head = float(q.shape[-1])
        q = q / (key_depth_per_head ** 0.5)

        attn_bias = ATTN_BIAS_VALUE * (1 - attn_mask)
        logits = torch.matmul(q, k.permute(0, 2, 1)) + attn_bias
        weights = nn.Softmax(dim=-1)(logits)  # [batch_size, n_q, n_kv]
        out = torch.matmul(weights, v)  # [batch_size, n_q, dim/n_heads]

        batch_size, n_kv, dim_per_head = v.shape[0], v.shape[1], v.shape[2]

        diag_flat_weights = torch.einsum('ij,jqk->iqjk',
                                         torch.eye(weights.shape[0]), weights)  # [b, n_q, b, n_kv]
        flat_jac_v = diag_flat_weights[:, :, None, :, :, None] * \
                     torch.eye(dim_per_head)[None, None, :, None, None, :]
        # ^-- shape: [batch_size, n_q, dim/h, batch_size, n_kv, dim/h]
        # torch.Size([1, 48, 64, 1, 48, 64])

        # ... just to get around this torch.tile(v[:, None], [1, out.shape[1], 1, 1])
        jac_out_wrt_weights = torch.cat(out.shape[1] * [v[:, None]], dim=1)
        jac_out_wrt_weights = jac_out_wrt_weights.permute([0, 1, 3, 2])
        # ^-- [batch_size, n_q, (dim), (n_kv)]

        softmax_jac = (weights[..., None] * torch.eye(weights.shape[-1])
                       - weights[..., None, :] * weights[..., :, None])  # <-- [batch_size, n_q, n_kv, n_kv]
        jac_out_wrt_logits = jac_out_wrt_weights @ softmax_jac  # [batch_size, n_q, (dim), (n_kv)]

        jac_out_wrt_k = jac_out_wrt_logits[..., None] * q[:, :, None, None, :]  # [b, (n_q, dim), (n_kv, dim)]

        # product axes:                    b  q  d  kv   d       b  q      d    kv d
        jac_out_wrt_q = jac_out_wrt_logits[:, :, :, :, None] * k[:, None, None, :, :]
        jac_out_wrt_q = jac_out_wrt_q.sum(dim=3, keepdim=True)
        jac_out_wrt_q = jac_out_wrt_q / float(key_depth_per_head) ** 0.5
        jac_out_wrt_q = jac_out_wrt_q * torch.eye(jac_out_wrt_q.shape[1])[None, :, None, :, None]

        flat_jac_k = jac_out_wrt_k[..., None, :, :] * torch.eye(q.shape[0])[:, None, None, :, None, None]
        flat_jac_q = jac_out_wrt_q[..., None, :, :] * torch.eye(q.shape[0])[:, None, None, :, None, None]
        # final shape of flat_jac_{q, k}: [(batch_size, n_q, dim), (batch_size, n_kv, dim)]

        return flat_jac_q, flat_jac_k, flat_jac_v

    def backward_lrp(self, relevance_score, layer_module_index, lrp_detour="quick"):
        """
        This is the lrp explicitily considering the attention layer.
        """

        layer_name_value = 'model.bert.encoder.' + str(layer_module_index) + \
                           '.attention.self.value'
        layer_name_query = 'model.bert.encoder.' + str(layer_module_index) + \
                           '.attention.self.query'
        layer_name_key = 'model.bert.encoder.' + str(layer_module_index) + \
                         '.attention.self.key'
        value_in = func_inputs[layer_name_value][0]
        value_out = func_activations[layer_name_value]
        query_in = func_inputs[layer_name_query][0]
        query_out = func_activations[layer_name_query]
        key_in = func_inputs[layer_name_key][0]
        key_out = func_activations[layer_name_key]
        layer_name_self = 'model.bert.encoder.' + str(layer_module_index) + \
                          '.attention.self'
        context_layer = func_activations[layer_name_self][0]
        attention_mask = func_inputs[layer_name_self][1]
        if lrp_detour == "quick":
            # Instead of jacobian, we may estimate this using a linear layer
            # This turns out to be a good estimate in general.
            relevance_query = \
                torch.autograd.grad(context_layer, query_out,
                                    grad_outputs=relevance_score,
                                    retain_graph=True)[0]
            relevance_key = \
                torch.autograd.grad(context_layer, key_out,
                                    grad_outputs=relevance_score,
                                    retain_graph=True)[0]
            relevance_value = \
                torch.autograd.grad(context_layer, value_out,
                                    grad_outputs=relevance_score,
                                    retain_graph=True)[0]

            relevance_query = backprop_lrp_fc(self.query.weight,
                                              self.query.bias,
                                              query_in,
                                              relevance_query)
            relevance_key = backprop_lrp_fc(self.key.weight,
                                            self.key.bias,
                                            key_in,
                                            relevance_key)
            relevance_value = backprop_lrp_fc(self.value.weight,
                                              self.value.bias,
                                              value_in,
                                              relevance_value)
            relevance_score = relevance_query + relevance_key + relevance_value
        elif lrp_detour == "jacobian":
            print("Full Jacobian can be very slow. Consider our validated quick method.")
            query_out_head = self.transpose_for_scores(query_out)
            key_out_head = self.transpose_for_scores(key_out)
            value_out_head = self.transpose_for_scores(value_out)
            relevance_score = self.transpose_for_context(relevance_score)  # [b, n_h, seq_l, h_dim]

            b_n, n_h, seq_l, h_dim = query_out_head.shape[0], query_out_head.shape[1], query_out_head.shape[2], \
                                     query_out_head.shape[3]
            query_out_head_flat = query_out_head.reshape([-1, seq_l, h_dim])
            key_out_head_flat = key_out_head.reshape([-1, seq_l, h_dim])
            value_out_head_flat = value_out_head.reshape([-1, seq_l, h_dim])
            relevance_score_flat = relevance_score.reshape([-1, seq_l, h_dim])
            attention_mask_flat = torch.cat(n_h * [attention_mask], dim=1).reshape([-1, 1, seq_l])

            # flatten them to save memory
            flat_relevence_qs = []
            flat_relevence_ks = []
            flat_relevence_vs = []
            for i in range(relevance_score_flat.shape[0]):
                flat_jac_q, flat_jac_k, flat_jac_v = \
                    self._attn_head_jacobian(query_out_head_flat[i, None],
                                             key_out_head_flat[i, None],
                                             value_out_head_flat[i, None],
                                             attention_mask_flat[i, None])
                output_flat = self.attention_core(query_out_head_flat[i, None],
                                                  key_out_head_flat[i, None],
                                                  value_out_head_flat[i, None],
                                                  attention_mask_flat[i, None])
                flat_relevence_q, flat_relevence_k, flat_relevence_v = \
                    backprop_lrp_jacobian((flat_jac_q, flat_jac_k, flat_jac_v),
                                          output_flat,
                                          relevance_score_flat[i, None],
                                          (query_out_head_flat[i, None],
                                           key_out_head_flat[i, None],
                                           value_out_head_flat[i, None]))
                flat_relevence_qs.append(flat_relevence_q)
                flat_relevence_ks.append(flat_relevence_k)
                flat_relevence_vs.append(flat_relevence_v)
            flat_relevence_qs = torch.stack(flat_relevence_qs, dim=0)
            flat_relevence_ks = torch.stack(flat_relevence_ks, dim=0)
            flat_relevence_vs = torch.stack(flat_relevence_vs, dim=0)
            relevance_query = flat_relevence_qs.reshape(b_n, n_h, seq_l, h_dim).contiguous().permute(0, 2, 1,
                                                                                                     3).reshape(b_n,
                                                                                                                seq_l,
                                                                                                                -1).contiguous()
            relevance_key = flat_relevence_ks.reshape(b_n, n_h, seq_l, h_dim).contiguous().permute(0, 2, 1, 3).reshape(
                b_n, seq_l, -1).contiguous()
            relevance_value = flat_relevence_vs.reshape(b_n, n_h, seq_l, h_dim).contiguous().permute(0, 2, 1,
                                                                                                     3).reshape(b_n,
                                                                                                                seq_l,
                                                                                                                -1).contiguous()

            # linear layers and we are done!
            relevance_query = backprop_lrp_fc(self.query.weight,
                                              self.query.bias,
                                              query_in,
                                              relevance_query)
            relevance_key = backprop_lrp_fc(self.key.weight,
                                            self.key.bias,
                                            key_in,
                                            relevance_key)
            relevance_value = backprop_lrp_fc(self.value.weight,
                                              self.value.bias,
                                              value_in,
                                              relevance_value)
            relevance_score = relevance_query + relevance_key + relevance_value
        return relevance_score


class MyBertSelfOutput(BertSelfOutput):
    def backward_lrp(self, relevance_score, layer_module_index):
        # residual conection handler
        layer_name = 'model.bert.encoder.' + str(layer_module_index) + \
                        '.attention.output'
        output_in_input = func_inputs[layer_name][1]
        output_out = func_activations[layer_name]
        relevance_score_residual = \
            torch.autograd.grad(output_out, output_in_input,
                                grad_outputs=relevance_score,
                                retain_graph=True)[0]
        # main connection
        layer_name_dense = 'model.bert.encoder.' + str(layer_module_index) + \
                            '.attention.output.dense'
        dense_out = func_activations[layer_name_dense]
        relevance_score = \
            torch.autograd.grad(output_out, dense_out,
                                grad_outputs=relevance_score,
                                retain_graph=True)[0]
        dense_in = func_inputs[layer_name_dense][0]
        relevance_score = backprop_lrp_fc(self.dense.weight,
                                          self.dense.bias,
                                          dense_in,
                                          relevance_score)
        return relevance_score, relevance_score_residual


class MyBertAttention(BertAttention):
    def __init__(self, config):
        super(MyBertAttention, self).__init__(config)
        # reconfigure self and output, we don't consider add_cross_attention is True
        self.self = MyBertSelfAttention(config)
        self.output = MyBertSelfOutput(config)

    def backward_lrp(self, relevance_score, layer_module_index):
        relevance_score, relevance_score_residual = \
            self.output.backward_lrp(relevance_score, layer_module_index)
        relevance_score = self.self.backward_lrp(relevance_score, layer_module_index)
        # merge
        relevance_score = relevance_score + relevance_score_residual
        return relevance_score


class MyBertIntermediate(BertIntermediate):
    def backward_lrp(self, relevance_score, layer_module_index):
        layer_name = 'model.bert.encoder.' + str(layer_module_index) + \
                        '.intermediate.dense'
        dense_in = func_inputs[layer_name][0]
        relevance_score = backprop_lrp_fc(self.dense.weight,
                                          self.dense.bias,
                                          dense_in,
                                          relevance_score)
        return relevance_score


class MyBertOutput(BertOutput):
    def backward_lrp(self, relevance_score, layer_module_index):
        # residual conection handler
        layer_name = 'model.bert.encoder.' + str(layer_module_index) + \
                        '.output'
        output_in_input = func_inputs[layer_name][1]   # the model.bert.encoder.layer_i.attention output
        output_out = func_activations[layer_name]    # the whole layer output
        relevance_score_residual = \
            torch.autograd.grad(output_out, output_in_input,
                                grad_outputs=relevance_score,
                                retain_graph=True)[0]
        # main connection
        layer_name_dense = 'model.bert.encoder.' + str(layer_module_index) + \
                            '.output.dense'
        dense_out = func_activations[layer_name_dense]
        relevance_score = \
            torch.autograd.grad(output_out, dense_out,
                                grad_outputs=relevance_score,
                                retain_graph=True)[0]
        dense_in = func_inputs[layer_name_dense][0]
        relevance_score = backprop_lrp_fc(self.dense.weight,
                                          self.dense.bias,
                                          dense_in,
                                          relevance_score)
        return relevance_score, relevance_score_residual


class MyBertLayer(BertLayer):
    def __init__(self, config):
        super(MyBertLayer, self).__init__(config)
        # we don't consider add_cross_attention is True
        self.attention = MyBertAttention(config)
        self.intermediate = MyBertIntermediate(config)
        self.output = MyBertOutput(config)

    def backward_lrp(self, relevance_score, layer_module_index):
        relevance_score, relevance_score_residual = self.output.backward_lrp(relevance_score, layer_module_index)
        relevance_score = self.intermediate.backward_lrp(relevance_score, layer_module_index)
        # merge
        relevance_score += relevance_score_residual
        relevance_score = self.attention.backward_lrp(relevance_score, layer_module_index)
        return relevance_score


class MyBertEncoder(BertEncoder):
    def __init__(self, config):
        self.num_hidden_layers = config.num_hidden_layers
        super(MyBertEncoder, self).__init__(config)
        self.layer = nn.ModuleList([MyBertLayer(config) for _ in range(config.num_hidden_layers)])

    def backward_lrp(self, relevance_score):
        # backout layer by layer from last to the first
        layer_module_index = self.num_hidden_layers - 1
        for layer_module in reversed(self.layer):
            relevance_score = layer_module.backward_lrp(relevance_score, layer_module_index)
            layer_module_index -= 1

        return relevance_score


class MyBertPooler(BertPooler):
    def __init__(self, config):
        super(MyBertPooler, self).__init__(config)

    def backward_lrp(self, relevance_score):
        dense_in = func_inputs['model.bert.pooler.dense'][0]
        relevance_score = backprop_lrp_fc(self.dense.weight,
                                          self.dense.bias,
                                          dense_in,
                                          relevance_score)
        # we need to scatter this to all hidden states, but only first
        # one matters!
        pooler_in = func_inputs['model.bert.pooler'][0]
        relevance_score_all = torch.zeros_like(pooler_in)
        relevance_score_all[:, 0] = relevance_score
        return relevance_score_all


class MyBertModel(BertModel):
    def __init__(self, config):
        super(MyBertModel, self).__init__(config)
        # reconfigure encoder and pooler
        self.encoder = MyBertEncoder(config)
        self.pooler = MyBertPooler(config)

    def backward_lrp(self, relevance_score):
        relevance_score = self.pooler.backward_lrp(relevance_score)
        relevance_score = self.encoder.backward_lrp(relevance_score)
        return relevance_score


class MyBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super(MyBertForSequenceClassification, self).__init__(config)
        # reconfigure self.bert
        self.bert = MyBertModel(config)

        init_hooks_lrp(self)

    def backward_lrp(self, relevance_score):
        classifier_in = func_inputs['model.classifier'][0]
        # classifier_out = func_activations['model.classifier']
        relevance_score = backprop_lrp_fc(self.classifier.weight,
                                          self.classifier.bias,
                                          classifier_in,
                                          relevance_score)
        relevance_score = self.bert.backward_lrp(relevance_score)
        return relevance_score
