import collections, torch, copy, math
import torch.nn as nn
from transformers.modeling_distilbert import DistilBertForSequenceClassification, DistilBertModel, \
    Transformer, TransformerBlock, MultiHeadSelfAttention
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
        Initialize all the hooks required for full lrp for DistilBERT model.
        """
    # in order to backout all the lrp through layers
    # you need to register hooks here.

    # DistilBertForSequenceClassification
    model.pre_classifier.register_forward_hook(
        get_inputivation('model.pre_classifier'))
    model.pre_classifier.register_forward_hook(
        get_activation('model.pre_classifier'))
    model.classifier.register_forward_hook(
        get_inputivation('model.classifier'))
    model.classifier.register_forward_hook(
        get_activation('model.classifier'))

    # DistilBertModel
    model.distilbert.embeddings.word_embeddings.register_forward_hook(
        get_activation('model.distilbert.embeddings.word_embeddings'))
    model.distilbert.embeddings.register_forward_hook(
        get_activation('model.distilbert.embeddings'))

    # Transformer, TransformerBlock
    layer_module_index = 0
    for module_layer in model.distilbert.transformer.layer:

        # transformer output_layer_norm
        layer_name_output_layer_norm = 'model.distilbert.transformer.' + str(layer_module_index) + \
                                       '.output_layer_norm'
        module_layer.output_layer_norm.register_forward_hook(
            get_inputivation(layer_name_output_layer_norm))
        # added by Snow for residual connection
        module_layer.output_layer_norm.register_forward_hook(
            get_activation(layer_name_output_layer_norm))

        # ffn
        layer_name_ffn_lin1 = 'model.distilbert.transformer.' + str(layer_module_index) + '.ffn.lin1'
        module_layer.ffn.lin1.register_forward_hook(
            get_inputivation(layer_name_ffn_lin1))
        module_layer.ffn.lin1.register_forward_hook(
            get_activation(layer_name_ffn_lin1))

        layer_name_ffn_lin2 = 'model.distilbert.transformer.' + str(layer_module_index) + '.ffn.lin2'
        module_layer.ffn.lin2.register_forward_hook(
            get_inputivation(layer_name_ffn_lin2))
        module_layer.ffn.lin2.register_forward_hook(
            get_activation(layer_name_ffn_lin2))

        # layer_name_ffn = 'model.distilbert.transformer.' + str(layer_module_index) + '.ffn'
        # module_layer.ffn.register_forward_hook(
        #     get_inputivation(layer_name_ffn))
        # module_layer.ffn.register_forward_hook(
        #     get_activation(layer_name_ffn))

        # transformer sa_layer_norm
        layer_name_sa_layer_norm = 'model.distilbert.transformer.' + str(layer_module_index) + '.sa_layer_norm'
        module_layer.sa_layer_norm.register_forward_hook(
            get_inputivation(layer_name_sa_layer_norm))
        # added by Snow for residual connection
        module_layer.sa_layer_norm.register_forward_hook(
            get_activation(layer_name_sa_layer_norm))

        # transformer, SelfAttention MultiHeadSelfAttention
        layer_name_attention_q_lin = 'model.distilbert.transformer.' + str(layer_module_index) + '.attention.q_lin'
        module_layer.attention.q_lin.register_forward_hook(
            get_inputivation(layer_name_attention_q_lin))
        module_layer.attention.q_lin.register_forward_hook(
            get_activation(layer_name_attention_q_lin))

        layer_name_attention_k_lin = 'model.distilbert.transformer.' + str(layer_module_index) + '.attention.k_lin'
        module_layer.attention.k_lin.register_forward_hook(
            get_inputivation(layer_name_attention_k_lin))
        module_layer.attention.k_lin.register_forward_hook(
            get_activation(layer_name_attention_k_lin))

        layer_name_attention_v_lin = 'model.distilbert.transformer.' + str(layer_module_index) + '.attention.v_lin'
        module_layer.attention.v_lin.register_forward_hook(
            get_inputivation(layer_name_attention_v_lin))
        module_layer.attention.v_lin.register_forward_hook(
            get_activation(layer_name_attention_v_lin))

        layer_name_attention = 'model.distilbert.transformer.' + str(layer_module_index) + '.attention'
        module_layer.attention.register_forward_hook(
            get_inputivation(layer_name_attention))
        module_layer.attention.register_forward_hook(
            get_activation_multi(layer_name_attention))

        layer_name_attention_out_lin = 'model.distilbert.transformer.' + str(layer_module_index) + '.attention.out_lin'
        module_layer.attention.out_lin.register_forward_hook(
            get_inputivation(layer_name_attention_out_lin))
        module_layer.attention.out_lin.register_forward_hook(
            get_activation(layer_name_attention_out_lin))

        #
        # # encapsulate module - Added by Snow 20 Jan 2020
        # layer_name_selfattention = 'model.distilbert.transformer.' + str(layer_module_index) + '.self_attention'
        # module_layer.self_attention.register_forward_hook(
        #     get_inputivation(layer_name_selfattention))
        # module_layer.self_attention.register_forward_hook(
        #     get_activation(layer_name_selfattention))

        layer_module_index += 1


# class MyMultiHeadSelfAttention(nn.Module):
class MyMultiHeadSelfAttention(MultiHeadSelfAttention):

    # def __init__(self, config):
    #     super(MyMultiHeadSelfAttention, self).__init__()
    #     # if config.hidden_size % config.num_attention_heads != 0:
    #     #     raise ValueError(
    #     #         "The hidden size (%d) is not a multiple of the number of attention "
    #     #         "heads (%d)" % (config.hidden_size, config.num_attention_heads))
    #     self.num_attention_heads = config.n_heads
    #     self.attention_head_size = int(config.dim / config.n_heads)
    #     self.all_head_size = self.num_attention_heads * self.attention_head_size
    #
    #     self.q_lin = nn.Linear(config.dim, self.all_head_size)
    #     self.k_lin = nn.Linear(config.dim, self.all_head_size)
    #     self.v_lin = nn.Linear(config.dim, self.all_head_size)
    #
    #     self.out_lin = nn.Linear(config.dim, self.all_head_size)
    #
    #     self.dropout = nn.Dropout(config.attention_dropout)
    #
    # def transpose_for_scores(self, x):
    #     new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    #     x = x.view(*new_x_shape)
    #     return x.permute(0, 2, 1, 3)
    #
    # def transpose_for_context(self, x):
    #     new_x_shape = x.size()[:2] + \
    #                   (self.num_attention_heads, self.attention_head_size,)
    #     x = x.view(*new_x_shape)
    #     return x.permute(0, 2, 1, 3).contiguous()
    #
    # def transpose_for_value(self, x):
    #     x = x.permute(0, 2, 1, 3).contiguous()
    #     new_x_shape = x.size()[:2] + (self.all_head_size,)
    #     return x.view(*new_x_shape)
    #
    # def forward(self, hidden_states, attention_mask):
    #     mixed_query_layer = self.q_lin(hidden_states)
    #     mixed_key_layer = self.k_lin(hidden_states)
    #     mixed_value_layer = self.v_lin(hidden_states)
    #
    #     query_layer = self.transpose_for_scores(mixed_query_layer)
    #     key_layer = self.transpose_for_scores(mixed_key_layer)
    #     value_layer = self.transpose_for_scores(mixed_value_layer)
    #
    #     # Take the dot product between "query" and "key" to get the raw attention scores.
    #     attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    #     attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    #     # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
    #     attention_scores = attention_scores + attention_mask
    #
    #     # Normalize the attention scores to probabilities.
    #     attention_probs = nn.Softmax(dim=-1)(attention_scores)
    #
    #     # This is actually dropping out entire tokens to attend to, which might
    #     # seem a bit unusual, but is taken from the original Transformer paper.
    #     attention_probs = self.dropout(attention_probs)
    #
    #     context_layer = torch.matmul(attention_probs, value_layer)
    #     context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    #     new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    #     context_layer = context_layer.view(*new_context_layer_shape)
    #
    #     context_layer = self.out_lin(context_layer)
    #     return context_layer, attention_probs

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

    def backprop_lrp(self, relevance_score, layer_module_index, lrp_detour="quick"):
        """
        This is the lrp explicitily considering the attention layer.
        """

        layer_name_value = 'model.distilbert.transformer.' + str(layer_module_index) + \
                           '.attention.v_lin'
        layer_name_query = 'model.distilbert.transformer.' + str(layer_module_index) + \
                           '.attention.q_lin'
        layer_name_key = 'model.distilbert.transformer.' + str(layer_module_index) + \
                         '.attention.k_lin'
        value_in = func_inputs[layer_name_value][0]
        value_out = func_activations[layer_name_value]
        query_in = func_inputs[layer_name_query][0]
        query_out = func_activations[layer_name_query]
        key_in = func_inputs[layer_name_key][0]
        key_out = func_activations[layer_name_key]
        layer_name_self = 'model.distilbert.transformer.' + str(layer_module_index) + \
                          '.attention'
        context_layer = func_activations[layer_name_self][0]

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

            relevance_query = backprop_lrp_fc(self.q_lin.weight,
                                              self.q_lin.bias,
                                              query_in,
                                              relevance_query)
            relevance_key = backprop_lrp_fc(self.k_lin.weight,
                                            self.k_lin.bias,
                                            key_in,
                                            relevance_key)
            relevance_value = backprop_lrp_fc(self.v_lin.weight,
                                              self.v_lin.bias,
                                              value_in,
                                              relevance_value)
            relevance_score = relevance_query + relevance_key + relevance_value
        elif lrp_detour == "jacobian":
            print("Full Jacobian can be very slow. Consider our validated quick method.")
            attention_mask = func_inputs[layer_name_self][1]
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


class MyTransformerBlock(TransformerBlock):
    def __init__(self, config):
        super(MyTransformerBlock, self).__init__(config)

        # revise attention class
        self.attention = MyMultiHeadSelfAttention(config)
        # self.out_lin = nn.Linear(config.dim, config.dim)

    # def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False):
    #     """
    #     Parameters:
    #         x: torch.tensor(bs, seq_length, dim)
    #         attn_mask: torch.tensor(bs, seq_length)
    #
    #     Returns:
    #         sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
    #         torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
    #     """
    #     # Self-Attention
    #     # sa_output = self.attention(x, attn_mask)
    #     sa_output = self.attention(query=x, key=x, value=x, mask=attn_mask, output_attentions=output_attentions)
    #     if output_attentions:
    #         sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
    #     else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
    #         assert type(sa_output) == tuple
    #         sa_output = sa_output[0]
    #     # sa_output = self.out_lin(sa_output)
    #     sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
    #
    #     # Feed Forward Network
    #     ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
    #     ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
    #
    #     output = (ffn_output,)
    #     if output_attentions:
    #         output = (sa_weights,) + output
    #     return output

    def backward_lrp(self, relevance_score, layer_module_index):
        # residual connection handler
        layer_name_output = 'model.distilbert.transformer.' + str(layer_module_index) + '.output_layer_norm'
        output_layer_output_in = func_inputs['model.distilbert.transformer.' + str(layer_module_index) + '.ffn.lin1'][0]
        output_layer_output_out = func_activations[layer_name_output]
        relevance_score_residual = torch.autograd.grad(output_layer_output_out,
                                                       output_layer_output_in,
                                                       grad_outputs=relevance_score, retain_graph=True)[0]

        # only pass the relevance score to ffn module
        # FFN
        # backprop reversely
        layer_name_ffn_lin2 = 'model.distilbert.transformer.' + str(layer_module_index) + '.ffn.lin2'
        ffn_lin2_out = func_activations[layer_name_ffn_lin2]

        # main connection: get relevance from layernorm first
        relevance_score = torch.autograd.grad(output_layer_output_out, ffn_lin2_out,
                                              grad_outputs=relevance_score,
                                              retain_graph=True)[0]

        ffn_lin2_in = func_inputs[layer_name_ffn_lin2][0]
        relevance_score = backprop_lrp_fc(self.ffn.lin2.weight,
                                          self.ffn.lin2.bias,
                                          ffn_lin2_in,
                                          relevance_score)

        layer_name_ffn_lin1 = 'model.distilbert.transformer.' + str(layer_module_index) + '.ffn.lin1'
        ffn_lin1_in = func_inputs[layer_name_ffn_lin1][0]

        relevance_score = backprop_lrp_fc(self.ffn.lin1.weight,
                                          self.ffn.lin1.bias,
                                          ffn_lin1_in,
                                          relevance_score)

        # combine the residual with relevance score here before sa_layer_norm
        relevance_score = relevance_score + relevance_score_residual

        # relevance_score, relevance_score_residual = self.self_attention.backward_lrp(relevance_score,
        #                                                                              layer_module_index)
        # residual handler
        layer_name_attention_output = 'model.distilbert.transformer.' + str(layer_module_index) + '.sa_layer_norm'
        attention_in = func_inputs['model.distilbert.transformer.' + str(layer_module_index) + '.attention.q_lin'][0]  # residual input
        attention_out = func_activations[layer_name_attention_output]
        relevance_score_residual = torch.autograd.grad(attention_out, attention_in,
                                                       grad_outputs=relevance_score,
                                                       retain_graph=True)[0]

        # MultiHeadSelfAttention
        # need to get the relevance from previous sa_layer_norm first for out_lin
        attention_outlin_out = func_activations['model.distilbert.transformer.' + str(layer_module_index) + '.attention'][0]
        # attention_outlin_out = func_activations[layer_name_outlin][0]
        relevance_score = torch.autograd.grad(attention_out, attention_outlin_out,
                                              grad_outputs=relevance_score,
                                              retain_graph=True)[0]

        layer_name_outlin = 'model.distilbert.transformer.' + str(layer_module_index) + '.attention.out_lin'
        attention_outlin_in = func_inputs[layer_name_outlin][0]
        relevance_score = backprop_lrp_fc(self.attention.out_lin.weight,
                                          self.attention.out_lin.bias,
                                          attention_outlin_in,
                                          relevance_score)

        relevance_score = self.attention.backprop_lrp(relevance_score, layer_module_index)

        # combine the residual with relevance score here again before passing back to previous layer if any
        relevance_score = relevance_score + relevance_score_residual

        return relevance_score


class MyTransformer(Transformer):
    def __init__(self, config):
        super(MyTransformer, self).__init__(config)

        # revise layer class
        layer = MyTransformerBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])

    def backward_lrp(self, relevance_score):
        layer_module_index = self.n_layers - 1
        for layer_module in reversed(self.layer):
            relevance_score = layer_module.backward_lrp(relevance_score, layer_module_index)
            layer_module_index -= 1
        return relevance_score


class MyDistilBertModel(DistilBertModel):

    def __init__(self, config):
        super(MyDistilBertModel, self).__init__(config)

        # skip embedding as it does need calculation in lrp
        # revise transformer class
        self.transformer = MyTransformer(config)  # Encoder

    # def forward(
    #         self,
    #         input_ids=None,
    #         attention_mask=None,
    #         head_mask=None,
    #         inputs_embeds=None,
    #         output_attentions=None,
    #         output_hidden_states=None,
    #         return_dict=None,
    # ):
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids)
    #         attention_mask = attention_mask.float()
    #         attention_mask.require_grad = True
    #         if torch.cuda.is_available(): attention_mask = attention_mask.cuda()
    #     # # We create a 3D attention mask from a 2D tensor mask.
    #     # # Sizes are [batch_size, 1, 1, from_seq_length]
    #     # # So we can broadcast to [batch_size, num_heads, to_seq_length, from_seq_length]
    #     # # this attention mask is more simple than the triangular masking of causal attention
    #     # # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    #     # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    #     #
    #     # # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    #     # # masked positions, this operation will create a tensor which is 0.0 for
    #     # # positions we want to attend and -10000.0 for masked positions.
    #     # # Since we are adding it to the raw scores before the softmax, this is
    #     # # effectively the same as removing these entirely.
    #     # extended_attention_mask = extended_attention_mask.float()
    #     # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    #     # extended_attention_mask.require_grad = True
    #     #
    #     # if torch.cuda.is_available(): extended_attention_mask = extended_attention_mask.cuda()
    #
    #     return super(MyDistilBertModel, self).forward(input_ids, attention_mask, head_mask, inputs_embeds,
    #                                                   output_attentions, output_hidden_states, return_dict)

    def backward_lrp(self, relevance_score):
        relevance_score = self.transformer.backward_lrp(relevance_score)
        return relevance_score


class MyDistilBertForSequenceClassification(DistilBertForSequenceClassification):

    def __init__(self, config):
        super(MyDistilBertForSequenceClassification, self).__init__(config)

        # revise distilBert class
        self.distilbert = MyDistilBertModel(config)
        init_hooks_lrp(self)

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None,
                labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):

        # set output_attentions to be true for backward pass
        return super(MyDistilBertForSequenceClassification, self).forward(input_ids, attention_mask,
                                                                          head_mask, inputs_embeds, labels,
                                                                          output_attentions=True,
                                                                          output_hidden_states=output_hidden_states,
                                                                          return_dict=return_dict)

    def backward_lrp(self, relevance_score):
        classifier_in = func_inputs['model.classifier'][0]
        preclassifier_in = func_inputs['model.pre_classifier'][0]
        relevance_score = backprop_lrp_fc(self.classifier.weight,
                                          self.classifier.bias,
                                          classifier_in,
                                          relevance_score)

        relevance_score = backprop_lrp_fc(self.pre_classifier.weight,
                                          self.pre_classifier.bias,
                                          preclassifier_in,
                                          relevance_score)

        # scatter the relevance score to all hidden states, but only the first one matter.
        last_layer = self.distilbert.transformer.n_layers - 1
        layer_name = 'model.distilbert.transformer.' + str(last_layer) + '.output_layer_norm'
        transformer_in = func_inputs[layer_name][0]
        relevance_score_all = torch.zeros_like(transformer_in)
        relevance_score_all[:, 0] = relevance_score

        relevance_score = self.distilbert.backward_lrp(relevance_score_all)
        return relevance_score
