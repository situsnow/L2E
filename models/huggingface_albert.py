import collections, torch, copy, math
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_albert import AlbertForSequenceClassification, AlbertModel, AlbertTransformer, \
    AlbertLayer, AlbertAttention, AlbertLayerGroup
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

    # AlBertForSequenceClassification
    model.classifier.register_forward_hook(
        get_inputivation('model.classifier'))
    model.classifier.register_forward_hook(
        get_activation('model.classifier'))

    # AlbertModel.Pooler
    model.albert.pooler.register_forward_hook(
        get_inputivation('model.albert.pooler'))
    model.albert.pooler.register_forward_hook(
        get_activation('model.albert.pooler'))

    # AlbertEmbedding
    model.albert.embeddings.word_embeddings.register_forward_hook(
        get_activation('model.albert.embeddings.word_embeddings'))
    model.albert.embeddings.register_forward_hook(
        get_activation('model.albert.embeddings'))

    # albert.encoder.embedding_hidden_mapping_in
    albert_encoder_embedding_hidden_mapping_in = 'model.albert.encoder.embedding_hidden_mapping_in'
    model.albert.encoder.embedding_hidden_mapping_in.register_forward_hook(
        get_inputivation(albert_encoder_embedding_hidden_mapping_in))
    model.albert.encoder.embedding_hidden_mapping_in.register_forward_hook(
        get_activation(albert_encoder_embedding_hidden_mapping_in))

    # there will be config.num_hidden_layers that share the parameters
    for i in range(model.albert.encoder.config.num_hidden_layers):
        # albert.encoder.albert_layer_groups
        # there will be only one group, and one layer in each group
        group_module_index = 0
        for group_module_layer in model.albert.encoder.albert_layer_groups:

            group_layer_module_index = 0
            for layer_module_layer in group_module_layer.albert_layers:
                indices = ".%d.%d.%d" % (i, group_layer_module_index, group_layer_module_index)

                # model.albert.encoder...full_layer_layer_norm
                full_layer_layer_norm = 'model.albert.encoder' + indices + '.full_layer_layer_norm'
                layer_module_layer.full_layer_layer_norm.register_forward_hook(
                    get_inputivation(full_layer_layer_norm))
                # added by Snow for residual connection
                layer_module_layer.full_layer_layer_norm.register_forward_hook(
                    get_activation(full_layer_layer_norm))

                # model.albert.encoder...ffn_output
                ffn_output = 'model.albert.encoder' + indices + '.ffn_output'
                layer_module_layer.ffn_output.register_forward_hook(
                    get_inputivation(ffn_output))
                layer_module_layer.ffn_output.register_forward_hook(
                    get_activation(ffn_output))

                # model.albert.encoder...ffn
                ffn = 'model.albert.encoder' + indices + '.ffn'
                layer_module_layer.ffn.register_forward_hook(
                    get_inputivation(ffn))
                layer_module_layer.ffn.register_forward_hook(
                    get_activation(ffn))

                # attention.LayerNorm
                layer_norm = 'model.albert.encoder' + indices + '.attention.LayerNorm'
                layer_module_layer.attention.LayerNorm.register_forward_hook(
                    get_inputivation(layer_norm))
                layer_module_layer.attention.LayerNorm.register_forward_hook(
                    get_activation(layer_norm))

                # attention.dense
                dense = 'model.albert.encoder' + indices + '.attention.dense'
                layer_module_layer.attention.dense.register_forward_hook(
                    get_inputivation(dense))
                layer_module_layer.attention.dense.register_forward_hook(
                    get_activation(dense))

                # attention.query
                query = 'model.albert.encoder' + indices + '.attention.query'
                layer_module_layer.attention.query.register_forward_hook(
                    get_inputivation(query))
                layer_module_layer.attention.query.register_forward_hook(
                    get_activation(query))

                # attention.key
                key = 'model.albert.encoder' + indices + '.attention.key'
                layer_module_layer.attention.key.register_forward_hook(
                    get_inputivation(key))
                layer_module_layer.attention.key.register_forward_hook(
                    get_activation(key))

                # attention.value
                value = 'model.albert.encoder' + indices + '.attention.value'
                layer_module_layer.attention.value.register_forward_hook(
                    get_inputivation(value))
                layer_module_layer.attention.value.register_forward_hook(
                    get_activation(value))

                # attention
                attention = 'model.albert.encoder' + indices + '.attention'
                layer_module_layer.attention.register_forward_hook(
                    get_inputivation(attention))
                layer_module_layer.attention.register_forward_hook(
                    get_activation_multi(attention))

                group_layer_module_index += 1
            group_module_index += 1


class MyAlbertAttention(AlbertAttention):

    # Copied from transformers.modeling_bert.BertSelfAttention.transpose_for_scores
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_ids, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(input_ids)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Should find a better way to do this
        # w = (
        #     self.dense.weight.t()
        #     .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
        #     .to(context_layer.dtype)
        # )
        # b = self.dense.bias.to(context_layer.dtype)
        #
        # projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b

        # copy from modeling_bert so that we can get the hook of self.dense
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = self.dense(context_layer)

        projected_context_layer_dropout = self.output_dropout(context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)

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

    def backward_lrp(self, relevance_score, layer_module_index, group_module_index, group_layer_module_index, lrp_detour="quick"):
        """
        This is the lrp explicitily considering the attention layer.
        """
        indices = '%d.%d.%d' % (layer_module_index, group_module_index, group_layer_module_index)
        layer_name_value = 'model.albert.encoder.' + indices + '.attention.value'
        layer_name_query = 'model.albert.encoder.' + indices + '.attention.query'
        layer_name_key = 'model.albert.encoder.' + indices + '.attention.key'
        value_in = func_inputs[layer_name_value][0]
        value_out = func_activations[layer_name_value]
        query_in = func_inputs[layer_name_query][0]
        query_out = func_activations[layer_name_query]
        key_in = func_inputs[layer_name_key][0]
        key_out = func_activations[layer_name_key]
        layer_name_self = 'model.albert.encoder.' + indices + '.attention'
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


class MyAlbertLayer(AlbertLayer):
    def __init__(self, config):
        super(MyAlbertLayer, self).__init__(config)
        self.attention = MyAlbertAttention(config)

    def backward_lrp(self, relevance_score, layer_module_index, group_module_index, group_layer_module_index):

        indices = ".%d.%d.%d." % (layer_module_index, group_module_index, group_layer_module_index)

        # residual connection handler
        full_layer_layer_norm = 'model.albert.encoder' + indices + 'full_layer_layer_norm'
        layer_norm_in = func_inputs['model.albert.encoder' + indices + 'ffn'][0]  # the second half input before layer norm
        layer_norm_out = func_activations[full_layer_layer_norm]

        relevance_score_residual = torch.autograd.grad(layer_norm_out, layer_norm_in,
                                                       grad_outputs=relevance_score, retain_graph=True)[0]

        # ffn_output
        ffn_output = 'model.albert.encoder' + indices + 'ffn_output'
        ffn_output_out = func_activations[ffn_output]
        # main connection, get relevance from layernorm first
        relevance_score = torch.autograd.grad(layer_norm_out, ffn_output_out,
                                              grad_outputs=relevance_score, retain_graph=True)[0]
        ffn_output_in = func_inputs[ffn_output][0]
        relevance_score = backprop_lrp_fc(self.ffn_output.weight,
                                          self.ffn_output.bias,
                                          ffn_output_in,
                                          relevance_score)
        # ffn
        ffn = 'model.albert.encoder' + indices + 'ffn'
        ffn_in = func_inputs[ffn][0]
        relevance_score = backprop_lrp_fc(self.ffn.weight,
                                          self.ffn.bias,
                                          ffn_in,
                                          relevance_score)

        # combine the residual with relevance score here before attention.LayerNorm
        relevance_score = relevance_score + relevance_score_residual

        # residual connection handler
        # attention.LayerNorm
        attention_layer_norm = 'model.albert.encoder' + indices + 'attention.LayerNorm'
        attention_layer_norm_in = func_inputs['model.albert.encoder' + indices + 'attention'][0]
        attention_layer_norm_out = func_activations[attention_layer_norm]
        relevance_score_residual = torch.autograd.grad(attention_layer_norm_out, attention_layer_norm_in,
                                                       grad_outputs=relevance_score, retain_graph=True)[0]

        # get relevance before attention.dense
        attention_out = func_activations['model.albert.encoder' + indices + 'attention'][0]
        relevance_score = torch.autograd.grad(attention_layer_norm_out, attention_out,
                                              grad_outputs=relevance_score, retain_graph=True)[0]

        # attention.dense
        attention_dense = 'model.albert.encoder' + indices + 'attention.dense'
        dense_in = func_inputs[attention_dense][0]
        relevance_score = backprop_lrp_fc(self.attention.dense.weight,
                                          self.attention.dense.bias,
                                          dense_in, relevance_score)
        # attention
        relevance_score = self.attention.backward_lrp(relevance_score, layer_module_index,
                                                      group_module_index, group_layer_module_index)
        # sum relevance_score and residual
        relevance_score = relevance_score + relevance_score_residual

        return relevance_score


class MyAlbertLayerGroup(AlbertLayerGroup):
    def __init__(self, config):
        super(MyAlbertLayerGroup, self).__init__(config)
        self.albert_layers = nn.ModuleList([MyAlbertLayer(config) for _ in range(config.inner_group_num)])
        self.config = config

    def backward_lrp(self, relevance_score, layer_module_index, group_module_index):
        group_layer_module_index = self.config.inner_group_num - 1
        for layer_module in reversed(self.albert_layers):
            relevance_score = layer_module.backward_lrp(relevance_score, layer_module_index,
                                                        group_module_index,
                                                        group_layer_module_index)
            group_layer_module_index -= 1
        return relevance_score


class MyAlbertTransformer(AlbertTransformer):
    def __init__(self, config):
        super(MyAlbertTransformer, self).__init__(config)
        self.albert_layer_groups = nn.ModuleList([MyAlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def backward_lrp(self, relevance_score):

        # parameters are shared among layers
        layer_module_index = self.config.num_hidden_layers - 1
        while layer_module_index >= 0:
            group_module_index = self.config.num_hidden_groups - 1
            for layer_module in reversed(self.albert_layer_groups):
                # there will be only one
                relevance_score = layer_module.backward_lrp(relevance_score, layer_module_index, group_module_index)
                group_module_index -= 1
            layer_module_index -= 1

        classifier_in = func_inputs['model.albert.encoder.embedding_hidden_mapping_in'][0]

        relevance_score = backprop_lrp_fc(self.embedding_hidden_mapping_in.weight,
                                          self.embedding_hidden_mapping_in.bias,
                                          classifier_in,
                                          relevance_score)
        return relevance_score


class MyAlbertModel(AlbertModel):
    def __init__(self, config):
        super(MyAlbertModel, self).__init__(config)
        self.encoder = MyAlbertTransformer(config)

    def backward_lrp(self, relevance_score):
        # pooler
        pooler_in = func_inputs['model.albert.pooler'][0]
        relevance_score = backprop_lrp_fc(self.pooler.weight,
                                          self.pooler.bias,
                                          pooler_in,
                                          relevance_score)

        # scatter to all hidden states, but only first the first one matters.
        last_layer = self.encoder.config.num_hidden_layers - 1
        last_group = self.encoder.config.num_hidden_groups - 1
        last_group_layer = self.encoder.config.inner_group_num - 1

        indices = "%d.%d.%d" % (last_layer, last_group, last_group_layer)
        layer_name = 'model.albert.encoder.' + indices + '.full_layer_layer_norm'
        encoder_in = func_inputs[layer_name][0]

        relevance_score_all = torch.zeros_like(encoder_in)
        relevance_score_all[:, 0] = relevance_score

        relevance_score = self.encoder.backward_lrp(relevance_score_all)

        return relevance_score


class MyAlbertForSequenceClassification(AlbertForSequenceClassification):
    def __init__(self, config):
        super(MyAlbertForSequenceClassification, self).__init__(config)

        self.albert = MyAlbertModel(config)

        init_hooks_lrp(self)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):

        # set output_attentions to be true for backward pass
        return super(MyAlbertForSequenceClassification, self).forward(input_ids, attention_mask, head_mask,
                                                                      inputs_embeds, labels, output_attentions=True,
                                                                      output_hidden_states=output_hidden_states,
                                                                      return_dict=return_dict)

    def backward_lrp(self, relevance_score):

        classifier_in = func_inputs['model.classifier'][0]

        relevance_score = backprop_lrp_fc(self.classifier.weight,
                                          self.classifier.bias,
                                          classifier_in,
                                          relevance_score)

        relevance_score = self.albert.backward_lrp(relevance_score)
        return relevance_score
