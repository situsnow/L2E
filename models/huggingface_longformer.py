import collections, torch, copy, math
import torch.nn as nn
from models.longformer.modeling_longformer import LongformerForSequenceClassification, \
    LongformerClassificationHead, LongformerModel, LongformerPooler, LongformerEncoder, LongformerLayer, \
    LongformerAttention, LongformerSelfAttention, LongformerSelfOutput, LongformerIntermediate, LongformerOutput

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

    # classifier
    model.classifier.dense.register_forward_hook(
        get_inputivation('model.classifier.dense'))
    model.classifier.dense.register_forward_hook(
        get_activation('model.classifier.dense'))

    model.classifier.out_proj.register_forward_hook(
        get_inputivation('model.classifier.out_proj'))
    model.classifier.out_proj.register_forward_hook(
        get_activation('model.classifier.out_proj'))

    model.classifier.register_forward_hook(
        get_inputivation('model.classifier'))
    model.classifier.register_forward_hook(
        get_activation('model.classifier'))

    # longformer
    # longformer.pooler
    if model.longformer.pooler is not None:
        model.longformer.pooler.dense.register_forward_hook(
            get_inputivation('model.longformer.pooler.dense'))
        model.longformer.pooler.dense.register_forward_hook(
            get_activation('model.longformer.pooler.dense'))
        model.longformer.pooler.register_forward_hook(
            get_inputivation('model.longformer.pooler'))
        model.longformer.pooler.register_forward_hook(
            get_activation('model.longformer.pooler'))

    # longformer.pooler
    model.longformer.embeddings.word_embeddings.register_forward_hook(
        get_activation('model.longformer.embeddings.word_embeddings'))
    model.longformer.embeddings.register_forward_hook(
        get_activation('model.longformer.embeddings'))

    # LongformerLayer
    layer_module_index = 0
    for module_layer in model.longformer.encoder.layer:

        ## Encoder Output Layer
        layer_name_output_layernorm = 'model.longformer.encoder.' + str(layer_module_index) + '.output.LayerNorm'
        module_layer.output.LayerNorm.register_forward_hook(
            get_inputivation(layer_name_output_layernorm))

        layer_name_dense = 'model.longformer.encoder.' + str(layer_module_index) + '.output.dense'
        module_layer.output.dense.register_forward_hook(
            get_inputivation(layer_name_dense))
        module_layer.output.dense.register_forward_hook(
            get_activation(layer_name_dense))

        layer_name_output = 'model.longformer.encoder.' + str(layer_module_index) + '.output'
        module_layer.output.register_forward_hook(
            get_inputivation(layer_name_output))
        module_layer.output.register_forward_hook(
            get_activation(layer_name_output))

        ## Encoder Intermediate Layer
        layer_name_inter = 'model.longformer.encoder.' + str(layer_module_index) + '.intermediate.dense'
        module_layer.intermediate.dense.register_forward_hook(
            get_inputivation(layer_name_inter))
        module_layer.intermediate.dense.register_forward_hook(
            get_activation(layer_name_inter))

        # LongformerAttention
        # LongformerAttention.LongformerSelfOutput
        layer_name_attn_layernorm = 'model.longformer.encoder.' + str(layer_module_index) + \
                                    '.attention.output.LayerNorm'
        module_layer.attention.output.LayerNorm.register_forward_hook(
            get_inputivation(layer_name_attn_layernorm))

        layer_name_attn = 'model.longformer.encoder.' + str(layer_module_index) + '.attention.output.dense'
        module_layer.attention.output.dense.register_forward_hook(
            get_inputivation(layer_name_attn))
        module_layer.attention.output.dense.register_forward_hook(
            get_activation(layer_name_attn))

        layer_name_attn_output = 'model.longformer.encoder.' + str(layer_module_index) + '.attention.output'
        module_layer.attention.output.register_forward_hook(
            get_inputivation(layer_name_attn_output))
        module_layer.attention.output.register_forward_hook(
            get_activation(layer_name_attn_output))

        # LongformerSelfAttention
        layer_name_self = 'model.longformer.encoder.' + str(layer_module_index) + '.attention.self'
        module_layer.attention.self.register_forward_hook(
            get_inputivation(layer_name_self))
        module_layer.attention.self.register_forward_hook(
            get_activation_multi(layer_name_self))

        layer_name_value = 'model.longformer.encoder.' + str(layer_module_index) + '.attention.self.value'
        module_layer.attention.self.value.register_forward_hook(
            get_inputivation(layer_name_value))
        module_layer.attention.self.value.register_forward_hook(
            get_activation(layer_name_value))

        layer_name_query = 'model.longformer.encoder.' + str(layer_module_index) + '.attention.self.query'
        module_layer.attention.self.query.register_forward_hook(
            get_inputivation(layer_name_query))
        module_layer.attention.self.query.register_forward_hook(
            get_activation(layer_name_query))

        layer_name_key = 'model.longformer.encoder.' + str(layer_module_index) + '.attention.self.key'
        module_layer.attention.self.key.register_forward_hook(
            get_inputivation(layer_name_key))
        module_layer.attention.self.key.register_forward_hook(
            get_activation(layer_name_key))

        layer_module_index += 1


class MyLongformerSelfAttention(LongformerSelfAttention):

    def backward_lrp(self, relevance_score, layer_module_index, lrp_detour="quick"):
        """
        This is the lrp explicitily considering the attention layer.
        """

        layer_name_value = 'model.longformer.encoder.' + str(layer_module_index) + \
                           '.attention.self.value'
        layer_name_query = 'model.longformer.encoder.' + str(layer_module_index) + \
                           '.attention.self.query'
        layer_name_key = 'model.longformer.encoder.' + str(layer_module_index) + \
                         '.attention.self.key'
        value_in = func_inputs[layer_name_value][0]
        value_out = func_activations[layer_name_value]
        query_in = func_inputs[layer_name_query][0]
        query_out = func_activations[layer_name_query]
        key_in = func_inputs[layer_name_key][0]
        key_out = func_activations[layer_name_key]
        layer_name_self = 'model.longformer.encoder.' + str(layer_module_index) + \
                          '.attention.self'
        context_layer = func_activations[layer_name_self][0]
        # attention_mask = func_inputs[layer_name_self][1]
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

        return relevance_score


class MyLongformerSelfOutput(LongformerSelfOutput):
    def backward_lrp(self, relevance_score, layer_module_index):
        # residual conection handler
        layer_name = 'model.longformer.encoder.' + str(layer_module_index) + \
                        '.attention.output'
        output_in_input = func_inputs[layer_name][1]
        output_out = func_activations[layer_name]
        relevance_score_residual = \
            torch.autograd.grad(output_out, output_in_input,
                                grad_outputs=relevance_score,
                                retain_graph=True)[0]
        # main connection
        layer_name_dense = 'model.longformer.encoder.' + str(layer_module_index) + \
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


class MyLongformerAttention(LongformerAttention):
    def __init__(self, config, layer_id=0):
        super(MyLongformerAttention, self).__init__(config)
        self.self = MyLongformerSelfAttention(config, layer_id)
        self.output = MyLongformerSelfOutput(config)

    def backward_lrp(self, relevance_score, layer_module_index):
        relevance_score, relevance_score_residual = \
            self.output.backward_lrp(relevance_score, layer_module_index)
        relevance_score = self.self.backward_lrp(relevance_score, layer_module_index)
        relevance_score = relevance_score.transpose(1, 0)   # because we use the longformer attention, the position is wrong
        # merge
        relevance_score = relevance_score + relevance_score_residual
        return relevance_score


class MyLongformerIntermediate(LongformerIntermediate):
    def backward_lrp(self, relevance_score, layer_module_index):
        layer_name = 'model.longformer.encoder.' + str(layer_module_index) + \
                        '.intermediate.dense'
        dense_in = func_inputs[layer_name][0]
        relevance_score = backprop_lrp_fc(self.dense.weight,
                                          self.dense.bias,
                                          dense_in,
                                          relevance_score)
        return relevance_score

class MyLongformerOutput(LongformerOutput):
    def backward_lrp(self, relevance_score, layer_module_index):
        # residual conection handler
        layer_name = 'model.longformer.encoder.' + str(layer_module_index) + \
                        '.output'
        output_in_input = func_inputs[layer_name][1]   # the model.longformer.encoder.layer_i.attention output
        output_out = func_activations[layer_name]    # the whole layer output
        relevance_score_residual = \
            torch.autograd.grad(output_out, output_in_input,
                                grad_outputs=relevance_score,
                                retain_graph=True)[0]
        # main connection
        layer_name_dense = 'model.longformer.encoder.' + str(layer_module_index) + \
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


class MyLongformerLayer(LongformerLayer):
    def __init__(self, config, layer_id=0):
        super(MyLongformerLayer, self).__init__(config)

        self.attention = MyLongformerAttention(config, layer_id)
        self.intermediate = MyLongformerIntermediate(config)
        self.output = MyLongformerOutput(config)

    def backward_lrp(self, relevance_score, layer_module_index):
        relevance_score, relevance_score_residual = self.output.backward_lrp(relevance_score, layer_module_index)
        relevance_score = self.intermediate.backward_lrp(relevance_score, layer_module_index)
        # merge
        relevance_score += relevance_score_residual
        relevance_score = self.attention.backward_lrp(relevance_score, layer_module_index)
        return relevance_score


class MyLongformerEncoder(LongformerEncoder):

    def __init__(self, config):
        super(MyLongformerEncoder, self).__init__(config)
        self.num_hidden_layers = config.num_hidden_layers
        self.layer = nn.ModuleList([MyLongformerLayer(config, layer_id=i) for i in range(self.num_hidden_layers)])

    def backward_lrp(self, relevance_score):
        # backout layer by layer from last to the first
        layer_module_index = self.num_hidden_layers - 1
        for layer_module in reversed(self.layer):
            relevance_score = layer_module.backward_lrp(relevance_score, layer_module_index)
            layer_module_index -= 1

        return relevance_score


class MyLongformerPooler(LongformerPooler):

    def backward_lrp(self, relevance_score):
        # self.dense
        dense_in = func_inputs['model.longformer.pooler.dense'][0]
        relevance_score = backprop_lrp_fc(self.dense.weight,
                                          self.dense.bias,
                                          dense_in,
                                          relevance_score)

        # we need to scatter this to all hidden states, but only first
        # one matters!
        pooler_in = func_inputs['model.longformer.pooler'][0]
        relevance_score_all = torch.zeros_like(pooler_in)
        relevance_score_all[:, 0] = relevance_score
        return relevance_score_all


class MyLongformerModel(LongformerModel):
    def __init__(self, config):
        super(MyLongformerModel, self).__init__(config)

        self.encoder = MyLongformerEncoder(config)
        self.pooler = None
        # self.pooler = MyLongformerPooler(config) if self.pooler is not None else None

    def backward_lrp(self, relevance_score):

        # if self.pooler is not None:
        #     relevance_score_all = self.pooler.backward_lrp(relevance_score)
        # else:
        # then we need to get the last layer's output
        last_layer = self.config.num_hidden_layers - 1
        layer_name = 'model.longformer.encoder.' + str(last_layer) + '.output'
        # longformer_output_in = func_inputs[layer_name][0]
        longformer_output_in = func_inputs[layer_name][1]
        relevance_score_all = torch.zeros_like(longformer_output_in)
        # there will be extra padding based on the attention window size
        # relevance_score_all[:, 0, :relevance_score.shape[1]] = relevance_score
        relevance_score_all[:, 0] = relevance_score

        return self.encoder.backward_lrp(relevance_score_all)


class MyLongformerClassificationHead(LongformerClassificationHead):

    def backward_lrp(self, relevance_score):
        out_proj_in = func_inputs['model.classifier.out_proj'][0]
        relevance_score = backprop_lrp_fc(self.out_proj.weight,
                                          self.out_proj.bias,
                                          out_proj_in,
                                          relevance_score)
        dense_in = func_inputs['model.classifier.dense'][0]
        relevance_score = backprop_lrp_fc(self.dense.weight,
                                          self.dense.bias,
                                          dense_in,
                                          relevance_score)

        return relevance_score


class MyLongformerForSequenceClassification(LongformerForSequenceClassification):
    def __init__(self, config):
        super(MyLongformerForSequenceClassification, self).__init__(config)
        self.longformer = MyLongformerModel(config)
        self.classifier = MyLongformerClassificationHead(config)

        init_hooks_lrp(self)

    def backward_lrp(self, relevance_score):
        relevance_score = self.classifier.backward_lrp(relevance_score)
        relevance_score = self.longformer.backward_lrp(relevance_score)

        return relevance_score

