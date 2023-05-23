# coding=utf-8
"""PyTorch BERT model with Growth"""


import math
import copy
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.bert.modeling_bert import BertEmbeddings, BertSelfAttention
from transformers import (BertModel, BertForPreTraining, BertForSequenceClassification)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions
)
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import (
    add_start_docstrings,
    logging,
)

from .utils_ex_v2 import LayerNormEx

logger = logging.get_logger(__name__)

CONSTANT_ATTENTION_HEAD_SIZE = 64
CONSTANT_MIN_MASK_VAL = 0.
CONSTANT_MAX_MASK_VAL = 1.0

class BertEmbeddingsEx(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__(config)
        self.LayerNorm = LayerNormEx(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config
        self.in_growth = False
        self.grow_mask_vec = None
        self.old_hidden_size = config.hidden_size
        self.hidden_size = config.hidden_size
        self.debug_print = False
        # before growing, layer_norm should still receive a all-one mask to preserve function 
        self.grow_mask_vec = torch.ones(self.hidden_size, dtype=self.word_embeddings.weight.dtype,
                                        device=self.word_embeddings.weight.device)
        self.grow_mask_vec[:] = CONSTANT_MAX_MASK_VAL


    def set_mask(self, new_dim, val):
        if new_dim is not None:
            new_mask = torch.ones(new_dim, dtype=self.word_embeddings.weight.dtype)
            new_mask[self.old_hidden_size:] = val
            self.grow_mask_vec = new_mask
            self.grow_mask_vec = self.grow_mask_vec.to(self.word_embeddings.weight.device)
        else:
            self.grow_mask_vec[self.old_hidden_size:] = val

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings, self.grow_mask_vec, self.in_growth, self.debug_print)
        embeddings = self.dropout(embeddings)
        if self.in_growth:
            embeddings = embeddings * self.grow_mask_vec
        
        return embeddings



class BertSelfAttentionEx(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super(BertSelfAttention, self).__init__()

        # modify original init function
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = CONSTANT_ATTENTION_HEAD_SIZE
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

        # added for ex
        self.in_grow_dim = False
        self.in_grow_head = False
        self.grow_mask_head = None # for head mask
        self.hidden_size = config.hidden_size
        self.new_head_count = 0
        self.debug_print = False

    def set_mask_head(self, target_head_num, val):
        if target_head_num is not None:
            new_mask = torch.ones(target_head_num * self.attention_head_size, dtype=self.query.weight.dtype)
            new_mask[self.attention_head_size*(target_head_num - self.new_head_count):] = val
            self.grow_mask_head = new_mask
            self.grow_mask_head = self.grow_mask_head.to(self.query.weight.device)
        else:
            temp_head_num = self.grow_mask_head.size()[0] // self.attention_head_size
            self.grow_mask_head[self.attention_head_size*(temp_head_num - self.new_head_count):] = val

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch_size * seq_len * hidden_size(==num_attention_heads * attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3) # batch_size * attention_heads * seq_length * attention_head_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        if self.debug_print and hidden_states.device == torch.device(type="cuda",index=2):
            print("before query")
            print(hidden_states[0][0].size())
            print(hidden_states[0][0][:10])

        mixed_query_layer = self.query(hidden_states)

        if self.in_grow_head:
            mixed_query_layer = mixed_query_layer * self.grow_mask_head

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_all = self.key(encoder_hidden_states)
            value_all = self.value(encoder_hidden_states)
            if self.in_grow_head:
                key_all = key_all * self.grow_mask_head
                value_all = value_all * self.grow_mask_head
            key_layer = self.transpose_for_scores(key_all)
            value_layer = self.transpose_for_scores(value_all)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_all = self.key(hidden_states)
            value_all = self.value(hidden_states)
            if self.in_grow_head:
                key_all = key_all * self.grow_mask_head
                value_all = value_all * self.grow_mask_head
            key_layer = self.transpose_for_scores(key_all)
            value_layer = self.transpose_for_scores(value_all)
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_all = self.key(hidden_states)
            value_all = self.value(hidden_states)
            if self.in_grow_head:
                key_all = key_all * self.grow_mask_head
                value_all = value_all * self.grow_mask_head
            key_layer = self.transpose_for_scores(key_all)
            value_layer = self.transpose_for_scores(value_all)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # batch * heads * seq_len * seq_len

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()


        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutputEx(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(CONSTANT_ATTENTION_HEAD_SIZE * config.num_attention_heads, config.hidden_size)
        self.LayerNorm = LayerNormEx(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.in_grow_dim = False
        self.in_grow_input = False
        self.grow_mask_vec = None 
        self.hidden_size = config.hidden_size
        self.old_hidden_size = config.hidden_size
        self.all_head_size = config.hidden_size
        self.debug_print = False

        # before growing, layer_norm should still receive a all-one mask to preserve function 
        self.grow_mask_vec = torch.ones(self.hidden_size, dtype=self.dense.weight.dtype,
                                        device=self.dense.weight.device)
        self.grow_mask_vec[:] = CONSTANT_MAX_MASK_VAL

    def set_mask(self, target_dim, val):
        if target_dim is not None:
            new_mask = torch.ones(target_dim, dtype=self.dense.weight.dtype)
            new_mask[self.old_hidden_size:] = val
            self.grow_mask_vec = new_mask
            self.grow_mask_vec = self.grow_mask_vec.to(self.dense.weight.device)
        else:
            self.grow_mask_vec[self.old_hidden_size:] = val
            
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.debug_print and hidden_states.device == torch.device(type="cuda",index=2):
            print("attention output")
            print(hidden_states[0][0][:10])
            print("attention input")
            print(input_tensor[0][0])
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor, self.grow_mask_vec, self.in_grow_dim)
        
        if self.in_grow_dim:
            hidden_states = hidden_states * self.grow_mask_vec

        if self.debug_print and hidden_states.device == torch.device(type="cuda",index=2):
            print("output of self_output")
            print(hidden_states[0][0])
        
        return hidden_states


class BertAttentionEx(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttentionEx(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutputEx(config)
        self.pruned_heads = set()
        self.in_growth = False
        

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediateEx(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        
        self.in_grow_dim = False
        self.in_grow_input = False
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.old_intermediate_size = config.intermediate_size
        self.grow_mask_vec = None
        self.debug_print = False

    def set_mask(self, target_dim, val):
        if target_dim is not None:
            new_mask = torch.ones(target_dim, dtype=self.dense.weight.dtype)
            # new_mask[self.old_intermediate_size:] = val
            new_mask[self.old_intermediate_size:] = val
            self.grow_mask_vec = new_mask
            self.grow_mask_vec = self.grow_mask_vec.to(self.dense.weight.device)
        else:
            self.grow_mask_vec[self.old_intermediate_size:] = val


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.debug_print and hidden_states.device == torch.device(type="cuda",index=2):
            print("input of intermediate layer")
            print(hidden_states[0])

        hidden_states = self.dense(hidden_states)
        if self.in_grow_dim:
            hidden_states = hidden_states * self.grow_mask_vec
        
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

class BertOutputEx(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNormEx(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hidden_size = config.hidden_size
        self.old_hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.in_grow_dim = False
        self.in_grow_input = False
        self.grow_mask_vec = None
        self.debug_print = False

        self.grow_mask_vec = torch.ones(self.hidden_size, dtype=self.dense.weight.dtype,
                                        device=self.dense.weight.device)
        self.grow_mask_vec[:] = CONSTANT_MAX_MASK_VAL

    def set_mask(self, target_dim, val):
        if target_dim is not None:
            new_mask = torch.ones(target_dim, dtype=self.dense.weight.dtype)
            new_mask[self.old_hidden_size:] = val
            self.grow_mask_vec = new_mask
            self.grow_mask_vec = self.grow_mask_vec.to(self.dense.weight.device)
        else:
            self.grow_mask_vec[self.old_hidden_size:] = val

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.LayerNorm(hidden_states + input_tensor, self.grow_mask_vec, self.in_grow_dim)

        if self.in_grow_dim:
            hidden_states = hidden_states * self.grow_mask_vec
        
        return hidden_states


class BertLayerEx(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttentionEx(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttentionEx(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediateEx(config)
        self.output = BertOutputEx(config)

        self.is_new_block = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoderEx(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayerEx(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
        self.num_hidden_layers = config.num_hidden_layers
        self.in_growth = False
        self.grow_mask_block = None
        self.dynamic_config = None
        self.debug_print = False # added by yyq

    def set_mask(self, val):
        self.grow_mask_block[:] = val

    def update_config(self):
        if self.dynamic_config is None:
            self.dynamic_config = copy.deepcopy(self.config)
        self.dynamic_config.hidden_size = self.layer[0].output.hidden_size
        self.dynamic_config.intermediate_size = self.layer[0].output.intermediate_size
        self.dynamic_config.num_attention_heads = self.layer[0].attention.self.all_head_size // self.layer[0].attention.self.attention_head_size


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_head_mask = None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            
            if self.in_growth and layer_module.is_new_block: 
                hidden_states = self.grow_mask_block * layer_outputs[0] + (1 - self.grow_mask_block) * hidden_states
            else:
                hidden_states = layer_outputs[0]
            
            if self.debug_print and hidden_states.device == torch.device(type="cuda",index=2):
                print(f"the output of {i}-th layer:")
                print(hidden_states[0].data.cpu())

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPoolerEx(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.hidden_size = config.hidden_size
        self.old_hidden_size = config.hidden_size
        self.in_growth = False
        self.grow_mask_vec = None
        self.debug_print = False

    def set_mask(self, target_dim, val):
        if target_dim is not None:
            new_mask = torch.ones(target_dim, dtype=self.dense.weight.dtype)
            new_mask[self.old_hidden_size:] = val
            self.grow_mask_vec = new_mask
            self.grow_mask_vec = self.grow_mask_vec.to(self.dense.weight.device)
        else:
            self.grow_mask_vec[self.old_hidden_size:] = val

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
    
        first_token_tensor = self.dense(first_token_tensor)
        pooled_output = first_token_tensor

        pooled_output = self.activation(pooled_output)
        if self.in_growth:
            pooled_output = pooled_output * self.grow_mask_vec

        return pooled_output


class BertPredictionHeadTransformEx(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = LayerNormEx(config.hidden_size, eps=config.layer_norm_eps)

        self.hidden_size = config.hidden_size
        self.old_hidden_size = config.hidden_size
        self.in_growth = False
        self.grow_mask_vec = None

        self.grow_mask_vec = torch.ones(self.hidden_size, dtype=self.dense.weight.dtype,
                                        device=self.dense.weight.device)
        self.grow_mask_vec[:] = CONSTANT_MAX_MASK_VAL

    def set_mask(self, target_dim, val):
        if target_dim is not None:
            new_mask = torch.ones(target_dim, dtype=self.dense.weight.dtype)
            new_mask[self.old_hidden_size:] = val
            self.grow_mask_vec = new_mask
            self.grow_mask_vec = self.grow_mask_vec.to(self.dense.weight.device)
        else:
            self.grow_mask_vec[self.old_hidden_size:] = val

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states, self.grow_mask_vec, self.in_growth)
        
        if self.in_growth:
            hidden_states = hidden_states * self.grow_mask_vec
        return hidden_states


class BertLMPredictionHeadEx(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transform = BertPredictionHeadTransformEx(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias


    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeadsEx(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHeadEx(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

        self.hidden_size = config.hidden_size
        self.old_hidden_size = config.hidden_size
        self.in_growth = False
        self.debug_print = False

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        
        return prediction_scores, seq_relationship_score

class BertModelEx(BertModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsEx(config)
        self.encoder = BertEncoderEx(config)

        self.pooler = BertPoolerEx(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


class BertForPreTrainingEx(BertForPreTraining):
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)

        self.bert = BertModelEx(config)
        self.cls = BertPreTrainingHeadsEx(config)

        # Initialize weights and apply final processing
        self.post_init()

class BertForSequenceClassificationEx(BertForSequenceClassification):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModelEx(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()
