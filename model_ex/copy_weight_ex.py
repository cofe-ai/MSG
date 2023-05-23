# coding=utf-8
"""Grow operations for all modules in Bert."""

import torch
import torch.utils.checkpoint
from torch import nn

from .utils_ex_v2 import expand_linear, expand_embedding, expand_norm, init_weight_ex
from .bert_ex_v2 import (BertEmbeddingsEx, BertSelfAttentionEx, BertSelfOutputEx, 
                         BertIntermediateEx, BertOutputEx, BertLayerEx, BertEncoderEx, BertPoolerEx,
                         BertPredictionHeadTransformEx, BertPreTrainingHeadsEx)

CONSTANT_ATTENTION_HEAD_SIZE = 64
CONSTANT_MIN_MASK_VAL = 0.

def grow_embedding(old_module, new_module, target_dim, args):
    assert isinstance(old_module, BertEmbeddingsEx) and isinstance(new_module, BertEmbeddingsEx)

    new_module.in_growth = True
    # expand weights
    word_embeddings_new = expand_embedding(old_module.word_embeddings, target_dim, old_module.config.vocab_size, args)
    position_embeddings_new = expand_embedding(old_module.position_embeddings, target_dim, 
                                                old_module.config.max_position_embeddings, args)
    token_type_embeddings_new = expand_embedding(old_module.token_type_embeddings, target_dim, 
                                                old_module.config.type_vocab_size, args)

    LayerNorm_new = expand_norm(old_module.LayerNorm, target_dim, args)

    # ensure torch.no_grad(), done in load_state_dict
    new_module.word_embeddings.load_state_dict(word_embeddings_new.state_dict(),strict=True)
    new_module.position_embeddings.load_state_dict(position_embeddings_new.state_dict(),strict=True)
    new_module.token_type_embeddings.load_state_dict(token_type_embeddings_new.state_dict(),strict=True)
    new_module.LayerNorm.load_state_dict(LayerNorm_new.state_dict(),strict=True)

    # update hidden size
    new_module.old_hidden_size = old_module.hidden_size
    new_module.hidden_size = target_dim

    # create zero mask
    new_module.set_mask(target_dim, val=CONSTANT_MIN_MASK_VAL)

    # update hidden size
    new_module.old_hidden_size = old_module.hidden_size
    new_module.hidden_size = target_dim


def grow_dim_self_att(old_module, new_module, target_size, args):
    assert isinstance(old_module, BertSelfAttentionEx) and isinstance(new_module, BertSelfAttentionEx)
    new_module.in_grow_dim = True
    assert target_size > old_module.hidden_size
    # grow Q, K, V
    extra = {"dim_setting": "d2h", "attention_head_size": old_module.attention_head_size}
    query_new = expand_linear(old_module.query, old_module.all_head_size, target_size, args, extra)
    key_new = expand_linear(old_module.key, old_module.all_head_size, target_size, args, extra)
    value_new = expand_linear(old_module.value, old_module.all_head_size, target_size, args, extra)
    # no need to create mask because only input dim is expanded.

    new_module.query.load_state_dict(query_new.state_dict(),strict=True)
    new_module.key.load_state_dict(key_new.state_dict(),strict=True)
    new_module.value.load_state_dict(value_new.state_dict(),strict=True)
    new_module.hidden_size = target_size


def grow_head_num(old_module, new_module, target_head_num, args):
    assert isinstance(old_module, BertSelfAttentionEx) and isinstance(new_module, BertSelfAttentionEx)
    new_module.in_grow_head = True
    # We always keep self.attention_head_size and just grow the head numbers
    new_head_count = target_head_num - old_module.num_attention_heads
    assert new_head_count > 0

    new_module.num_attention_heads = target_head_num
    new_module.new_head_count = new_head_count
    
    new_module.all_head_size = target_head_num * new_module.attention_head_size
    
    # grow Q, K, V
    extra = {"dim_setting": "d2h", "attention_head_size": old_module.attention_head_size}
    query_new = expand_linear(old_module.query, new_module.all_head_size, old_module.hidden_size, args, extra)
    key_new = expand_linear(old_module.key, new_module.all_head_size, old_module.hidden_size, args, extra)
    value_new = expand_linear(old_module.value, new_module.all_head_size, old_module.hidden_size, args, extra)

    new_module.query.load_state_dict(query_new.state_dict(),strict=True)
    new_module.key.load_state_dict(key_new.state_dict(),strict=True)
    new_module.value.load_state_dict(value_new.state_dict(),strict=True)

    new_module.set_mask_head(target_head_num, val=CONSTANT_MIN_MASK_VAL)


def grow_dim_self_output(old_module, new_module, target_size, all_head_size, args):
    assert isinstance(old_module, BertSelfOutputEx) and isinstance(new_module, BertSelfOutputEx)
    if target_size > old_module.hidden_size:
        new_module.in_grow_dim = True
    if all_head_size > old_module.all_head_size:
        new_module.in_grow_input = True
    # grow dense and layernorm
    extra = {"dim_setting": "h2d", "attention_head_size": CONSTANT_ATTENTION_HEAD_SIZE}
    dense_new = expand_linear(old_module.dense, target_size, all_head_size, args, extra)
    new_module.dense.load_state_dict(dense_new.state_dict(),strict=True)
    if new_module.in_grow_dim:
        LayerNorm_new = expand_norm(old_module.LayerNorm, target_size, args)
        new_module.LayerNorm.load_state_dict(LayerNorm_new.state_dict(),strict=True)
        new_module.old_hidden_size = old_module.hidden_size
        new_module.hidden_size = target_size
        new_module.set_mask(target_size, val=CONSTANT_MIN_MASK_VAL)
    else:
        new_module.LayerNorm.load_state_dict(old_module.LayerNorm.state_dict(),strict=True)
        
    new_module.all_head_size = all_head_size


def grow_dim_intermediate(old_module, new_module, target_intermediate_size, target_input_size, args):
    assert isinstance(old_module, BertIntermediateEx) and isinstance(new_module, BertIntermediateEx)

    if target_intermediate_size > old_module.intermediate_size:
        new_module.in_grow_dim = True
    if target_input_size > old_module.hidden_size:
        new_module.in_grow_input = True

    # grow dense and layernorm
    extra = {"dim_setting": "d2i"}
    dense_new = expand_linear(old_module.dense, target_intermediate_size, target_input_size, args, extra)
    new_module.dense.load_state_dict(dense_new.state_dict(),strict=True)
    if new_module.in_grow_dim:
        new_module.old_intermediate_size = old_module.intermediate_size
        new_module.intermediate_size = target_intermediate_size
        new_module.set_mask(target_intermediate_size, val=CONSTANT_MIN_MASK_VAL)

    new_module.hidden_size = target_input_size


def grow_dim_output(old_module, new_module, target_hidden_size, target_input_size, args):
    assert isinstance(old_module, BertOutputEx) and isinstance(new_module, BertOutputEx)
    if target_hidden_size > old_module.hidden_size:
        new_module.in_grow_dim = True
    if target_input_size > old_module.intermediate_size:
        new_module.in_grow_input = True

    # grow dense and layernorm
    extra = {"dim_setting": "i2d"}
    dense_new = expand_linear(old_module.dense, target_hidden_size, target_input_size, args, extra)
    new_module.dense.load_state_dict(dense_new.state_dict(),strict=True)
    if new_module.in_grow_dim:
        LayerNorm_new = expand_norm(old_module.LayerNorm, target_hidden_size, args)
        new_module.LayerNorm.load_state_dict(LayerNorm_new.state_dict(),strict=True)
        new_module.old_hidden_size = old_module.hidden_size
        new_module.hidden_size = target_hidden_size
        new_module.set_mask(target_hidden_size, val=CONSTANT_MIN_MASK_VAL)
    else:
        new_module.LayerNorm.load_state_dict(old_module.LayerNorm.state_dict(),strict=True)
        
        
    new_module.intermediate_size = target_input_size


def grow_block(new_module, position, args):
    assert isinstance(new_module, BertEncoderEx)
    if position > len(new_module.layer):
        pos = len(new_module.layer)
    else:
        pos = position
    assert pos > 0
    new_module.num_hidden_layers += 1
    new_layer = BertLayerEx(new_module.dynamic_config)
    if hasattr(args, "new_block_init_strategy") and args.new_block_init_strategy == "random":
        for m in new_layer.modules():
            init_weight_ex(m)
    else:
        new_layer.load_state_dict(state_dict=new_module.layer[pos-1].state_dict(), strict=True)
    # new_layer = new_layer.to(self.layer[0].output.dense.weight.device)
    new_layer.is_new_block = True
    new_module.layer.insert(pos, new_layer)
    if new_module.grow_mask_block is None or new_module.dynamic_config.hidden_size > new_module.grow_mask_block.size()[0]:
        new_module.grow_mask_block = torch.zeros(new_module.dynamic_config.hidden_size, dtype=new_module.layer[0].output.dense.weight.dtype)
        new_module.set_mask(CONSTANT_MIN_MASK_VAL)
        # new_module.grow_mask_block = new_module.grow_mask_block.to(new_module.layer[0].output.dense.weight.device)
    else:
        new_module.set_mask(CONSTANT_MIN_MASK_VAL)

    new_module.in_growth = True

def stack_block(new_module, from_pos, args):
    assert isinstance(new_module, BertEncoderEx)
    assert from_pos >= 0
    new_module.num_hidden_layers += 1
    new_layer = BertLayerEx(new_module.dynamic_config)
    if hasattr(args, "new_block_init_strategy") and args.new_block_init_strategy == "random":
        for m in new_layer.modules():
            init_weight_ex(m)
    else:
        new_layer.load_state_dict(state_dict=new_module.layer[from_pos].state_dict(), strict=True)
    # new_layer = new_layer.to(self.layer[0].output.dense.weight.device)
    new_layer.is_new_block = True
    new_module.layer.append(new_layer)
    if new_module.grow_mask_block is None or new_module.dynamic_config.hidden_size > new_module.grow_mask_block.size()[0]:
        new_module.grow_mask_block = torch.zeros(new_module.dynamic_config.hidden_size, dtype=new_module.layer[0].output.dense.weight.dtype)
        new_module.set_mask(CONSTANT_MIN_MASK_VAL)
        # self.grow_mask_block = self.grow_mask_block.to(self.layer[0].output.dense.weight.device)
    else:
        new_module.set_mask(CONSTANT_MIN_MASK_VAL)
    new_module.in_growth = True

def grow_dim_pooler(old_module, new_module, target_hidden_size, args):
    assert isinstance(old_module, BertPoolerEx) and isinstance(new_module, BertPoolerEx)
    if target_hidden_size > old_module.hidden_size:
        new_module.in_growth = True

    dense_new = expand_linear(old_module.dense, target_hidden_size, target_hidden_size, args)
    new_module.dense.load_state_dict(dense_new.state_dict(),strict=True)
    new_module.old_hidden_size = old_module.hidden_size
    new_module.hidden_size = target_hidden_size
    new_module.set_mask(target_hidden_size, val=CONSTANT_MIN_MASK_VAL)
    

def grow_dim_transform(old_module, new_module, target_hidden_size, args):
    assert isinstance(old_module, BertPredictionHeadTransformEx) and isinstance(new_module, BertPredictionHeadTransformEx)
    if target_hidden_size > old_module.hidden_size:
        new_module.in_growth = True

    dense_new = expand_linear(old_module.dense, target_hidden_size, target_hidden_size, args)
    new_module.dense.load_state_dict(dense_new.state_dict(),strict=True)
    LayerNorm_new = expand_norm(old_module.LayerNorm, target_hidden_size, args)
    new_module.LayerNorm.load_state_dict(LayerNorm_new.state_dict(),strict=True)
    new_module.old_hidden_size = old_module.hidden_size
    new_module.hidden_size = target_hidden_size
    new_module.set_mask(target_hidden_size, val=CONSTANT_MIN_MASK_VAL)
    

def grow_dim_nsp(old_module, new_module, target_hidden_size, args):
    assert isinstance(old_module, BertPreTrainingHeadsEx) and isinstance(new_module, BertPreTrainingHeadsEx)
    if target_hidden_size > old_module.hidden_size:
        new_module.in_growth = True

    extra = {"dim_setting": "other"}
    seq_relationship_new = expand_linear(old_module.seq_relationship, 2, target_hidden_size, args, extra)
    new_module.old_hidden_size = old_module.hidden_size
    new_module.hidden_size = target_hidden_size
    new_module.seq_relationship.load_state_dict(seq_relationship_new.state_dict(),strict=True)

def vanilla_copy(old_module, new_module):
    assert type(old_module) == type(new_module)
    new_module.load_state_dict(old_module.state_dict(), strict=True)