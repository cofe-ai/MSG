MAX_MASK_VAL = 1.0
from accelerate.utils.other import extract_model_from_parallel
import torch

import sys
sys.path.append("..")
from .copy_weight_ex import (grow_embedding, grow_dim_self_att, grow_head_num, grow_dim_self_output,
                            grow_dim_intermediate, grow_dim_output, grow_block, stack_block, grow_dim_pooler,
                            grow_dim_transform, grow_dim_nsp, vanilla_copy)
from .utils_ex_v2 import transfer_states, count_parameters

class grow_ops(object):
    def __init__(self, model):
        model = extract_model_from_parallel(model)
        self.config = model.config
        self.available_to_grow = True
        self.curr_mode = None

        self.step_size_dict = {"hidden_size": 0.,
        "intermediate_size": 0.,
        "heads": 0.,
        "layers": 0.}

        self.temp_mask_dict = {"hidden_size": 0.,
        "intermediate_size": 0.,
        "heads": 0.,
        "layers": 0.
        }

    def mask_to_gpu(self, model):
        model = extract_model_from_parallel(model)
        for m in model.modules():
            if hasattr(m, "grow_mask_head") and m.grow_mask_head is not None:
                m.grow_mask_head = m.grow_mask_head.to(model.bert.embeddings.word_embeddings.weight.device)
            if hasattr(m, "grow_mask_vec") and m.grow_mask_vec is not None:
                m.grow_mask_vec = m.grow_mask_vec.to(model.bert.embeddings.word_embeddings.weight.device)
            if hasattr(m, "grow_mask_block") and m.grow_mask_block is not None:
                m.grow_mask_block = m.grow_mask_block.to(model.bert.embeddings.word_embeddings.weight.device)


    def set_grow(self, m1, m2, mode, target, steps, opt1, opt2, args):
        
        if not self.available_to_grow:
            raise ValueError("grow locked by other type, please check.")
        
        m1 = extract_model_from_parallel(m1)

        m1.requires_grad_(False)

        self.step_size_dict[mode] = MAX_MASK_VAL / steps

        if mode == "hidden_size":
            # grow embeddings
            assert m1.bert.embeddings.word_embeddings.weight is m1.cls.predictions.decoder.weight

            grow_embedding(m1.bert.embeddings, m2.bert.embeddings, target, args)
            # grow encoder layers
            for l1, l2 in zip(m1.bert.encoder.layer, m2.bert.encoder.layer):
                grow_dim_self_att(l1.attention.self, l2.attention.self, target, args)
                grow_dim_self_output(l1.attention.output, l2.attention.output, target, l1.attention.self.all_head_size, args)
                grow_dim_intermediate(l1.intermediate, l2.intermediate, l1.intermediate.intermediate_size, target, args)
                grow_dim_output(l1.output, l2.output, target, l1.intermediate.intermediate_size, args)
            # grow pooler
            if m2.bert.pooler:
                grow_dim_pooler(m1.bert.pooler, m2.bert.pooler, target, args)
            # grow pretrain head
            grow_dim_transform(m1.cls.predictions.transform, m2.cls.predictions.transform, target, args)
            m2.cls.predictions.decoder.weight = m2.bert.embeddings.word_embeddings.weight
            grow_dim_nsp(m1.cls, m2.cls, target, args)

        elif mode == "intermediate_size":
            vanilla_copy(m1.bert.embeddings, m2.bert.embeddings)

            for l1, l2 in zip(m1.bert.encoder.layer, m2.bert.encoder.layer):
                vanilla_copy(l1.attention.self, l2.attention.self)
                vanilla_copy(l1.attention.output, l2.attention.output)
                grow_dim_intermediate(l1.intermediate, l2.intermediate, target, l1.intermediate.hidden_size, args)
                grow_dim_output(l1.output, l2.output, l1.output.hidden_size, target, args)
            if m2.bert.pooler:
                vanilla_copy(m1.bert.pooler, m2.bert.pooler)
            
            vanilla_copy(m1.cls, m2.cls)
            vanilla_copy(m1.cls.predictions.transform, m2.cls.predictions.transform)
            m2.cls.predictions.decoder.weight = m2.bert.embeddings.word_embeddings.weight
            
        elif mode == "heads":
            vanilla_copy(m1.bert.embeddings, m2.bert.embeddings)

            for l1, l2 in zip(m1.bert.encoder.layer, m2.bert.encoder.layer):
                grow_head_num(l1.attention.self, l2.attention.self, target, args)
                grow_dim_self_output(l1.attention.output, l2.attention.output, 
                                     l1.attention.output.hidden_size,
                                     l2.attention.self.all_head_size, args)
                vanilla_copy(l1.intermediate, l2.intermediate)
                vanilla_copy(l1.output, l2.output)

            if m2.bert.pooler:
                vanilla_copy(m1.bert.pooler, m2.bert.pooler)
            
            vanilla_copy(m1.cls, m2.cls)
            vanilla_copy(m1.cls.predictions.transform, m2.cls.predictions.transform)
            m2.cls.predictions.decoder.weight = m2.bert.embeddings.word_embeddings.weight

        elif mode == "layers":
            vanilla_copy(m1, m2)
            param_list_1 = list(m1.parameters())
            param_list_2 = list(m2.parameters())
            assert len(param_list_1) == len(param_list_2)
            for p1, p2 in zip(param_list_1, param_list_2):
                transfer_states(p1, p2, opt1, opt2)

            if hasattr(args, "grow_init_strategy") and args.grow_init_strategy == "random-interpolate":
                # Interpolate strategy
                if target > 2 * len(m2.bert.encoder.layer):
                    extra_pos = [target for _ in range(target - 2 * len(m2.bert.encoder.layer))]
                else:
                    extra_pos = []
                interpolate_pos = list(range(len(m2.bert.encoder.layer), 0, -1))
                m2.bert.encoder.update_config()
                for pos in (extra_pos + interpolate_pos):
                    if len(m2.bert.encoder.layer) >= target:
                        break
                    grow_block(m2.bert.encoder, pos, args)

                # print([l.is_new_block for l in m2.bert.encoder.layer])
                m2.bert.encoder.dynamic_config.num_hidden_layers = target
            
            elif (not hasattr(args, "grow_init_strategy")) or args.grow_init_strategy is None \
                  or args.grow_init_strategy == "random":
                sm_layers = len(m2.bert.encoder.layer)
                sm_layer_idxs = [i for i in range(sm_layers)]
                sm_layer_idx_for_bert2bert_top = []
                n_times = target // sm_layers - 1
                sm_layer_idx_for_bert2bert_top.extend(sm_layer_idxs * n_times)
                top_layers = target % sm_layers
                if top_layers !=0:
                    sm_layer_idx_for_bert2bert_top.extend(sm_layer_idxs[-top_layers:])
                print(sm_layer_idx_for_bert2bert_top)

                m2.bert.encoder.update_config()
                for pos in sm_layer_idx_for_bert2bert_top:
                    stack_block(m2.bert.encoder, pos, args)
                print([l.is_new_block for l in m2.bert.encoder.layer])
                m2.bert.encoder.dynamic_config.num_hidden_layers = target
            else:
                raise ValueError("Unsupported grow strategy for layers")

        else:
            raise ValueError("Unsupported grow mode choice.")
        
        if mode != "layers":
            param_list_1 = list(m1.parameters())
            param_list_2 = list(m2.parameters())

            assert len(param_list_1) == len(param_list_2)
            for p1, p2 in zip(param_list_1, param_list_2):
                transfer_states(p1, p2, opt1, opt2)
        else:
            for l in m2.bert.encoder.layer:
                if l.is_new_block:
                    no_decay = ["bias", "LayerNorm.weight"]
                    optimizer_grouped_parameters = [
                        {"params": [p for n, p in l.named_parameters() if not any(nd in n for nd in no_decay)],
                         "weight_decay": args.weight_decay,},
                        {"params": [p for n, p in l.named_parameters() if any(nd in n for nd in no_decay)],
                         "weight_decay": 0.0,},
                    ]
                    for g in optimizer_grouped_parameters:
                        opt2.add_param_group(g)

        m2.requires_grad_(True)
        self.available_to_grow = False
        self.curr_mode = mode

    def end_grow(self, model, mode):
        if mode is None:
            if self.curr_mode is None:
                raise ValueError("No growing in process")
            mode = self.curr_mode
        else:
            if mode != self.curr_mode:
                raise ValueError("The growth process doesn't match")
        model = extract_model_from_parallel(model)
        self.step_size_dict[mode] = 0.
        self.temp_mask_dict[mode] = 0.
        if mode == "hidden_size":
            # embeddings
            model.bert.embeddings.in_growth = False
            model.bert.embeddings.set_mask(None, MAX_MASK_VAL)
            # encoder
            for single_layer in model.bert.encoder.layer:
                single_layer.attention.self.in_grow_dim = False
                single_layer.attention.output.in_grow_dim = False
                single_layer.attention.output.set_mask(None, MAX_MASK_VAL)
                single_layer.intermediate.in_grow_input = False
                single_layer.output.in_grow_dim = False
                single_layer.output.set_mask(None, MAX_MASK_VAL)
            # pooler
            if model.bert.pooler:
                model.bert.pooler.in_growth = False
                model.bert.pooler.set_mask(None, MAX_MASK_VAL)
            # head
            model.cls.predictions.transform.in_growth = False
            model.cls.predictions.transform.set_mask(None, MAX_MASK_VAL)
            model.cls.predictions.in_growth = False
            model.cls.in_growth = False

        elif mode == "intermediate_size":
            for single_layer in model.bert.encoder.layer:
                single_layer.intermediate.in_grow_dim = False
                single_layer.intermediate.set_mask(None, MAX_MASK_VAL)
                single_layer.output.in_grow_input = False
        
        elif mode == "heads":
            for single_layer in model.bert.encoder.layer:
                single_layer.attention.self.in_grow_head = False
                single_layer.attention.self.set_mask_head(None, MAX_MASK_VAL)
                single_layer.attention.output.in_grow_input = False
        
        elif mode == "layers":
            for single_layer in model.bert.encoder.layer:
                single_layer.is_new_block = False
            model.bert.encoder.in_growth = False
            model.bert.encoder.set_mask(MAX_MASK_VAL)
        
        else:
            raise ValueError("Unsupported end growth mode.")
        
        self.available_to_grow = True
        self.curr_mode = None
        
    def increase_mask(self, model, mode):
        if mode is None:
            if self.curr_mode is None:
                raise ValueError("No growing in process")
            mode = self.curr_mode
            
        if mode != self.curr_mode:
            raise ValueError("Can not increase mask because mode don't match")

        model = extract_model_from_parallel(model)
        self.temp_mask_dict[mode] += self.step_size_dict[mode]
        new_val = self.temp_mask_dict[mode]
        if mode == "hidden_size":
            model.bert.embeddings.set_mask(None, new_val)
            for single_layer in model.bert.encoder.layer:
                single_layer.attention.output.set_mask(None, new_val)
                single_layer.output.set_mask(None, new_val)
            if model.bert.pooler:
                model.bert.pooler.set_mask(None, new_val)
            model.cls.predictions.transform.set_mask(None, new_val)
        elif mode == "intermediate_size":
            for single_layer in model.bert.encoder.layer:
                single_layer.intermediate.set_mask(None, new_val)
        elif mode == "heads":
            for single_layer in model.bert.encoder.layer:
                single_layer.attention.self.set_mask_head(None, new_val)
        elif mode == "layers":
            model.bert.encoder.set_mask(new_val)
        else:
            raise ValueError("Unsupported end growth mode.")

    def print_all_flags(self, model):
        model = extract_model_from_parallel(model)
        if model.bert.embeddings.word_embeddings.weight.device == torch.device(type="cuda",index=0):
            all_flags = []
            all_flags.append([model.bert.embeddings.in_growth])
            for layer in model.bert.encoder.layer:
                all_flags.append([layer.is_new_block, layer.attention.self.in_grow_dim,
                layer.attention.self.in_grow_head, layer.attention.output.in_grow_dim, 
                layer.intermediate.in_grow_dim, layer.output.in_grow_dim])
            all_flags.append([model.bert.encoder.in_growth])
            all_flags.append([model.bert.pooler.in_growth])
            all_flags.append([model.cls.predictions.transform.in_growth, model.cls.in_growth])
            print(all_flags)

    def print_all_masks(self, model):
        def n(tensor):
            return tensor[-1].item() if tensor is not None else None

        model = extract_model_from_parallel(model)
        if model.bert.embeddings.word_embeddings.weight.device == torch.device(type="cuda",index=0):
            all_masks = []
            all_masks.append([n(model.bert.embeddings.grow_mask_vec)])
            for layer in model.bert.encoder.layer:
                all_masks.append([n(layer.attention.self.grow_mask_head),
                n(layer.attention.output.grow_mask_vec), 
                n(layer.intermediate.grow_mask_vec), n(layer.output.grow_mask_vec)])
            all_masks.append([n(model.bert.encoder.grow_mask_block)])
            all_masks.append([n(model.bert.pooler.grow_mask_vec)])
            all_masks.append([n(model.cls.predictions.transform.grow_mask_vec)])
            print(all_masks)

    def count_parameters(self, model):
        return count_parameters(model)