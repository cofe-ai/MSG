import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR 
from accelerate.utils.other import extract_model_from_parallel
import warnings

class LayerNormEx(torch.nn.LayerNorm):    
    def forward(self, input_vec, ln_mask, in_growth=False, debug_print=False):
        if not in_growth:
            assert ln_mask is not None
            weighted_mean = torch.mean(input_vec, dim=-1, keepdim=True)
            weighted_var = torch.mean((torch.square(input_vec - weighted_mean)), dim=-1, keepdim=True)
            output = (input_vec - weighted_mean) / torch.sqrt(weighted_var + self.eps)
            output = output * self.weight + self.bias
            return output
        else:
            weighted_mean = (torch.sum(input_vec * ln_mask, dim=-1, keepdim=True)) / torch.tensor(torch.sum(ln_mask).item())
            weighted_var = torch.sum((torch.square(input_vec - weighted_mean) * ln_mask), dim=-1, keepdim=True) / torch.tensor(torch.sum(ln_mask).item())
            output = (input_vec - weighted_mean) / torch.sqrt(weighted_var + self.eps)
            output = output * self.weight + self.bias
            return output

class LambdaLR_ex(LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        super(LambdaLR_ex, self).__init__(optimizer, lr_lambda, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        if len(self.optimizer.param_groups) > len(self.base_lrs):
            self.base_lrs = [self.base_lrs[0] for _ in range(len(self.optimizer.param_groups))]
            self.lr_lambdas = [self.lr_lambdas[0] for _ in range(len(self.optimizer.param_groups))]

        return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]

def get_scheduler_ex(name, optimizer, num_warmup_steps, num_training_steps,
                     rewind_bool=False, rewind_step_1=None, rewind_step_2=None, last_epoch=-1):
    def lr_lambda(current_step: int):
        if not rewind_bool:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        else:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif num_warmup_steps <= current_step < rewind_step_1:
                return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
                )
            else:
                if rewind_step_2 is None or rewind_step_1 <= current_step < rewind_step_2:
                    return max(
                    0.0, float(num_training_steps - current_step + rewind_step_1 - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                    )
                else:
                    return max(
                    0.0, float(num_training_steps - current_step + rewind_step_2 - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                    )

    return LambdaLR_ex(optimizer, lr_lambda, last_epoch)

def count_parameters(model):
    model = extract_model_from_parallel(model)
    # removed "if p.requires_grad"
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_total_norm(model):
    model = extract_model_from_parallel(model)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2).to(model.device) for g in grads]), 2)
    return total_norm

def init_weight_ex(module, initializer_range=0.02, reference_module=None):
    if reference_module is not None:
        if isinstance(module, nn.Linear):
            std_w, mean_w = torch.std_mean(reference_module.weight)
            std_b, mean_b = torch.std_mean(reference_module.bias)
            std_w, mean_w, std_b, mean_b = std_w.item(), mean_w.item(), std_b.item(), mean_b.item()
            module.weight.data.normal_(mean=mean_w, std=std_w)
            if module.bias is not None:
                module.bias.data.normal_(mean=mean_b, std=std_b)

        elif isinstance(module, nn.Embedding):
            std_w, mean_w = torch.std_mean(reference_module.weight)
            std_w, mean_w = std_w.item(), mean_w.item()
            module.weight.data.normal_(mean=mean_w, std=std_w)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, LayerNormEx):
            std_w, mean_w = torch.std_mean(reference_module.weight)
            std_b, mean_b = torch.std_mean(reference_module.bias)
            std_w, mean_w, std_b, mean_b = std_w.item(), mean_w.item(), std_b.item(), mean_b.item()
            module.bias.data.normal_(mean=mean_b, std=std_b)
            module.weight.data.normal_(mean=mean_w, std=std_w)
    
    else:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, LayerNormEx) or isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def transfer_states(old_param, new_param, opt_old, opt_new):
    if len(old_param.size()) == 2:
        old_1, old_2 = old_param.size()
    else:
        old_1 = old_param.size()[0]

    opt_old = opt_old.optimizer
    assert opt_old.state[old_param] is not None and opt_new.state[new_param] is not None

    new_param_state = {}
    new_param_state["step"] = opt_old.state[old_param]["step"]

    new_param_state["exp_avg"] = torch.zeros_like(new_param, memory_format=torch.preserve_format)
    if len(old_param.size()) == 2:
        new_param_state["exp_avg"][:old_1, :old_2] = opt_old.state[old_param]["exp_avg"][:,:]
    else:
        new_param_state["exp_avg"][:old_1] = opt_old.state[old_param]["exp_avg"][:]
    new_param_state["exp_avg_sq"] = torch.zeros_like(new_param, memory_format=torch.preserve_format)
    if len(old_param.size()) == 2:
        new_param_state["exp_avg_sq"][:old_1, :old_2] = opt_old.state[old_param]["exp_avg_sq"][:,:]
    else:
        new_param_state["exp_avg_sq"][:old_1] = opt_old.state[old_param]["exp_avg_sq"][:]
    opt_new.state[new_param] = new_param_state
    # del opt_old.state[old_param]
    old_param.detach_()

def expand_embedding(old_embedding, new_in_num, vocab_size, args):
    old_vocab_size, old_in_num = old_embedding.weight.size()

    assert old_vocab_size == vocab_size
    if old_in_num > new_in_num:
        raise ValueError("New embedding smaller than old")
        
    new_emb = nn.Embedding(vocab_size, new_in_num)

    init_weight_ex(new_emb, reference_module=None)
    # copy weights
    new_emb.weight.data[:, :old_in_num] = old_embedding.weight.data[:, :]
    
    return new_emb


def expand_linear(old_linear, new_out_num, new_in_num, args, extra=None):

    old_out_num, old_in_num = old_linear.weight.size()

    if old_in_num > new_in_num or old_out_num > new_out_num:
        raise ValueError("New Linear smaller than old")
    
    has_bias = old_linear.bias is not None
    new_linear = nn.Linear(new_in_num, new_out_num, bias=has_bias)

    init_weight_ex(new_linear, reference_module=None)

    # copy weights
    new_linear.weight.data[:old_out_num, :old_in_num] = old_linear.weight.data[:,:]
    if has_bias:
        new_linear.bias.data[:old_out_num] = old_linear.bias.data[:]
    
    return new_linear


def expand_norm(old_ln, new_out_num, args):

    old_out_num = old_ln.weight.size()[0]

    if old_out_num >= new_out_num:
        raise ValueError("New LN smaller than or equal to old")

    new_ln = LayerNormEx(new_out_num, eps=1e-12)

    init_weight_ex(new_ln, reference_module=None)

    # copy weights
    new_ln.weight.data[:old_out_num] = old_ln.weight.data[:]
    new_ln.bias.data[:old_out_num] = old_ln.bias.data[:]

    return new_ln
