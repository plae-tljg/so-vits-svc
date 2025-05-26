import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, g, dim):
        # 更严格的数值检查
        if torch.isnan(v).any() or torch.isinf(v).any():
            print("Warning: Weight contains NaN/Inf values")
            v = torch.nan_to_num(v, nan=0.0, posinf=0.05, neginf=-0.05)
            
        if torch.isnan(g).any() or torch.isinf(g).any():
            print("Warning: Gain contains NaN/Inf values")
            g = torch.nan_to_num(g, nan=1.0, posinf=1.0, neginf=1.0)
        
        # 使用更稳定的范数计算
        norm = torch.norm(v, dim=dim, keepdim=True)
        norm = torch.clamp(norm, min=1e-10)  # 使用更小的最小值
        v_normalized = v / norm
        
        # 对增益进行更严格的范围限制
        g = torch.clamp(g, min=0.05, max=5.0)  # 更保守的范围
        w = v_normalized * g
        
        # 对最终权重进行裁剪
        w = torch.clamp(w, min=-0.1, max=0.1)
        
        # 保存反向传播需要的张量
        ctx.save_for_backward(v, g, norm)
        ctx.dim = dim
        
        return w
    
    @staticmethod
    def backward(ctx, grad_output):
        v, g, norm = ctx.saved_tensors
        dim = ctx.dim
        
        # 更严格的梯度检查
        if torch.isnan(grad_output).any() or torch.isinf(grad_output).any():
            print("Warning: Gradient contains NaN/Inf values")
            grad_output = torch.nan_to_num(grad_output, nan=0.0, posinf=0.05, neginf=-0.05)
        
        # 使用更稳定的梯度计算
        v_normalized = v / norm
        
        # 计算 v 的梯度
        grad_v = grad_output * g
        grad_v = grad_v - v_normalized * torch.sum(grad_v * v_normalized, dim=dim, keepdim=True)
        
        # 计算 g 的梯度
        grad_g = torch.sum(grad_output * v_normalized, dim=dim)
        
        # 更严格的梯度裁剪
        grad_v = torch.clamp(grad_v, min=-0.05, max=0.05)
        grad_g = torch.clamp(grad_g, min=-0.05, max=0.05)
        
        return grad_v, grad_g, None

class WeightNorm(nn.Module):
    def __init__(self, module, name='weight', dim=0):
        super(WeightNorm, self).__init__()
        self.module = module
        self.name = name
        self.dim = dim
        
        # 获取原始权重
        w = getattr(module, name)
        del module._parameters[name]
        
        # 使用更保守的初始化方法
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            # 使用更小的初始化范围
            nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='linear')
            w = torch.clamp(w, min=-0.05, max=0.05)  # 更保守的范围
        else:
            nn.init.normal_(w, mean=0.0, std=0.005)  # 更小的标准差
            w = torch.clamp(w, min=-0.05, max=0.05)
        
        # 计算范数并归一化
        norm = torch.norm(w, dim=dim, keepdim=True)
        norm = torch.clamp(norm, min=1e-10)
        v = w / norm
        
        # 初始化增益为 1.0,并添加更小的随机扰动
        g = torch.ones_like(norm.squeeze(dim)) * (1.0 + 0.005 * torch.randn_like(norm.squeeze(dim)))
        g = torch.clamp(g, min=0.05, max=5.0)  # 更保守的范围
        
        # 注册参数
        module.register_parameter(name + '_v', nn.Parameter(v))
        module.register_parameter(name + '_g', nn.Parameter(g))
        
        # 设置钩子
        module.register_forward_pre_hook(self.forward_pre_hook)
        
    def forward_pre_hook(self, module, input):
        """前向传播前的钩子函数"""
        v = getattr(module, self.name + '_v')
        g = getattr(module, self.name + '_g')
        
        # 更严格的数值检查
        if torch.isnan(v).any() or torch.isinf(v).any():
            print(f"Warning: Weight contains NaN/Inf in {module.__class__.__name__} layer")
            nn.init.kaiming_normal_(v, mode='fan_in', nonlinearity='linear')
            v = torch.clamp(v, min=-0.05, max=0.05)
            
        if torch.isnan(g).any() or torch.isinf(g).any():
            print(f"Warning: Gain contains NaN/Inf in {module.__class__.__name__} layer")
            g = torch.ones_like(g) * (1.0 + 0.005 * torch.randn_like(g))
            g = torch.clamp(g, min=0.05, max=5.0)
        
        # 计算权重
        w = WeightNormFunction.apply(v, g, self.dim)
        setattr(module, self.name, w)
        
    @staticmethod
    def apply(module, name, dim):
        """应用 weight_norm 到模块"""
        fn = WeightNorm(module, name, dim)
        return fn

def weight_norm(module, name='weight', dim=0):
    """weight_norm 函数接口"""
    WeightNorm.apply(module, name, dim)
    return module

def remove_weight_norm(module, name='weight'):
    """移除 weight_norm"""
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError(f"weight_norm of '{name}' not found in {module}")

# 预定义的带 weight_norm 的层
class WeightNormConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(WeightNormConv1d, self).__init__(*args, **kwargs)
        weight_norm(self)

class WeightNormConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(WeightNormConv2d, self).__init__(*args, **kwargs)
        weight_norm(self)

class WeightNormLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(WeightNormLinear, self).__init__(*args, **kwargs)
        weight_norm(self)

class WeightNormConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super(WeightNormConvTranspose1d, self).__init__(*args, **kwargs)
        weight_norm(self) 