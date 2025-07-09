import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from timm.models.layers import trunc_normal_, DropPath
from torchvision import transforms
import timm

FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)


class Block(nn.Module):
    """
    ConvNeXt blokining oddiy implementatsiyasi.
    
    Args:
        dim (int): Kiruvchi kanallar soni.
        drop_path (float): DropPath (stochastic depth) ehtimoli.
        layer_scale_init_value (float): Layer scale uchun boshlang‘ich qiymat.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Blok bo‘ylab forward pass.

        Args:
            x (Tensor): Kiruvchi tensor (N, C, H, W)

        Returns:
            Tensor: Chiquvchi tensor.
        """
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return input + self.drop_path(x)


class ConvNeXt(nn.Module):
    """
    ConvNeXt arxitekturasi: Vision uchun convolutional tarmoq.

    Args:
        in_chans (int): Rasm kirishi kanallari soni.
        num_classes (int): Klassifikatsiya chiqish klasslari soni.
        depths (List[int]): Har bir bosqichdagi bloklar soni.
        dims (List[int]): Har bosqichdagi kanal o‘lchamlari.
        drop_path_rate (float): Umumiy stochastic depth ehtimoli.
        layer_scale_init_value (float): Layer scale uchun boshlang‘ich qiymat.
        head_init_scale (float): Head qatlamlarining og‘irliklari uchun ko‘paytiruvchi.
    """
    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()

        self.dims = dims

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            ))

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                for j in range(depths[i])
            ]))
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """
        Downsample va bloklardan o‘tib, global pooling bilan chiqadi.

        Args:
            x (Tensor): Kiruvchi rasm.

        Returns:
            Tensor: Ekranga tayyorlangan tasvir embedding.
        """
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        """
        Modelning to‘liq forward pass'i.

        Args:
            x (Tensor): Kirish tensor.

        Returns:
            Tensor: Chiqish klassifikatsiya logitlari.
        """
        x = self.forward_features(x)
        return self.head(x)


class LayerNorm(nn.Module):
    """
    LayerNorm moduli `channels_first` va `channels_last` formatlarni qo‘llab-quvvatlaydi.

    Args:
        normalized_shape (int): Normallash o‘lchami.
        eps (float): Numerik barqarorlik uchun epsilon.
        data_format (str): "channels_last" yoki "channels_first"
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]


def get_convnext_model():
    """
    ConvNeXt modelini yaratadi va transform qaytaradi.

    Returns:
        Tuple[nn.Module, transform]: Model va `torchvision` transform.
    """
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    model.head = nn.Sequential(
        nn.Linear(768, 512),
        nn.GELU(),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Linear(256, 2),
    )
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return model, transform


def conversion_helper(val, conversion):
    """
    Recursive ravishda `conversion` funksiyasini `val` ga qo‘llaydi.

    Args:
        val: Tensor, tuple yoki list
        conversion: funksiyani qo‘llovchi callable

    Returns:
        Converted val.
    """
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    return tuple(rtn) if isinstance(val, tuple) else rtn


def fp32_to_fp16(val):
    """ FP32'dan FP16'ga konversiya """
    def half_conversion(v):
        if isinstance(v, (Parameter, Variable)):
            v = v.data
        if isinstance(v, FLOAT_TYPES):
            v = v.half()
        return v
    return conversion_helper(val, half_conversion)


def fp16_to_fp32(val):
    """ FP16'dan FP32'ga konversiya """
    def float_conversion(v):
        if isinstance(v, (Parameter, Variable)):
            v = v.data
        if isinstance(v, HALF_TYPES):
            v = v.float()
        return v
    return conversion_helper(val, float_conversion)


class FP16Module(nn.Module):
    """
    Modelni FP16 da ishga tushirish uchun wrapper.

    Args:
        module (nn.Module): Original model.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module.half()

    def forward(self, *inputs, **kwargs):
        return fp16_to_fp32(self.module(*(fp32_to_fp16(inputs)), **kwargs))


def get_convnext_tiny_model(model_path: str, device: str = None, fp16: bool = False):
    """
    ConvNeXt-tiny modelini .pth fayldan yuklaydi.

    Args:
        model_path (str): Model fayli yo‘li.
        device (str): Qurilma: "cuda" yoki "cpu".
        fp16 (bool): FP16 rejimda ishga tushirish.

    Returns:
        Tuple[nn.Module, transform]: Yuklangan model va transform.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, transform = get_convnext_model()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model fayli topilmadi: {model_path}")

    weights = torch.load(model_path, map_location=device)
    model.load_state_dict(weights)

    if fp16:
        model = FP16Module(model)

    model = model.eval().to(device)
    return model, transform
