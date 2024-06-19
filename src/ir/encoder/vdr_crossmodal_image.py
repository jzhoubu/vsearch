import logging
from collections import OrderedDict
from contextlib import nullcontext
from typing import List, Union

from PIL import Image
import torch
from torch import Tensor as T
import torch.nn.functional as F
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from wordcloud import WordCloud

from .vdr_crossmodal_text import VALID_TOKEN_IDS, VID2LID
from ..utils.sparsify_utils import build_bow_mask, build_topk_mask, elu1p
from ..utils.visualize_utils import wordcloud_from_dict

logger = logging.getLogger(__name__)


class Bottleneck(torch.torch.nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = torch.nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)

        self.conv2 = torch.nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.avgpool = torch.nn.AvgPool2d(stride) if stride > 1 else torch.nn.Identity()

        self.conv3 = torch.nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes * self.expansion)

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = torch.nn.Sequential(OrderedDict([
                ("-1", torch.nn.AvgPool2d(stride)),
                ("0", torch.nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", torch.nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(torch.nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = torch.nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.c_proj = torch.nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class LayerNorm(torch.nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(torch.nn.Module):
    def __init__(self, d_model: int, n_head: int):
                 # , attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = torch.nn.Sequential(OrderedDict([
            ("c_fc", torch.nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", torch.nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask=None):
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.attn_mask = attn_mask
        self.heads = heads
        self.resblocks = torch.nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask=None):
        if attn_mask is not None: # change non-none text attn_mask to 3D
            attn_mask = (1.0 - attn_mask) * -10000.0  # NL
            attn_mask = attn_mask.unsqueeze(1) # N 1 L
            target_seq_length = attn_mask.shape[-1]
            attn_mask = attn_mask.repeat(self.heads, target_seq_length, 1) # N*heads, 1, L
        for i, layer_module in enumerate(self.resblocks):
            x = layer_module(x, attn_mask)

        return x



class VDRImageEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        resolution=224,
        tokenizer_id="bert-base-uncased",
        patch_size=32,
        width=768,
        layers=12,
        heads=12, 
        topk=768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resolution = resolution
        self.tokenizer_id = tokenizer_id
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.topk = topk


class VDRImageEncoder(PreTrainedModel):
    config_class = VDRImageEncoderConfig

    def __init__(self, config: VDRImageEncoderConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=config.width, kernel_size=config.patch_size, stride=config.patch_size, bias=False)
        scale = config.width ** -0.5
        self.positional_embedding = torch.nn.Parameter(scale * torch.randn((config.resolution // config.patch_size) ** 2, config.width))
        self.ln_pre = LayerNorm(config.width)
        self.transformer = Transformer(config.width, config.layers, config.heads)
        self.ln_post = LayerNorm(config.width)
        self.proj = torch.nn.Parameter(torch.ones([len(VALID_TOKEN_IDS), config.width]))
        self.valid_token_ids = VALID_TOKEN_IDS
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # [N, L, D] -> [L, N, D]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [L, N, D] -> [N, L, D]
        x = self.ln_post(x)
        return x

    def embed(self, image: Union[str, T], training: bool = False, topk=None):
        topk = topk or self.config.topk
        with torch.no_grad() if not training else nullcontext():
            if isinstance(image, str):
                image = self.load_image_file(image)
            img_emb = self(image.type(self.dtype).to(self.device)) # [N, L, D]
            img_emb = img_emb @ self.proj.t() # [N, L, V]
            img_emb = img_emb.max(1)[0]
            img_emb = elu1p(img_emb)
            img_emb = F.normalize(img_emb)
            topk_mask = build_topk_mask(img_emb, k=topk)
            img_emb = img_emb * topk_mask
        return img_emb

    def load_image_file(self, file_path):
        image = Image.open(file_path).convert('RGB')
        image = preprocess(image)
        return image.unsqueeze(0)

    def disentangle(self, image: Union[str, T], topk: int = None, visual=False, save_file=None):
        topk = topk or self.config.topk
        topk_result = self.embed(image).topk(topk)
        topk_token_ids = topk_result.indices.flatten().tolist()
        topk_token_ids = [VID2LID[x] for x in topk_token_ids]
        topk_values = topk_result.values.flatten().tolist()
        topk_tokens = self.tokenizer.convert_ids_to_tokens(topk_token_ids)
        results = dict(zip(topk_tokens,topk_values))
        if visual:
            wordcloud_from_dict(results, max_words=topk, save_file=save_file)
        return results


    dst = disentangle

    def display_image(self, image: Union[str, T] = None, save_file=None):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        crop_size = min(image.width, image.height)
        preprocess_ = Compose([
            Resize(crop_size, interpolation=Image.BICUBIC),
            CenterCrop(crop_size),
        ])
        image = preprocess_(image)
        if save_file is not None:
            image.save(save_file, format='PNG')
        return image
        



# for embedding
preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

# for visualize image
preprocess_ = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
])
