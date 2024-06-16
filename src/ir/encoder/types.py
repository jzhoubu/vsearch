"""Encoder types mapping."""

from .dpr import DPREncoderConfig, DPREncoder
from .vdr import VDREncoderConfig, VDREncoder
from .vdr_crossmodal_image import VDRImageEncoderConfig, VDRImageEncoder
from .vdr_crossmodal_text import VDRTextEncoderConfig, VDRTextEncoder

ENCODER_TYPES = {
    "vdr": VDREncoder,
    "dpr": DPREncoder,
    "vdr_crossmodal_image": VDRImageEncoder,
    "vdr_crossmodal_text":  VDRTextEncoder
}

CONFIG_TYPES = {
    "vdr": VDREncoderConfig,
    "dpr": DPREncoderConfig,
    "vdr_crossmodal_image": VDRImageEncoderConfig,
    "vdr_crossmodal_text":  VDRTextEncoderConfig,

}