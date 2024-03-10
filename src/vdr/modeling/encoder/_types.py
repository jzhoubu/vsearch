"""Encoder types."""

from .dpr import DPREncoderConfig, DPREncoder
from .vdr import VDREncoderConfig, VDREncoder
from .vdr_cm_image import VDRImageEncoderConfig, VDRImageEncoder
from .vdr_cm_text import VDRTextEncoderConfig, VDRTextEncoder

ENCODER_TYPES = {
    "vdr": VDREncoder,
    "dpr": DPREncoder,
    "vdr_cm_image": VDRImageEncoder,
    "vdr_cm_text":  VDRTextEncoder
}

CONFIG_TYPES = {
    "vdr": VDREncoderConfig,
    "dpr": DPREncoderConfig,
    "vdr_cm_image": VDRImageEncoderConfig,
    "vdr_cm_text":  VDRTextEncoderConfig,

}