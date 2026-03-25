"""Compatibility shim for older imports."""

from src.smart_flip.quantization.awq import AWQConfig as QuantizationConfig
from src.smart_flip.quantization.awq import AWQQuantizerXL as SmartFlipAWQQuantizerXL
from src.smart_flip.quantization.state import IntegerQuantizedTensorState

__all__ = ["QuantizationConfig", "SmartFlipAWQQuantizerXL", "IntegerQuantizedTensorState"]
