from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

MODEL = convnext_tiny
MODEL_WEIGHTS = ConvNeXt_Tiny_Weights.DEFAULT
TRANSFORM =  ConvNeXt_Tiny_Weights.DEFAULT.transforms()