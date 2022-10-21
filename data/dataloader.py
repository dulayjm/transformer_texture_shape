from transformers import AutoFeatureExtractor

class DataloaderWrapper: 
    def __init__(self, name, **kwargs):

        # and add other transforms ...
        if name == 'swinformer':
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")