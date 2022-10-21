import sys
sys.path.append('../')
from transformers import SwinForImageClassification
import yaml

class ModelWrapper: 
    def __init__(self, name, **kwargs):
        if name == 'swinformer':
            self.model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    # code for if we use a config file for model attributes
    def get_model_config():
        with open("../config/config.yaml", "r") as ymlfile:
            cfg = yaml.load(ymlfile)
        return cfg