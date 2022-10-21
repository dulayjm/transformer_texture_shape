import sys
sys.path.append('../')
from argparse import ArgumentParser
from PIL import Image

from data.dataset import DatasetWrapper
from data.dataloader import DataloaderWrapper
from model.model import ModelWrapper

#TODO: logging via wandb
#TODO: configs that work over time 

class Trainer:
    def __init__(self, args, **kwargs):
        pass

    def train_step(self):
        pass

    def train_epoch(self):
        pass

    def eval_model(self):
        pass

    def configure_optimizer(self):
        pass

    def finetune(args, **kwargs):
        """
        Finetune model over epochs on data
        """

        return 


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--fold',
        dest='fold',
        default=0,
        type=int,
        help='which fold we are on'
    )
    parser.add_argument(
        '--datasetname',
        dest='datasetname',
        default='tiny-imagenet',
        type=str,
        help='dataset to use'
    )
    parser.add_argument(
        '--modelname',
        dest='modelname',
        default='swinformer',
        type=str,
        help='modelname to use'
    )
    args = parser.parse_args()

    dataset_interface = DatasetWrapper(args.datasetname)
    dataset = dataset_interface.dataset

    # wrap them together
    feature_extractor_interface = DataloaderWrapper('swinformer')
    feature_extractor = feature_extractor_interface.feature_extractor

    model_interface = ModelWrapper('swinformer')
    model = model_interface.model

    # TODO: finetune function
    trainer = Trainer()
    results = trainer.finetune()

    # inputs = feature_extractor(images=image, return_tensors="pt")
    # outputs = model(**inputs)
    # logits = outputs.logits
    # # model predicts one of the 1000 ImageNet classes
    # predicted_class_idx = logits.argmax(-1).item()
    # print("Predicted class:", model.config.id2label[predicted_class_idx])

    #TODO save model - trainer.save_model() etc ...