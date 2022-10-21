import sys
sys.path.append('../')
from argparse import ArgumentParser
from PIL import Image
from tqdm.auto import tqdm
import wandb
import yaml

from data.dataset import DatasetWrapper
from data.dataloader import DataloaderWrapper
from model.model import ModelWrapper

#TODO: logging via wandb
#TODO: configs that work over time 

class Trainer:
    # Trainer only trains, because I don't like how monolithing lightning is :)
    def __init__(self, model, dataset, dataloader, logger, hparams, **kwargs):
        # essential items
        self.model = model
        self.dataset = dataset
        self.train_loader = dataloader
        self.val_loader = dataloader
        self.logger = logger
        
        # hyperparamters - unpack dictionary
        self.batch_size = hparams['batch_size']
        self.epochs = hparams['epochs']
        self.lr = hparams['lr']
        self.optim = hparams['optimizer']
        self.seed = hparams['seed']

        if self.seed:
            # set a seed in feature
            pass

    def configure_dataloaders(self):
        pass

    def train_step(self):
        pass

    def train_epoch(self):
        pass

    def eval_step(self):
        pass

    def finetune(self, args, **kwargs):
        """
        Finetune model over epochs on data
        """
        best_eval = 0.0
        for epoch_idx in tqdm(range(1, self.epochs + 1)):
            self.train_epoch(epoch=epoch_idx)
            if epoch_idx % 5 == 0: 
                eval_acc = self.eval_model(self.val_loader)
                # add to logs
                # self.logger.add_scalar('val/acc', eval_acc, global_step=epoch_idx)
                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    self.save_model(step=epoch_idx)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--config',
        dest='config',
        default='tiny-imagenet',
        type=str,
        help='config file to use'
    )
    parser.add_argument(
        '--logger',
        dest='logger',
        default=0,
        type=int,
        help='bool to use wandb logger'
    )

    args = parser.parse_args()
    # read in config file
    config_file_path = args.config
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    # TODO: refactor everything into the config state
    if args.log == 1:         
        wandb.init(
            project="transformer_paper",
            notes="debugging",
            tags=["debugging", "transformer_paper"],
            config=cfg,
        )

    dataset_interface = DatasetWrapper(args.datasetname)
    dataset = dataset_interface.dataset

    # wrap them together
    feature_extractor_interface = DataloaderWrapper(args.modelname)
    feature_extractor = feature_extractor_interface.feature_extractor

    model_interface = ModelWrapper(args.modelname)
    model = model_interface.model

    hparams = {
        'batch_size': 16,
        'epochs': 5,
        'lr': 0.001,
        'optimizer': None,
        'seed': None
    }

    # TODO: finetune function
    trainer = Trainer(
        model=model,
        dataset=dataset,
        dataloader=feature_extractor,
        logger = None, 
        hparams = hparams
    )
    results = trainer.finetune()

    # inputs = feature_extractor(images=image, return_tensors="pt")
    # outputs = model(**inputs)
    # logits = outputs.logits
    # # model predicts one of the 1000 ImageNet classes
    # predicted_class_idx = logits.argmax(-1).item()
    # print("Predicted class:", model.config.id2label[predicted_class_idx])

    #TODO save model - trainer.save_model() etc ...