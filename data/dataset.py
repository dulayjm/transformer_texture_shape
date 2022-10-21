from datasets import load_dataset

# subclass it?
class DatasetWrapper:
    def __init__(self, name, **kwargs):
        if name == 'tiny-imagenet':
            self.dataset = load_dataset('Maysee/tiny-imagenet', split='train')
            
        # TODO add more datasets, wrapped here