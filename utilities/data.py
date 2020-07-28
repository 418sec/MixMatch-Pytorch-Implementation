import numpy as np
import itertools
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

class TwoStreamBatchSampler(Sampler): # copy from: Curious AI
    
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.

    e.g. 
    total training = 837, batch_size=15, 
    data_loader = 837/15=56, unlabel_data=92, each_batch_has_unlabel = 92/56=2, each_batch_has_label=745/56=13
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        """
        primary_indices: unlabeled_idx
        secondary_indices: labeled_idx
        batch_size: batch_size, each batch has how many img
        secondary_batch_size: each batch has how many labeled img
        """
        self.primary_indices = primary_indices # list: unlabeled idx
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size # 62: one batch has 62 labeled
        self.primary_batch_size = batch_size - secondary_batch_size # 100-62=38: one batch has 38 unlabeled

    def __iter__(self): # BatchSampler的作用就是将前面的Sampler采样得到的索引值进行合并，当数量等于一个batch大小后就将这一批的索引值返回。
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)                                                                                      
            in  zip(grouper(primary_iter, self.primary_batch_size),  # unlabel idx里取38个，labeled idx里取62个，拼在一起，构成一个完整的batch
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size 

def iterate_once(iterable): # copy from: Curious AI
    return np.random.permutation(iterable) # schuffle index list

def iterate_eternally(indices): # copy from: Curious AI
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n): # copy from: Curious AI
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def splitSampler(df, target_col:str, split_ratio:float):
    sample_df = df.groupby(target_col, as_index=False).apply(lambda x: x.sample(frac=split_ratio, random_state=1)) # random_state=seed

    # get primary index list
    primary_idx = sample_df.index.get_level_values(1).tolist()

    # get labeled index list
    all_idx = df.index.values.tolist()
    secondary_idx = set(all_idx) - set(primary_idx)
    # check if conflict label
    check = set(secondary_idx) & set(primary_idx)
    if check:
        print('conflict label')
    else:
        return secondary_idx, primary_idx 
    
def transform(dataset):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2470,  0.2435,  0.2616])
    if dataset == 'train':
        transformation = transforms.Compose([
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation((-30,30)),
                    transforms.ToTensor(),
                    transforms.Normalize(**channel_stats)
                ])
    elif dataset == 'val' or dataset == 'test':
        transformation = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(**channel_stats)
            ])
    return transformation