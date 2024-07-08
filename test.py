from data_utils.hrs_dataset import HSRDataset
from torch.utils.data import DataLoader


train_dataset = HSRDataset(file_list_path='train.csv',
                           mixture_file_dir='mixture_train',
                           source_file_dir='concated_fg/machine',
                           bg_file_dir='concated_bg/env',
                           return_bg=True)

train_loader = DataLoader(train_dataset, batch_size=4,
                          shuffle=True, num_workers=4,
                          drop_last=True)

for i, data in enumerate(train_loader):
    mix, source, bg = data
    print(mix.shape, source.shape, bg.shape)