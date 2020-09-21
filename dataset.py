import torch
import numpy as np
from faker import Faker
import random
from babel.dates import format_date
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

def transform(human_readable, machine_readable, human_vocab, machine_vocab, human_readable_length):
    X = list(map(lambda x: human_vocab.get(x, '<unk>'), human_readable))
    if len(X) < human_readable_length:
        X += [human_vocab['<pad>']] * (human_readable_length - len(X))
    elif len(X) > human_readable_length:
        X = X[:length]
    Y = list(map(lambda x: machine_vocab.get(x, '<unk>'), machine_readable)) #len(Y) is always 10, because the format is YYYY-MM-DD
    # print(human_readable)
    # print(machine_readable)
    # print(X)
    # print(Y)
    def zcs(length, idx):
        ret = np.zeros(length)
        ret[idx] = 1
        return ret
    # Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))
    Xoh = np.array(list(map(partial(zcs, len(human_vocab)), X)), dtype=np.float32)
    Yoh = np.array(list(map(partial(zcs, len(machine_vocab)), Y)), dtype=np.float32) #如果使用交叉熵loss，pytorch不需要label是onehot的；通过对比发现用MSELoss更好一些
    return Xoh, Yoh, {'human_readable':human_readable, 'machine_readable':machine_readable}
    # return Xoh, np.array(Y)

# def collate_fn(batch):
#     batch_x, batch_y = zip(*batch)
#     batch_x = [torch.FloatTensor(x) for x in batch_x]
#     batch_y = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True)
#     return batch_x, batch_y

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, n_datas=10000, seed=12345):
        self.transform = transform

        self.fake = Faker()
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)

        self.human_readable_length = 30 #the maximum length is less than 30
        self.human_vocab = set()
        self.machine_vocab = set()
        self.dataset = []
        for i in tqdm(range(n_datas)):
            human_readable, machine_readable = self.load_date()
            self.dataset.append((human_readable, machine_readable))
            self.human_vocab.update(tuple(human_readable))
            self.machine_vocab.update(tuple(machine_readable))

        self.human_vocab = dict(zip(sorted(self.human_vocab) + ['<unk>', '<pad>'], list(range(len(self.human_vocab) + 2))))
        self.inv_machine_vocab = dict(enumerate(sorted(self.machine_vocab)))
        self.machine_vocab = {v:k for k, v in self.inv_machine_vocab.items()}
        # print(self.human_vocab)
        # print(self.machine_vocab)
        # print(self.inv_machine_vocab)

    def __getitem__(self, idx):
        human_readable, machine_readable = self.dataset[idx] #dataset is a list of tuples
        return self.transform(human_readable, machine_readable, self.human_vocab, self.machine_vocab, self.human_readable_length)

    def __len__(self):
        return len(self.dataset)

    def load_date(self):
        """
            Loads some fake dates 
            :returns: tuple containing human readable string, machine readable string, and date object
        """
        dt = self.fake.date_object()

        human_readable = format_date(dt, format=random.choice(FORMATS),  locale='en_US')
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',', '')
        machine_readable = dt.isoformat()

        return human_readable, machine_readable

if __name__ == '__main__':
    dataset = Dataset(transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=2,
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=None)

    for step, (batch_x, batch_y) in enumerate(dataloader):
        print(step, batch_x.shape, batch_y.shape)
        print(batch_y)
        if step >= 1:
            break
