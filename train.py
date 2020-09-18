import os
import torch
import numpy as np
from tqdm import tqdm
from dataset import Dataset, transform
from model import Model
from util import save_load as sl

def calc_accuracy(pred, answer):
    # print(pred)
    pred = np.argmax(pred, axis=2)
    # print(pred.shape, answer.shape)
    correct = (pred == answer).astype(np.int)
    accuracy = correct.sum() / (pred.shape[0] * pred.shape[1])
    # print(accuracy)
    return accuracy

def train(model, loss_fn, optimizer, dataloader, epoch, use_gpu=False):
    pbar = tqdm(total=len(dataloader), bar_format='{l_bar}{r_bar}', dynamic_ncols=True)
    pbar.set_description(f'Epoch %d' % epoch)

    for step, (batch_x, batch_y) in enumerate(dataloader):
        if use_gpu:
            batch_x.cuda()
            batch_y.cuda()
        pred = model(batch_x)
        accuracy = calc_accuracy(pred.detach().cpu().numpy(), batch_y.detach().cpu().numpy())
        # print(pred.shape, batch_y.shape)
        loss = loss_fn(pred.transpose(1, 2), batch_y)
        # print(step, loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(**{'loss':loss.detach().cpu().item(), 'accuracy':accuracy})
        pbar.update()
    sl.save_checkpoint('./checkpoint', epoch, model, optimizer)

    pbar.close()
    
def main(gpu_ids=None):
    dataset = Dataset(transform=transform, n_datas=20000)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=100,
                                            shuffle=True,
                                            num_workers=10,
                                            collate_fn=None)

    model = Model(output_len=10)
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        n_gpus = torch.cuda.device_count()
        print('use %d gpus [%s]' % (n_gpus, gpu_ids))
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(n_gpus)])
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters())

    model = sl.load_model('./checkpoint', -1, model)
    optimizer = sl.load_optimizer('./checkpoint', -1, optimizer)

    try:
        trained_epoch = sl.find_last_checkpoint('./checkpoint')
        print('train form epoch %d' % (trained_epoch + 1))
    except Exception as e:
        print('train from the very begining, {}'.format(e))
        trained_epoch = -1
    for epoch in range(trained_epoch+1, 40):
        train(model, loss_fn, optimizer, dataloader, epoch, use_gpu=True)

if __name__ == '__main__':
    main(gpu_ids = '1, 2')