import sys
import os
import re
import pdb
import json
import math
import pickle
import logging
from tqdm import *
import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
# from warpctc_pytorch import CTCLoss

from model import *
from opts import *
from data import *
from utils.util import *
from utils.coding import *

def get_loaders(dataset, batch_size):
    dataset_size = len(dataset)
    validation_split = .2
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    random_seed= 42
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size,
                                                collate_fn=WordErrors.collate_fn,
                                                sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=batch_size,
                                                    collate_fn=WordErrors.collate_fn,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = 0.0001 * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def process(data_loader, split, model, optimizer, criterion):
    if split == 'train':
        model.train()
    else:
        model.eval()
    avgLoss = AverageMeter("{} loss".format(split))
    for iteration, batch in enumerate(tqdm(data_loader)):
        input_, target = batch[0].cuda(), batch[1].cuda()
        B, T = input_.size()
        hidden = model.initHidden(B)
        loss = 0
        for t in range(input_.size(1)):
            src, tgt = input_[:, t], target[:, t]
            pred, hidden = model(src, hidden)
            loss += criterion(pred, tgt)
        if split != 'val':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avgLoss.add(loss.item())
    loss = avgLoss.compute()
    return loss

def process_seq2seq(data_loader, split, model, batch_size, 
                    optimizer, criterion,
                    vocab):
    lmap, ilmap = vocab['v2i'], vocab['i2v']
    if split == 'train':
        model.train()
    else:
        model.eval()
    avgLoss = AverageMeter("{} loss".format(split))
    for iteration, batch in enumerate(tqdm(data_loader)):
        input_, target, lengths = batch[0].cuda(), batch[1].cuda(), batch[2]
        enc_out, enc_hidden = model.enc(input_, lengths)
        dec_inputs = torch.Tensor([lmap['#']]*batch[0].size(0)).cuda().long()
        dec_inputs = dec_inputs.unsqueeze(0)
        dec_hidden = enc_hidden
        loss = 0
        for t_ in range(input_.size(1)):
            dec_out, hidden, attn = model.dec(dec_inputs, dec_hidden, enc_out)
            loss += criterion(dec_out, target[:, t_])
            dec_inputs = target[:, t_].unsqueeze(0)
        if split != 'val':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avgLoss.add(loss.item())
    loss = avgLoss.compute()
    return loss


def train(**kwargs):
    input_ = kwargs['input']
    checkpoint = kwargs['checkpoint']
    savepath = kwargs['savepath']
    start_epoch = checkpoint['epoch']
    model = kwargs['model']
    vocab = kwargs['vocab']
    criterion = kwargs['criterion']
    optimizer = kwargs['optimizer']
    lmap, ilmap = vocab['v2i'], vocab['i2v']
    batch_size = kwargs['batch_size']
    epochs = kwargs['epochs']
    decoder = Decoder(lmap, ilmap)
    train_data_loader, val_data_loader = get_loaders(input_, batch_size)
    loader = {'train': train_data_loader, 'val': val_data_loader}
    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch)
        print('Epochs:[%d]/[%d]' % (epoch, kwargs['epochs']))
        losses = []
        for key in ['train', 'val']:
            losses.append(process(loader[key], key, model, optimizer, criterion))
            # losses.append(process_seq2seq(loader[key], key, model, batch_size, optimizer, criterion, vocab))
        print("Loss: {:.4f}...".format(losses[0]),
              "Val Loss: {:.4f}".format(losses[1]))
        info = '%d %.2f %.2f \n' % (epoch+1, losses[0], losses[1])
        logging.info(info)
        state = losses[1]
        print(checkpoint['best'])
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best': state
                        }, savepath,
                        is_best)


def main(**kwargs):
    opt = Config()
    opt._parse(kwargs)
    path = opt.path
    language = opt.language
    path = os.path.join(path, language)
    dataset = WordErrors('books', language)
    lmap = dataset.lmap
    ilmap = dataset.ilmap
    epochs = opt.epochs
    nHidden = opt.nHidden
    nClasses = len(lmap)
    save_dir = opt.save_dir
    batch_size = 32
    save_file = 'Bi_{}_error_LSTM.t7'.format(language)
    # save_file = 'seq2seq.t7'
    savepath = save_dir + '/' + save_file
    lr = opt.lr

    # model = Delayed_LSTM(nClasses, nHidden, nClasses).cuda()
    embed_dim = 300
    enc_units = 256
    dec_units = 256
    # model = Seq2Seq(nClasses, embed_dim, enc_units, dec_units)
    model = Delayed_LSTM(nClasses, nHidden, nClasses, bidirectional=True).cuda()
    if os.path.isfile(savepath):
        checkpoint = torch.load(savepath)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(savepath))
        checkpoint = {
            "epoch": 0,
            "best": float("inf")
        }

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gmkdir('logs')
    logging.basicConfig(filename="logs/%s.log" % save_file, level=logging.INFO)
    vocab = dict(v2i=lmap, i2v=ilmap)
    decoder = Decoder(lmap, ilmap)
    train(input=dataset,
          model=model,
          vocab=vocab,
          epochs=epochs,
          checkpoint=checkpoint,
          savepath=savepath,
          optimizer=optimizer,
          criterion=criterion,
          batch_size=batch_size)


if __name__ == '__main__':
    import fire
    fire.Fire(main)




'''
B, _ = input_.size()
if B != 32:
    hidden = tuple([each.data[:,:B,:] for each in hidden])
else:
    hidden = tuple([each.data for each in hidden[:B]])
B, T = target.size()
output = model(input_)
prediction = output.contiguous()
optimizer.zero_grad()
loss = criterion(prediction, target.view(B*T).long())
loss.backward()
optimizer.step()
avgLoss.add(loss.item())
'''