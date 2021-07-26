import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np
import torch.nn.functional as F
import pandas as pd
import itertools

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from utils import pad_to_len
from model_cnn_v2 import SeqClassifier
# from model_slot import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, tag2idx, args.max_len) #split_data為train及eval的全部資料
        for split, split_data in data.items()
    }
    

    # TODO: crecate DataLoader for train / dev datasets
    x_train_token = [item['tokens'] for item in datasets['train'].data]
    x_train = np.array(vocab.encode_batch(x_train_token))
    y_train = [[datasets['train'].label2idx(tag) for tag in item['tags']] for item in datasets['train'].data]
    y_train = np.array(pad_to_len(y_train, x_train.shape[1], 1))


    x_dev_token = [item['tokens'] for item in datasets['eval'].data]
    x_dev = np.array(vocab.encode_batch(x_dev_token, to_len=x_train.shape[1]))
    y_dev = [[datasets['eval'].label2idx(tag) for tag in item['tags']] for item in datasets['eval'].data]
    y_dev = np.array(pad_to_len(y_dev, x_dev.shape[1], 1))

    trainset = Dataset(x_train, y_train)
    devset = Dataset(x_dev, y_dev)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle = True, num_workers=0)
    devloader = torch.utils.data.DataLoader(devset, batch_size = args.batch_size, shuffle = True, num_workers=0)


    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)

    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets['train'].num_classes, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else args.device)
    model.to(device)   


    # TODO: init optimizer
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr = args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    epoch_pbar = trange(args.num_epoch, desc="Epoch")

    # best_loss = 100000
    best_acc = 0
    patience = 5000
    count =0
    for epoch in epoch_pbar:
        acc_total = 0
        # TODO: Training loop - iterate over train dataloader and update model weights
        for batch_id, (X_batch, y_batch) in enumerate(trainloader):
            flag = False
            model.train()
            X_batch, y_batch = X_batch.to(device).to(torch.int64), y_batch.to(device, dtype=torch.int64) 
            optimizer.zero_grad()

            y_pred = model(X_batch)
            y_pred = y_pred.view(-1, datasets['train'].num_classes, 35)
            y_pred = y_pred.to(device)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            acc = acc_test(y_pred, y_batch, device)
            acc_total += acc.item()

            if batch_id % 100 == 0:
                

                dev_acc, dev_loss = validation(model, devset, devloader, device, criterion, datasets)

                #early stoppping
                if dev_acc > best_acc:
                    trigger_times = 0
                    best_model = model
                    best_acc = dev_acc
                    best_epoch = epoch
                    best_batch = batch_id
                    flag = True

                elif dev_acc < best_acc:
                    trigger_times += 100
                    if trigger_times >= patience:
                        print('Early stopping at epoch: %d, batch : %d' % (best_epoch, best_batch))
                        torch.save(best_model.state_dict(), args.ckpt_dir/'best.pt')
                        return 0
                        

                print(f'Epoch {epoch+0:03} | Batch : {batch_id+0:03} | Loss: {dev_loss:.5f} | Acc: {dev_acc:.5f}')
                if flag:
                    print('------------------------------------------------------best model------------------------------------------------------------------')
        train_acc = (acc_total/len(trainset)) * 100
        print(f'Epoch {epoch+0:03} | Acc: {train_acc:.5f}')

        # TODO: Evaluation loop - calculate accuracy and save model weights

    torch.save(best_model.state_dict(), args.ckpt_dir/'best.pt')   
        #pass

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=256)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def validation(model, devset, devloader, device, criterion, datasets):
    loss_total = 0
    acc_total = 0
    model.eval()
    with torch.no_grad():
        for batch_id, (x_dev, y_dev) in enumerate(devloader):
            x_dev = x_dev.to(device, dtype = torch.int64)
            y_dev = y_dev.to(device, dtype = torch.int64)
            y_pred = model(x_dev)
            y_pred = y_pred.view(-1, datasets['eval'].num_classes, 35)
            acc_dev = acc_test(y_pred, y_dev, device)
            acc_total += acc_dev.item()
            loss = criterion(y_pred, y_dev)
            loss_total += loss.item()
    acc_final = acc_total/len(devset)
    return acc_final*100, loss_total/len(devloader)

def acc_test(y_pred, y_test, device):
    _, y_pred = torch.max(y_pred, dim=1)
    y_pred = y_pred.to(device, dtype = torch.int64)
    # threshold = torch.tensor([0.0]).to(device)
    correct_results_sum = 0  
    #result = torch.round(y_pred).float()

    correct_results_sum = (y_pred == y_test).all(axis = 1).sum()
    
    return correct_results_sum

class testDataset(torch.utils.data.Dataset):
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)


