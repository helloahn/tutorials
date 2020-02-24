import torch
import torchtext

from dataset import Dataset
from get_csv_iterator import get_csv_iterator
from sklearn.metrics import fbeta_score

N_EPOCHS = 1
NGRAMS = 2
BATCH_SIZE = 16
LEARNING_RATE = 4.0
CROSS_VALIDATION = 5
TRAIN_VALID_RATIO = 0.95
THRESHOLD = 0.5
train_len = 243344
test_len = 30418

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data_path = ".data/aclass/train_word.csv"
test_data_path = ".data/aclass/test_word.csv"
vocab = torch.load("test_vocab.i")

######################################################################
# Define the model

import torch.nn as nn
import torch.nn.functional as F
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


######################################################################
# Initiate an instance

VOCAB_SIZE = len(vocab)
EMBED_DIM = 32
NUN_CLASS = 20
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)


######################################################################
# Functions used to generate batch

def generate_batch(batch):
    label = [entry[0] for entry in batch]
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    text = torch.cat(text)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    label = torch.tensor(label)

    return text, offsets, label


######################################################################
# Define functions to train the model and evaluate results.

from torch.utils.data import DataLoader
from sklearn.metrics import fbeta_score
from tqdm import tqdm

def train_func(sub_train_, length):

    # Train the model
    train_loss = 0
    train_acc = []
    div_num = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE,
                      collate_fn=generate_batch)
    with tqdm(unit_scale=0, unit='lines', total=length) as t:
        for i, (text, offsets, cls) in enumerate(data):
            t.update(len(cls))
            optimizer.zero_grad()
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            output = model(text, offsets)
            loss = criterion(output, cls)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc.append(fbeta_score(cls.cpu(), output.cpu() > THRESHOLD, beta=0.5, average='micro'))
            div_num += 1

    # Adjust the learning rate
    scheduler.step()

    return train_loss / div_num, sum(train_acc) / len(train_acc)

def test(data_, length):
    loss = 0
    acc = []
    div_num = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)

    with tqdm(unit_scale=0, unit='lines', total=length) as t:
        for text, offsets, cls in data:
            t.update(len(cls))
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            with torch.no_grad():
                output = model(text, offsets).detach()
                loss = criterion(output, cls)
                loss += loss.item()
                acc.append(fbeta_score(cls.cpu(), output.cpu() > THRESHOLD, beta=0.5, average='micro'))
                div_num += 1

    return loss / div_num, sum(acc)/len(acc)


######################################################################
# Split the dataset and run the model

import time
from torch.utils.data.dataset import random_split
min_valid_loss = float('inf')

criterion = torch.nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_dataset = Dataset(
        get_csv_iterator(
            train_data_path,
            NGRAMS,
            vocab,
            start=0,
            num_lines=train_len * TRAIN_VALID_RATIO),
        int(train_len * TRAIN_VALID_RATIO))

    valid_dataset = Dataset(
        get_csv_iterator(
            train_data_path,
            NGRAMS,
            vocab,
            start=train_len * TRAIN_VALID_RATIO,
            num_lines=train_len),
        train_len - int(train_len * TRAIN_VALID_RATIO))

    test_dataset = Dataset(
        get_csv_iterator(
            test_data_path,
            NGRAMS,
            vocab),
        test_len)

    train_loss, train_acc = train_func(train_dataset, int(train_len * TRAIN_VALID_RATIO))
    valid_loss, valid_acc = test(valid_dataset, int(train_len * (1 - TRAIN_VALID_RATIO)))

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')



######################################################################
# Evaluate the model with test dataset

print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset, test_len)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')



######################################################################
# Test on a random news

from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output > THRESHOLD

ex_text_str = "173 711 1032 57 534 173 681 57 711 574 1032 841 3438 340 859 263 353 20 869 33 488 1032" # 1, 0

model = model.to("cpu")

print(predict(ex_text_str, model, vocab, NGRAMS))


ex_text_str = "15514 11 879 246 2670 19151 11130 15514 548 11 879 246 2670 9566 94448 378 93138 93139 2834 15514 378 94448 93919 58476 93919 58476 3344 2951 58476 244 1659 246 699 719 656 512 30708 29343 3344 58476 31978 699 719 374 376 56866 370 2288 11 93539 93540 2395 41698 23641 3344 366 521 1065 3344 366 521 2444 374 719 656 512 4130 30708 699 719 656 512 4130 30708 969 507 3344 14031 244 561 521 246" # "14 6"

model = model.to("cpu")

print(predict(ex_text_str, model, vocab, NGRAMS))


ex_text_str = "77924 4144 271 79489 77867 81326 77872 16189 77851 19462 81327 23655 39607 78583 14533 77853" # "17"

model = model.to("cpu")

print(predict(ex_text_str, model, vocab, NGRAMS))