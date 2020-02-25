import torch
import torchtext

from dataset import Dataset
from get_csv_iterator import get_csv_iterator
from sklearn.metrics import fbeta_score

N_EPOCHS = 100
NGRAMS = 2
BATCH_SIZE = 16
LEARNING_RATE = 4.0
CROSS_VALIDATION = 5
TRAIN_VALID_RATIO = 0.95
THRESHOLD = 0.5
train_len = 243344
test_len = 30418
EMBED_DIM = 64

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
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, num_class)
        # self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        # return self.fc(embedded)
        return self.fc2(self.fc(embedded))


######################################################################
# Initiate an instance

VOCAB_SIZE = len(vocab)
EMBED_DIM = 32
NUM_CLASS = 20
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)


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
        ret = output > THRESHOLD
        result = []
        print(ret)
        cnt = 0
        for r in ret[0]:
            if r:
                result.append(cnt)
            cnt += 1
        return result

ex_text_str = []
answer = []

answer.append("1 0")
ex_text_str.append("173 711 1032 57 534 173 681 57 711 574 1032 841 3438 340 859 263 353 20 869 33 488 1032") 

answer.append("14 6")
ex_text_str.append("15514 11 879 246 2670 19151 11130 15514 548 11 879 246 2670 9566 94448 378 93138 93139 2834 15514 378 94448 93919 58476 93919 58476 3344 2951 58476 244 1659 246 699 719 656 512 30708 29343 3344 58476 31978 699 719 374 376 56866 370 2288 11 93539 93540 2395 41698 23641 3344 366 521 1065 3344 366 521 2444 374 719 656 512 4130 30708 699 719 656 512 4130 30708 969 507 3344 14031 244 561 521 246")
answer.append( "17")
ex_text_str.append("77924 4144 271 79489 77867 81326 77872 16189 77851 19462 81327 23655 39607 78583 14533 77853") 

answer.append("18 14 6")
ex_text_str.append("15514 75526 96543 15514 548 35799 2683 4521 31221 6411 16301 79807 16301 79807 79807 6412 85562 59731 3344 24301 1757 3841 951 33046 1545 951 2937 3344 3344 746 169 170 171")

answer.append("6")
ex_text_str.append("15514 611 624 632 611 624 2367 611 624 421 344 1305 421 2367 1042 611 624 3182 6959")

answer.append("5")
ex_text_str.append("485 2005 20 271 2005 20 271 8286 12204 85 10366 3500 2058 125398 985 26674 997 231600 7970 977 4417 214050 11964 60705 979 21150 997 225289 4744 12731 4906 18921 22605 65965 11844 977 985 11803 16678 997 11281 11804 3256 115304 979 7112 997 12320 232204 1007 997 226923 64727 163655 979 224925 125398 231340 11281 66243 11857 66202 18941 3706 235349 231342 954 18996 954 2224 27639 129478 4614 294173 12320 2058 44078 11281 66243 6945 232567 3256 243767 8199 3975 5588 6940 97016 954 118485 985 225439 4614 41775 7240 5588 6940 240691 979 997 225763 4614 27800 7547 394 3097 997 11281 11804 1105 60802 224732 954 985 16909 155261 229555 11859 28014 128770 979 11859 11281 11804 129479 1652 7248 231490 982 224872")

answer.append("5")
"3473 86 70801 227476 4144 271 61 62 1009"

answer.append("15")
ex_text_str.append("22660 2339 3141 809 13771 9791 29 271 80 32036 10291 142 36520 11 812 3136 3138 3137 3139 7426 38167 38168 3136 3137 3138 3139 3136 3138 38169 38170 13 5871 7426 38171 38172 3959 20871 11299 3141 214 3135 214 3135 2358 189 809 764 80 1591 10156 764 36338 36386 80 10156 764 36338 37472 948 809 11 3141 764 1760 36507 764 36338 1760 36338 36410 38001 193 3130 1393 11299 37409 37834 2358")

answer.append("13 12")
ex_text_str.append("5322 93184 93089 241 271 93184 93089 93089 7308 766 95801 371 677 441 9489 378 7308 31923 7308 1093 246 17865 91 3346 1204 3778 2367 42 6917 453 632 4643 271")

model = model.to("cpu")

for i in range(len(ex_text_str)):
    print(predict(ex_text_str[i], model, vocab, NGRAMS))
    print(answer[i])
