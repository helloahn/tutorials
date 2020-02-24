import torch
import io

from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def _csv_iterator(data_path, ngrams, yield_cls=False):
    tokenizer = get_tokenizer("basic_english")
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            tokens = ' '.join(row[1:])
            # print(tokens)
            tokens = tokenizer(tokens)

            # print(tokens)
            if yield_cls:
                yield int(row[0]) - 1, ngrams_iterator(tokens, ngrams)
            else:
                yield ngrams_iterator(tokens, ngrams)

# vocab = build_vocab_from_iterator(_csv_iterator(".data/ag_news_csv/train.csv", 2))
vocab = build_vocab_from_iterator(_csv_iterator(".data/aclass/train_word.csv", 2))
print ('Vocab Size: %d' % len(vocab))
torch.save(vocab, "test_vocab.i")