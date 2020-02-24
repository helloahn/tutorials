import io
import torch

from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.utils import unicode_csv_reader

def get_csv_iterator(data_path, ngrams, vocab, start=0, num_lines=None):
    r"""
    Generate an iterator to read CSV file.
    The yield values are an integer for the label and a tensor for the text part.

    Arguments:
        data_path: a path for the data file.
        ngrams: the number used for ngrams.
        vocab: a vocab object saving the string-to-index information
        start: the starting line to read (Default: 0). This is useful for
            on-fly multi-processing data loading.
        num_lines: the number of lines read by the iterator (Default: None).

    """
    def iterator(start, num_lines):
        tokenizer = get_tokenizer("basic_english")
        with io.open(data_path, encoding="utf8") as f:
            reader = unicode_csv_reader(f)
            for i, row in enumerate(reader):
                if i == start:
                    break
            for _ in range(num_lines):
                tokens = ' '.join(row[1:])
                tokens = ngrams_iterator(tokenizer(tokens), ngrams)

                label_onehot = [0.0 for _ in range(20)]
                for classNum in row[0].split(' '):
                    label_onehot[int(classNum)] = 1.0
                
                yield label_onehot, torch.tensor([vocab[token] for token in tokens])
                try:
                    row = next(reader)
                except StopIteration:
                    f.seek(0)
                    reader = unicode_csv_reader(f)
                    row = next(reader)
    return iterator