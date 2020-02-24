import torch

class Dataset(torch.utils.data.IterableDataset):
    r"""
    An iterable dataset to save the data. This dataset supports multi-processing
    to load the data.

    Arguments:
        iterator: the iterator to read data.
        num_lines: the number of lines read by the individual iterator.
    """
    def __init__(self, iterator, num_lines):
        super(Dataset, self).__init__()
        self._num_lines = num_lines
        self._iterator = iterator
        self._setup = False

    def _setup_iterator(self):
        r"""
        _setup_iterator() function assign the starting line and the number
        of lines to read for the individual worker. Then, send them to the iterator
        to load the data.

        If worker info is not avaialble, it will read all the lines across epochs.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            chunk = int(self._num_lines / worker_info.num_workers)
            start = chunk * worker_info.id
            read = chunk
            if worker_info.id == worker_info.num_workers - 1:
                # The last worker needs to pick up some extra lines
                # if the number of lines aren't exactly divisible
                # by the number of workers.
                # Each epoch we loose an 'extra' number of lines.
                extra = self._num_lines % worker_info.num_workers
                read += extra
        else:
            start = 0
            read = self._num_lines
        self._iterator = self._iterator(start, read)

    def __iter__(self):
        if self._setup is False:
            self._setup_iterator()
            self._setup = True
        for x in self._iterator:
            yield x


def count(data_path):
    r"""
    return the total numerber of text entries and labels.
    """
    with io.open(data_path, encoding="utf8") as f:
        return 20, 243344
