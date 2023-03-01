import numpy as np
import random


class Enwik9Loader:
    """Iterator that returns shuffled slices of Enwik9"""

    def __init__(self, batch_size: int, seq_len: int, datapath: str):
        self.arr = np.fromfile(datapath, dtype=np.uint8)
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __iter__(self):
        # Make slice boundaries randomized across epochs
        offset = random.randint(0, self.seq_len - 1)
        offset_len = self.arr.size - offset
        seqs = offset_len // self.seq_len
        slices = np.array(
            [
                self.arr[start : start + self.seq_len]
                for start in range(offset, offset + seqs * self.seq_len, self.seq_len)
            ]
        )
        np.random.default_rng().shuffle(slices)
        short_batch = len(slices) % self.batch_size
        batches = [
            slices[start : start + self.batch_size]
            for start in range(0, len(slices) - short_batch, self.batch_size)
        ]
        return iter(batches)


if __name__ == "__main__":
    x = Enwik9Loader(100, 256, './enwik9')
    y = iter(x)
