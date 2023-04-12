# %% [markdown]
# Dealing with large chunked data is one of the challenges of this competition, which may impose an additional obstacle for people willing to join. Since the competition was not very active recently, I decided to share quite a simple way of dealing with the provided data without the time-consuming conversion of data into other formats or trying to load everything into the RAM, which is quite challenging, especially at Kaggle.
#
# The chunked data format is not very friendly to Pytorch, and some additional tricks are needed. My approach consists of two things: (1) A **dataloader caching the considered chunks** and (2) **Random Chunked Sampler**, which randomly selects a chunk and then goes through all ids in the chunk before selecting a new one. With this sampler, the caching pipeline spends time reading the corresponding chunk only at the first request, and the following requests are finished quickly with reading data from RAM. Meanwhile, when the cache is full, the earliest record is removed, which limits RAM usage. At inference and test, when reading is sequential, no extra modifications are needed to the data sampler.
# It would be great if the Pytorch team implemented native support of chunked data.

# %% [code] {"scrolled":true,"_kg_hide-output":true,"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-03-30T14:40:46.115794Z","iopub.execute_input":"2023-03-30T14:40:46.116497Z","iopub.status.idle":"2023-03-30T14:41:31.84028Z","shell.execute_reply.started":"2023-03-30T14:40:46.116447Z","shell.execute_reply":"2023-03-30T14:41:31.838453Z"}}
# install torch_geometric
# change to gpu version in gpu kernel

# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true,"execution":{"iopub.status.busy":"2023-03-30T14:41:39.584925Z","iopub.execute_input":"2023-03-30T14:41:39.585442Z","iopub.status.idle":"2023-03-30T14:41:56.605834Z","shell.execute_reply.started":"2023-03-30T14:41:39.585389Z","shell.execute_reply":"2023-03-30T14:41:56.604357Z"}}
import polars as pl
import pandas as pd
import gc, os, random, math
import numpy as np
from tqdm.notebook import tqdm
from collections import OrderedDict
from bisect import bisect_right
from typing import List, Tuple

import torch
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2023-03-30T14:42:01.571184Z","iopub.execute_input":"2023-03-30T14:42:01.571636Z","iopub.status.idle":"2023-03-30T14:42:01.581745Z","shell.execute_reply.started":"2023-03-30T14:42:01.571596Z","shell.execute_reply":"2023-03-30T14:42:01.580223Z"}}
PATH = 'data/icecube-neutrinos-in-deep-ice/'
META = 'data/icecube-neutrinos-in-deep-ice/'

SEED = 2023


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


seed_everything(SEED)


# %% [markdown]
# Here is an example of a dataset with caching. I use torch_geometric for illustration purposes, and you can easily modify it according to the data format used in your model. Be mindful of the number of workers and cache_size (to my understanding each worker has its own cache).

# %% [code] {"execution":{"iopub.status.busy":"2023-03-30T14:42:03.183429Z","iopub.execute_input":"2023-03-30T14:42:03.183855Z","iopub.status.idle":"2023-03-30T14:42:03.216681Z","shell.execute_reply.started":"2023-03-30T14:42:03.183815Z","shell.execute_reply":"2023-03-30T14:42:03.215173Z"}}
class IceCubeCache(Dataset):
    def __init__(self, path=PATH, mode='test', meta=None, pulse_limit=128, cache_size=1):
        val_fnames = ['batch_655.parquet', 'batch_656.parquet', 'batch_657.parquet',
                      'batch_658.parquet', 'batch_659.parquet']
        chunk_size = 200000
        self.mode, self.path_meta = mode, meta

        if mode == 'train' or mode == 'eval':
            assert meta is not None, 'Need to provide labels'
            self.path = os.path.join(path, 'train')
            self.files = [p for p in sorted(os.listdir(self.path)) \
                          if p != 'batch_660.parquet']  # 660 is shorter
            if mode == 'train':
                self.files = sorted(set(self.files) - set(val_fnames))
            else:
                self.files = val_fnames
            self.chunks = [chunk_size] * len(self.files)
        elif mode == 'test':
            self.path = os.path.join(path, 'test')
            self.files = [p for p in sorted(os.listdir(self.path))]

            # make sure that all files are considered regardless the number of events
            self.chunks = []
            for fname in self.files:
                ids = pl.read_parquet(os.path.join(self.path, fname) \
                                      ).select(['event_id']).unique().to_numpy().reshape(-1)
                self.chunks.append(len(ids))
            gc.collect()
        else:
            raise NotImplementedError

        self.chunk_cumsum = np.cumsum(self.chunks)
        self.cache, self.meta = None, None
        self.pulse_limit, self.cache_size = pulse_limit, cache_size
        self.geometry = pd.read_csv(os.path.join(path, 'sensor_geometry.csv'))
        self.geometry = (self.geometry[['x', 'y', 'z']].values / 500.0).astype(np.float32)

    def load_data(self, fname):
        if self.cache is None: self.cache = OrderedDict()
        if fname not in self.cache:
            df = pl.read_parquet(os.path.join(self.path, fname))
            df = df.groupby("event_id").agg([
                pl.count(),
                pl.col("sensor_id").list(),
                pl.col("time").list(),
                pl.col("charge").list(),
                pl.col("auxiliary").list(), ])
            self.cache[fname] = df.sort('event_id')
            if len(self.cache) > self.cache_size: del self.cache[list(self.cache.keys())[0]]

    def load_meta(self, fname):
        if self.meta is None: self.meta = OrderedDict()
        if fname not in self.meta:
            fidx = fname.split('.')[0].split('_')[-1]
            self.meta[fname] = pl.read_parquet(os.path.join(self.path_meta,
                                                            f'train_meta.parquet')).sort('event_id')
            if len(self.meta) > self.cache_size: del self.meta[list(self.meta.keys())[0]]

    def __len__(self):
        return self.chunk_cumsum[-1]
    def len(self):
        return self.chunk_cumsum[-1]
    def get(self, idx0):
        fidx = bisect_right(self.chunk_cumsum, idx0)
        fname = self.files[fidx]
        idx = int(idx0 - self.chunk_cumsum[fidx - 1]) if fidx > 0 else idx0

        self.load_data(fname)
        df = self.cache[fname][idx]
        sensor_id = df['sensor_id'][0].item().to_numpy()
        time = df['time'][0].item().to_numpy()
        charge = df['charge'][0].item().to_numpy()
        auxiliary = df['auxiliary'][0].item().to_numpy()

        pos = self.geometry[sensor_id]
        time = (time - 1e4) / 3e4
        charge = np.log10(charge) / 3.0
        auxiliary = auxiliary - 0.5

        x = np.stack([pos[:, 0], pos[:, 1], pos[:, 2], time, charge, auxiliary],
                     -1).astype(np.float32)
        x = torch.from_numpy(x)
        data = Data(x=x, n_pulses=torch.tensor(x.shape[0], dtype=torch.int32))

        # Downsample large events
        if data.n_pulses > self.pulse_limit:
            data.x = data.x[torch.randperm(len(data.x)).numpy()[:self.pulse_limit]]
            data.n_pulses = torch.tensor(self.pulse_limit, dtype=torch.int32)

        if self.mode != 'test':
            self.load_meta(fname)
            meta = self.meta[fname][idx]
            azimuth = meta['azimuth'].item()
            zenith = meta['zenith'].item()
            target = np.array([azimuth, zenith]).astype(np.float32)
            target = torch.from_numpy(target)
        else:
            target = df['event_id'].item()

        return data, target

    def __getitem__(self, idex0):
        return self.get(idex0)


# %% [markdown]
# This class performs chunked-based random sampling: selection of a chunk at random and then sampling all ids within the chunk (so the cache may be used effectively). It is nearly completely borrowed from the Pytotch source code with adding a special consideration for chunks.

# %% [code] {"execution":{"iopub.status.busy":"2023-03-30T14:42:14.920697Z","iopub.execute_input":"2023-03-30T14:42:14.921277Z","iopub.status.idle":"2023-03-30T14:42:14.93697Z","shell.execute_reply.started":"2023-03-30T14:42:14.921219Z","shell.execute_reply":"2023-03-30T14:42:14.93569Z"}}
class RandomChunkSampler(torch.utils.data.Sampler[int]):
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, chunks, num_samples: Optional[int] = None,
                 generator=None, **kwargs) -> None:
        # chunks - a list of chunk sizes
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator
        self.chunks = chunks

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        cumsum = np.cumsum(self.chunks)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        chunk_list = torch.randperm(len(self.chunks), generator=generator).tolist()
        # sample indexes chunk by chunk
        for i in chunk_list:
            chunk_len = self.chunks[i]
            offset = cumsum[i - 1] if i > 0 else 0
            yield from (offset + torch.randperm(chunk_len, generator=generator)).tolist()

    def __len__(self) -> int:
        return self.num_samples


# %% [markdown]
# Three examples below illustrate how to use the code for building train, evaluation, and test dataloaders. I use "num_workers=0" because at Kaggle the dataloader freezes if workers are created. At my home computer, 1000 batches (256,000 events > 1 data chunk) are sampled approximately within 20 seconds if 4 workers are used.

# %% [code] {"execution":{"iopub.status.busy":"2023-03-30T14:42:16.971571Z","iopub.execute_input":"2023-03-30T14:42:16.972042Z","iopub.status.idle":"2023-03-30T14:42:32.954045Z","shell.execute_reply.started":"2023-03-30T14:42:16.971997Z","shell.execute_reply":"2023-03-30T14:42:32.951913Z"}}
ds_train = IceCubeCache(mode='train', meta=META)
sampler = RandomChunkSampler(ds_train, chunks=ds_train.chunks)
dl_train = DataLoader(ds_train, batch_size=256, sampler=sampler, drop_last=True, num_workers=0)

for i, (x, y) in enumerate(dl_train):
    if i >= 1000: break

# %% [code]
ds_val = IceCubeCache(mode='eval', meta=META)
dl_val = DataLoader(ds_val, batch_size=256, shuffle=False, drop_last=False, num_workers=0)
for i, (x, y) in enumerate(dl_val):
    if i >= 1000: break

# %% [code]
ds_test = IceCubeCache(mode='test')
# use num_workers=0 at inference
dl_test = DataLoader(ds_test, batch_size=256, shuffle=False, drop_last=False, num_workers=0)
for x, y in dl_test:
    pass
