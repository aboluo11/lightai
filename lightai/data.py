from .core import *
from torch.utils.data.dataloader import default_collate, _DataLoaderIter


class _MyDataLoaderIter(_DataLoaderIter):
    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            self.sample_iter = iter(self.batch_sampler)
            indices = next(self.sample_iter, None)
        self.index_queues[self.worker_queue_idx].put((self.send_idx, indices))
        self.worker_queue_idx = (self.worker_queue_idx + 1) % self.num_workers
        self.batches_outstanding += 1
        self.send_idx += 1


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super().__init__(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None)
        self._dataloader = _DataLoaderIter(self)

    def __iter__(self):
        return self._dataloader
