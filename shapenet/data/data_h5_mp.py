import sys
if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue
import atexit
import logging
import multiprocessing as mp
import random
import h5py
import time
import numpy as np


def pc_normalize(pc):
    """ pc: NxC, return NxC """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc

class H5Dataset:
    """
    Multiprocess dataloader
    """
    maxnverts = 10000
    maxntris = 10000
    texture = 0

    datapath = []

    def __init__(self, listfile, maxnverts=10000, maxntris=10000, batch_size=20, normalize=True, texture=0, n_thread=8):

        H5Dataset.maxnverts = maxnverts
        H5Dataset.maxntris = maxntris

        self.batch_size = batch_size
        self.inputlistfile = listfile
        self.normalize = normalize
        H5Dataset.texture = texture

        H5Dataset.datapath = H5Dataset.read_list(listfile)#[:90]

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 18000

        self.n_thread = n_thread
        if n_thread > 0:
            self.multi_thread = True
        else:
            self.multi_thread = False

        self.stop_word = '==STOP--'
        self.current_batch = None
        self.data_num = len(H5Dataset.datapath)
        self.current = None
        self.worker_proc = None

        self.order = list(range(len(H5Dataset.datapath)))

        self.cache = {}
        self.cache_size = 7000

        if self.multi_thread:
            self.stop_flag = mp.Value('b', False)
            self.result_queue = mp.Queue(maxsize=self.batch_size*3)
            self.data_queue = mp.Queue()
            self.reset()
    
    def __len__(self):
        # print(H5Dataset.datapath)
        # print ('len(H5Dataset.datapath)', len(H5Dataset.datapath))
        return len(H5Dataset.datapath)

    def __iter__(self):
        return self

    @staticmethod        
    def read_list(data_list):
        with open(data_list, 'r') as f:
            lines = f.read().splitlines()
            # print ('lines', len(lines))
            datapath_ = [line.strip() for line in lines]
        return datapath_

    def _insert_queue(self):
        for item in self.order:
            self.data_queue.put(item)
        [self.data_queue.put(self.stop_word) for pid in range(self.n_thread)]

    def _thread_start(self):
        self.stop_flag = False
        self.worker_proc = [mp.Process(target=H5Dataset._worker,
                                       args=[pid,
                                             self.data_queue,
                                             self.result_queue,
                                             self.stop_word,
                                             self.stop_flag])
                            for pid in range(self.n_thread)]
        [item.start() for item in self.worker_proc]

        def cleanup():
            self.shutdown()
        atexit.register(cleanup)

    @staticmethod
    def _worker(worker_id, data_queue, result_queue, stop_word, stop_flag):
        count = 0
        for item in iter(data_queue.get, stop_word):
            if stop_flag == True:
                break
            single_mesh = H5Dataset._get_single(item)
            result_queue.put(single_mesh)
            count += 1
        
    def reset(self): 
        self.current = 0
        self.shuffle()
        if self.multi_thread:
            self.shutdown()
            self._insert_queue()
            self._thread_start()

    def shutdown(self):
        if self.multi_thread:

            # clean queue
            while True:
                try:
                    self.data_queue.get(timeout=1)
                except Queue.Empty:
                    break
            while True:
                try:
                    self.result_queue.get(timeout=1)
                except Queue.Empty:
                    break

            # stop worker
            self.stop_flag = True
            if self.worker_proc:
                for i, worker in enumerate(self.worker_proc):
                    worker.join(timeout=1)
                    if worker.is_alive():
                        # logging.error('worker {} is join fail'.format(i))
                        worker.terminate()

    def shuffle(self):
        # pass
        random.shuffle(self.order)

    def __getitem__(self, index):

        if index + self.batch_size > self.data_num:
            index = index + self.batch_size - self.data_num

        batch_verts = np.zeros((self.batch_size, self.maxnverts, 3))
        batch_nverts = np.zeros((self.batch_size, 1)).astype(np.int32)
        batch_tris = np.zeros((self.batch_size, self.maxntris, 3)).astype(np.int32)
        batch_ntris = np.zeros((self.batch_size, 1)).astype(np.int32)
        batch_vertsmask = np.zeros((self.batch_size, self.maxnverts, 1)).astype(np.float32)
        batch_trismask = np.zeros((self.batch_size, self.maxntris, 1)).astype(np.float32)

        cnt = 0
        # index = index * self.batch_size
        # print (index)
        for i in range(index, index + self.batch_size):#(self.current, self.current + batch_size):
            # single_mesh = None
            # while single_mesh == None:

            if self.order[i] in self.cache:
                single_mesh = self.cache[self.order[i]]
            else:
                if self.multi_thread:
                    single_mesh = self.result_queue.get()
                else:
                    single_mesh = H5Dataset._get_single(self.order[i])
                if len(self.cache) < self.cache_size:
                    self.cache[self.order[i]] = single_mesh
            # print(single_mesh)

            if len(single_mesh) == 3:
                v1, t1, text1 = single_mesh
            else:
                v1, t1 = single_mesh
                
            if self.normalize:
                v1 = pc_normalize(v1)
            batch_verts[cnt,:len(v1),:] = v1
            batch_tris[cnt,:len(t1),:] = t1
            batch_nverts[cnt,0] = len(v1)
            batch_ntris[cnt,0] = len(t1)
            batch_vertsmask[cnt,:len(v1),0] = 1.
            batch_trismask[cnt,:len(t1),0] = 1.

            cnt += 1


        batch_data = {}
        batch_data['verts'] = batch_verts
        batch_data['nverts'] = batch_nverts
        batch_data['tris'] = batch_tris
        batch_data['ntris'] = batch_ntris
        batch_data['vertsmask'] = batch_vertsmask
        batch_data['trismask'] = batch_trismask


        return batch_data

    def __next__(self):
        return self.next()

    def next(self):

        if self.current + self.batch_size > self.data_num:
            raise StopIteration
        else:
            self.__getitem__(self.current)
            self.current += self.batch_size


    @staticmethod
    def _get_single(index):

        if H5Dataset.texture == 0:   
            try:
                h5_f = h5py.File(H5Dataset.datapath[index])
                # print (h5_f['verts'].shape[0], h5_f['tris'].shape[0])
                if h5_f['verts'].shape[0] > H5Dataset.maxnverts or h5_f['tris'].shape[0] > H5Dataset.maxntris:
                    h5_f.close()
                    raise Exception()
            except:
                if (index + 1) < len(H5Dataset.datapath):
                    return H5Dataset._get_single(index+1)
                else:
                    return H5Dataset._get_single(0)
            verts, tris = h5_f['verts'][:], h5_f['tris'][:]

            h5_f.close()
            return (verts, tris)

        else:
            try:
                h5_f = h5py.File(H5Dataset.datapath[index])
                if h5_f['verts'].shape[0] > H5Dataset.maxnverts or h5_f['tris'].shape[0] > H5Dataset.maxntris:
                    h5_f.close()
                    raise Exception()
            except:
                if (index + 1) < len(H5Dataset.datapath):
                    return H5Dataset._get_single(index+1)
                else:
                    return H5Dataset._get_single(0)

            verts, tris, textures = h5_f['verts'][:], h5_f['tris'][:], h5_f['textures'][:]
            h5_f.close()

            return (verts, tris, textures)

        # if H5Dataset.texture == 0:   
        #     h5_f = h5py.File(datapath)
        #     if h5_f['verts'].shape[0] > H5Dataset.maxnverts or h5_f['tris'].shape[0] > H5Dataset.maxntris:
        #         h5_f.close()
        #         return None
        #     verts, tris = h5_f['verts'][:], h5_f['tris'][:]
        #     h5_f.close()
        #     return (verts, tris)
        # else:
        #     h5_f = h5py.File(datapath)
        #     if h5_f['verts'].shape[0] > H5Dataset.maxnverts or h5_f['tris'].shape[0] > H5Dataset.maxntris:
        #         h5_f.close()
        #         return None

        #     verts, tris, textures = h5_f['verts'][:], h5_f['tris'][:], h5_f['textures'][:]
        #     h5_f.close()
        #     return (verts, tris, textures)



if __name__ == '__main__':

    d = H5Dataset('/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/filelists/ShapeNetCore.v2.h5/03001627_obj.lst', n_thread=0)

    print (len(H5Dataset.datapath) / d.batch_size)
    # tic = time.time()
    # for i in range(10):
    #     a = d[i]
    #     # print(a['verts'])#.shape)
    #     print(i, a.keys())

    # print('time1:', time.time() - tic)

    d = H5Dataset('/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/filelists/ShapeNetCore.v2.h5/03001627_obj.lst',maxnverts=5000, maxntris=5000, n_thread=0)
    tic = time.time()
    for i in range(int(len(H5Dataset.datapath) / d.batch_size)+1):
        a = d[i]
        print(i, a['verts'].shape)
        # print(a.keys())

    print('time2:', time.time() - tic)

