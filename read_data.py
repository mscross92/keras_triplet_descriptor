## Modification from original HardNet implementation in 
## https://raw.githubusercontent.com/DagnyT/hardnet/master/code/dataloaders/HPatchesDatasetCreator.py
## I need to clean it a little bit and modify some things, but it works

import os
import numpy as np
import cv2
import sys
import json
import keras
from tqdm import tqdm
import glob
import random
import tensorflow as tf

splits = ['a']
# tps = ['0','1','2','3','4','5','6','7','8','9','10',\
#        '11','12','13','14','15','16','17','18']
# tps = ['0','1','2','3','4','5','6','7','8','9','10',\
#        '11']

tps = ['0','1','2','3','4']



class DenoiseHPatches(keras.utils.Sequence):
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps
    def __init__(self, seqs, batch_size = 32):
        self.all_paths = []
        self.batch_size = batch_size
        self.dim = (29, 29)
        self.n_channels = 1
        self.sequences = {}
        self.sequences_n = {}
        for base in tqdm(seqs):
            name = base.split('/')
            self.name = name[-1]
            self.base = base
            for t in self.itr:
                im_path = os.path.join(base, t + '.png')
                img_n = cv2.imread(im_path.replace('.png', '_noise.png'), 0)
                img = cv2.imread(im_path, 0)
                N = img.shape[0] / 29
                seq_im = np.array(np.split(img, N),
                                  dtype=np.uint8)
                seq_im_n = np.array(np.split(img_n, N),
                                    dtype=np.uint8)
                for i in range(int(N)):
                    path = os.path.join(base, t, str(i) + '.png')
                    self.all_paths.append(path)
                    self.sequences[path] = seq_im[i]
                    self.sequences_n[path] = seq_im_n[i]
        self.on_epoch_end()

    def get_images(self, index):
        path = self.all_paths[index]
        img = self.sequences[path].astype(np.float32)
        img_n = self.sequences_n[path].astype(np.float32)
        return img, img_n

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.all_paths) / self.batch_size))

    def __getitem__(self, index):
        img_clean = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        img_noise = np.empty((self.batch_size,) + self.dim + (self.n_channels,))

        for i in range(self.batch_size):
            img, img_n = self.get_images(index*self.batch_size+i)    
            img_clean[i] = np.expand_dims(img, -1)
            img_noise[i] = np.expand_dims(img_n, -1)

        return img_noise, img_clean    

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        random.shuffle(self.all_paths)

class hpatches_sequence_folder:
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps
    def __init__(self, base, noise=1):
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        if noise:
            noise_path = '_noise'
        else:
            noise_path = ''
        for t in self.itr:
            im_path = os.path.join(base, t+noise_path+'.png')
            print(im_path)
            im = cv2.imread(im_path,0)
            self.N = im.shape[0]/29
            setattr(self, t, np.split(im, self.N))



def generate_triplets(labels, num_triplets, batch_size):
    def create_indices(_labels):
        inds = dict()
        for idx, ind in enumerate(_labels):
            if ind not in inds:
                inds[ind] = []
            inds[ind].append(idx)
        return inds
    triplets = []
    indices = create_indices(np.asarray(labels))
    unique_labels = np.unique(np.asarray(labels))
    n_classes = unique_labels.shape[0]

    # add only unique indices in batch
    already_idxs = set() # initiate as empty set
    
    for x in tqdm(range(num_triplets)):
        if len(already_idxs) >= batch_size:
            already_idxs = set() # ensures each batch has a different positive class
        
        c1 = np.random.randint(0, n_classes) # random anchor class
        while c1 in already_idxs:
            c1 = np.random.randint(0, n_classes)
        already_idxs.add(c1)
        c2 = np.random.randint(0, n_classes) # random negative class
        while c1 == c2:
            c2 = np.random.randint(0, n_classes)
        if len(indices[c1]) == 2:  # hack to speed up process
            n1, n2 = 0, 1
        else:
            n1 = np.random.randint(0, len(indices[c1])) # random index of anchor in class
            n2 = np.random.randint(0, len(indices[c1])) # random index of positive in anchor class
            while n1 == n2:
                n2 = np.random.randint(0, len(indices[c1]))
        n3 = np.random.randint(0, len(indices[c2])) # random index of negative in class

        triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]]) # create triplet


    return np.array(triplets)

def get_current_descriptors(x,model):
        x = np.expand_dims(x, -1)
        desc = model.predict(x=x)
        return desc

# input: index 
def get_hard_negs(desc_x,a_idx,class_idxs):

    # anchor descriptor
    desc_a = desc_x[a_idx]

    # get distance from anchor and all other descriptors

    # add to distance vector if from a different class to anchor
    negs_idx = []
    negs_dist = []
    for ii,dsc in enumerate(desc_x):
        if ii not in class_idxs:
            # compute distance
            d = tf.norm(desc_a - dsc, axis=-1)
            negs_dist.append(d)
            negs_idx.append(ii)
    
    # concat ids and distances in a matrix
    m = np.concatenate((negs_dist,negs_idx))
    print(m.shape)

    # sort descending by distance
    m = m[m[:,0].argsort()[::-1]]
    print(m[0][0])
    print(m[0][1])

    # return indices for first K neighbours - currently returning all!
    return m[1]
    
def generate_hard_triplets(labels, num_triplets, batch_size, descrptrs):
    def create_indices(_labels):
        inds = dict()
        for idx, ind in enumerate(_labels):
            if ind not in inds:
                inds[ind] = []
            inds[ind].append(idx)
        return inds
    triplets = []
    indices = create_indices(np.asarray(labels))
    unique_labels = np.unique(np.asarray(labels))
    n_classes = unique_labels.shape[0]
    # add only unique indices in batch
    already_idxs = set()
    
    for x in tqdm(range(num_triplets)):
        if len(already_idxs) >= batch_size:
            already_idxs = set()
        
        # randomly select anchor class
        c1 = np.random.randint(0, n_classes)
        while c1 in already_idxs: # ensuring class isnt already in batch
            c1 = np.random.randint(0, n_classes)
        already_idxs.add(c1)

        # select indices for specific positive and anchor samples
        if len(indices[c1]) == 2:  # hack to speed up process
            n1, n2 = 0, 1
        else:
            n1 = np.random.randint(0, len(indices[c1]))
            n2 = np.random.randint(0, len(indices[c1]))
            while n1 == n2:
                n2 = np.random.randint(0, len(indices[c1]))
        
        indx = indices[c1][n1] # anchor index

        # check if anchor index has hard negatives
        negative_indices = get_hard_negs(descrptrs,indx,indices[c1])
        if(len(negative_indices)>0):
            # randomly select a hard negative from hardest 7 (as in hardnet!)
            negative_indx = random.choice(negative_indices[:7])
        else:
            # randomly select any negative
            c2 = np.random.randint(0, n_classes)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes)
            n3 = np.random.randint(0, len(indices[c2]))
            negative_indx = indices[c2][n3]

        triplets.append([indices[c1][n1], indices[c1][n2], negative_indx])

    return np.array(triplets)


class HPatches():
    def __init__(self, train=True, transform=None, download=False, train_fnames=[],
                 test_fnames=[], denoise_model=None, use_clean=True):
        self.train = train
        self.transform = transform
        self.train_fnames = train_fnames
        self.test_fnames = test_fnames
        self.denoise_model = denoise_model
        self.use_clean = use_clean

    def set_denoise_model(self, denoise_model):
        self.denoise_model = denoise_model

    def denoise_patches(self, patches):
        batch_size = 100
        for i in tqdm(range(int(len(patches) / batch_size)), file=sys.stdout):
            batch = patches[i * batch_size:(i + 1) * batch_size]
            batch = np.expand_dims(batch, -1)
            batch = np.clip(self.denoise_model.predict(batch).astype(int),
                                        0, 255).astype(np.uint8)[:,:,:,0]
            patches[i*batch_size:(i+1)*batch_size] = batch
        batch = patches[i*batch_size:]
        batch = np.expand_dims(batch, -1)
        batch = np.clip(self.denoise_model.predict(batch).astype(int),
                        0, 255).astype(np.uint8)[:,:,:,0]
        patches[i*batch_size:] = batch
        return patches

    def read_image_file(self, data_dir, train = 1):
        """Return a Tensor containing the patches
        """
        if self.denoise_model and not self.use_clean:
            print('Using denoised patches')
        elif not self.denoise_model and not self.use_clean:
            print('Using noisy patches')
        elif self.use_clean:
            print('Using clean patches')
        sys.stdout.flush()
        patches = []
        labels = []
        counter = 0
        hpatches_sequences = [x[1] for x in os.walk(data_dir)][0]
        if train:
            list_dirs = self.train_fnames
        else:
            list_dirs = self.test_fnames

        for directory in tqdm(hpatches_sequences, file=sys.stdout):
           if (directory in list_dirs):
            for tp in tps:
                if self.use_clean:
                    sequence_path = os.path.join(data_dir, directory, tp)+'.png'
                else:
                    sequence_path = os.path.join(data_dir, directory, tp)+'_noise.png'
                image = cv2.imread(sequence_path, 0)
                h, w = image.shape
                n_patches = int(h / w)
                for i in range(n_patches):
                    patch = image[i * (w): (i + 1) * (w), 0:w]
                    patch = cv2.resize(patch, (29, 29))
                    patch = np.array(patch, dtype=np.uint8)
                    patches.append(patch)
                    labels.append(i + counter)
            counter += n_patches

        patches = np.array(patches, dtype=np.uint8)
        if self.denoise_model and not self.use_clean:
            print('Denoising patches...')
            patches = self.denoise_patches(patches)
        return patches, labels


class DataGeneratorDesc(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, data, labels, num_triplets = 1000000, batch_size=128, mdl = None, mining_strategy=0, dim=(29,29), n_channels=1, shuffle=True):
        # 'Initialization'
        self.transform = None
        self.out_triplets = True
        self.dim = dim
        self.batch_size = batch_size
        self.mining_strategy = mining_strategy
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.data = data
        self.labels = labels
        self.num_triplets = num_triplets
        self.mdl = mdl
        self.on_epoch_end()
        

    def get_image(self, t):
        def transform_img(img):
            if self.transform is not None:
                img = transform(img.numpy())
            return img

        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a).astype(float)
        img_p = transform_img(p).astype(float)
        img_n = transform_img(n).astype(float)

        img_a = np.expand_dims(img_a, -1)
        img_p = np.expand_dims(img_p, -1)
        img_n = np.expand_dims(img_n, -1)
        if self.out_triplets:
            return img_a, img_p, img_n
        else:
            return img_a, img_p

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.triplets) / self.batch_size))
                
    def __getitem__(self, index):
        y = np.zeros((self.batch_size, 1))
        img_a = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        img_p = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        if self.out_triplets:
            img_n = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        for i in range(self.batch_size):
            t = self.triplets[self.batch_size*index + i]    
            img_a_t, img_p_t, img_n_t = self.get_image(t)
            img_a[i] = img_a_t
            img_p[i] = img_p_t
            if self.out_triplets:
                img_n[i] = img_n_t

        return {'a': img_a, 'p': img_p, 'n': img_n}, y

    def on_epoch_end(self):
        if self.mining_strategy>0:

            if self.mdl is not None:
                # get current descriptors
                self.descriptors = get_current_descriptors(self.data,self.mdl)

                # get triplets, hard negative mining using descriptors
                self.triplets = generate_hard_triplets(self.labels, self.num_triplets, self.batch_size,self.descriptors)
            else:
                # randomly select negatives
                self.triplets = generate_triplets(self.labels, self.num_triplets, self.batch_size)
        else:
            # randomly select negatives
            self.triplets = generate_triplets(self.labels, self.num_triplets, self.batch_size)
