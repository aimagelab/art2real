from __future__ import print_function

import glob
import os
import pickle

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms
from util import h5_vs_binary
import time

class Contextual_loss():

    def __init__(self, patch_size1, patch_size2, patch_size3, stride1, stride2, stride3, weight, k, preload_indexes, preload_mem_patches, name, artistic_masks_dir, which_mem_bank):
        self.patch_sizes = []
        self.strides = []

        self.patch_sizes.append(patch_size1)
        self.strides.append(stride1)
        self.scales = 1
        if patch_size2 > 0:
            self.patch_sizes.append(patch_size2)
            self.strides.append(stride2)
            self.scales = 2
        if patch_size3 > 0:
            self.patch_sizes.append(patch_size3)
            self.strides.append(stride3)
            self.scales = 3

        self.k = k
        self.weight = weight
        self.preload_indexes = preload_indexes
        self.preload_mem_patches = preload_mem_patches

        self.indexes = []
        for s in range(self.scales):
            self.indexes.append({})

        self.mem_banks = []
        for s in range(self.scales):
            self.mem_banks.append({})

        self.means = []
        for s in range(self.scales):
            self.means.append({})

        self.artistic_masks = []
        for s in range(self.scales):
            self.artistic_masks.append({})

        self.name = name
        self.channels = 3 #rgb patches
        self.which_mem_bank = which_mem_bank
        self.artistic_masks_dir = os.path.join(self.which_mem_bank, artistic_masks_dir)

        if self.preload_mem_patches: #load RGB patches
            for s in range(self.scales):
                patch_size = self.patch_sizes[s]
                for i, filename in enumerate(glob.glob(self.which_mem_bank + '/patch_size_' + str(patch_size) + '/memory_bank_patches_binary/' + '*.bin')):
                    class_name, extension = os.path.splitext(filename)
                    class_name = class_name.split('/')[-1]
                    class_bank = h5_vs_binary.binary_read_all(open(filename, 'r'), patch_size * patch_size * self.channels)
                    self.mem_banks[s][class_name] = class_bank

            for s in range(self.scales):
                patch_size = self.patch_sizes[s]
                for i, filename in enumerate(glob.glob(self.which_mem_bank + '/patch_size_' + str(patch_size) + '/means/' + '*.npy')):
                    class_name, extension = os.path.splitext(filename)
                    class_name = class_name.split('/')[-1]
                    mean = np.load(filename)
                    self.means[s][class_name] = mean

        if self.preload_indexes: #load FAISS indexes
            for s in range(self.scales):
                patch_size = self.patch_sizes[s]
                for i, filename in enumerate(glob.glob(self.which_mem_bank + '/patch_size_' + str(patch_size) + '/faiss_indexes/' + '*.index')):
                    class_name, extension = os.path.splitext(filename)
                    class_name = class_name.split('/')[-1]
                    index = faiss.read_index(filename)
                    self.indexes[s][class_name] = index


    def load_obj(self, name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def affinities(self, a, b, mean):
        epsilon = 1e-5
        h = 1
        #mean = torch.mean(b)
        a = F.normalize(a - mean.repeat(a.shape[0], 1), 2)  # (b_s, d)
        b = F.normalize(b - mean.repeat(b.shape[0], 1), 2)  # (b_s, d)
        dist = 1 - torch.matmul(a, b.t())  # (b_s, b_s)
        dist = dist / (torch.min(dist, -1, keepdim=True)[0] + epsilon)
        aff = F.softmax(1 - dist / h, dim=1)
        return aff

    def compute_contextual(self, batch_gen, batch_art_img_name):
        loss_cx_full = 0
        for b in range(batch_gen.shape[0]):
            art_img_name = batch_art_img_name[b]
            gen = batch_gen[b].unsqueeze(0)

            self.imsize = gen.shape[2]
            channels = gen.shape[1]
            gen = gen.view(channels, self.imsize, self.imsize)

            if not os.path.isfile(self.artistic_masks_dir + '/' + art_img_name + '.pkl'):
                artistic_masks = []
            else:
                artistic_masks = self.load_obj(self.artistic_masks_dir + '/' + art_img_name)

            gens = []
            for s in range(self.scales):
                patch_size = self.patch_sizes[s]
                stride = self.strides[s]
                gens.append(gen.unfold(1, patch_size, stride).unfold(2, patch_size, stride).permute(1, 2, 0, 3, 4).contiguous().view(-1, channels, patch_size, patch_size).view(-1, patch_size*patch_size*channels))

            artistic_masks_unfolded = []
            for s in range(self.scales):
                artistic_masks_unfolded.append([])

            for s in range(self.scales):
                patch_size = self.patch_sizes[s]
                stride = self.strides[s]
                for i in range(len(artistic_masks)):
                    mask = artistic_masks[i][1]
                    mask = torch.from_numpy(mask).view(1, mask.shape[0], mask.shape[1])
                    artistic_masks_unfolded[s].append((artistic_masks[i][0], mask.unfold(1, patch_size, stride).unfold(2, patch_size, stride).permute(1, 2, 0, 3, 4).contiguous().view(-1, 1, patch_size, patch_size).view(-1, patch_size*patch_size)))

            for s in range(self.scales):
                patch_size = self.patch_sizes[s]
                stride = self.strides[s]
                if len(artistic_masks_unfolded[s]) > 0:
                    bck_mask = 1 - torch.clamp(torch.sum(torch.cat([torch.unsqueeze(x[1], 0) for x in artistic_masks_unfolded[s]]), 0), 0, 1)
                    artistic_masks_unfolded[s].append((0, bck_mask))
                else:
                    bck_mask = torch.ones((gen.shape[1], gen.shape[2]))
                    bck_mask = bck_mask.view(1, bck_mask.shape[0], bck_mask.shape[1])
                    artistic_masks_unfolded[s].append((0, bck_mask.unfold(1, patch_size, stride).unfold(2, patch_size, stride).permute(1, 2, 0, 3, 4).contiguous().view(-1, 1, patch_size, patch_size).view(-1, patch_size* patch_size)))

            gen_classes = []
            for s in range(self.scales):
                gen_classes.append({})

            for s in range(self.scales):
                for m in artistic_masks_unfolded[s]:
                    condition = torch.sum(m[1].cuda(), -1).float() >= m[1].shape[1] / 2.0
                    valid_idxs = torch.nonzero(condition).view(-1)
                    if valid_idxs.numel() > 0:
                        if m[0] not in gen_classes[s]:
                            gen_classes[s][m[0]] = torch.index_select(gens[s], 0, valid_idxs)
                        else:
                            gen_classes[s][m[0]] = torch.cat((gen_classes[s][m[0]], torch.index_select(gens[s], 0, valid_idxs)))

            cx_loss = 0

            for s in range(self.scales):
                patch_size = self.patch_sizes[s]

                for cl, patches in gen_classes[s].items():
                    if not self.indexes[s]: #we didn't load the indexes
                        if os.path.isfile(self.which_mem_bank + '/patch_size_' + str(patch_size) + '/faiss_indexes/' + str(cl) + '.index'):
                            D, I = faiss.read_index(self.which_mem_bank + '/patch_size_' + str(patch_size) + '/faiss_indexes/' + str(cl) + '.index').search(patches.cpu().detach().numpy(), self.k)
                        else:
                            D, I = faiss.read_index(self.which_mem_bank + '/patch_size_' + str(patch_size) + '/faiss_indexes/0.index').search(patches.cpu().detach().numpy(), self.k) # if a class found in the artistic image is not found in the memory bank, we use patches from the background
                    else: #we loaded all the indexes
                        if str(cl) in self.indexes[s]:
                            D, I = self.indexes[s][str(cl)].search(patches.cpu().detach().numpy(), self.k)
                        else:
                            D, I = self.indexes[s]['0'].search(patches.cpu().detach().numpy(), self.k)

                    I = I.flatten()
                    I = np.unique(I)
                    I = I[I >= 0]
                    I = np.sort(I)
                    if I.size == 0:
                        continue

                    if not self.mem_banks[s]:  # we didn't load the memory banks
                        if os.path.isfile(self.which_mem_bank + '/patch_size_' + str(patch_size) + '/memory_bank_patches_binary/' + str(cl) + '.bin'):
                            nearest_real_patches = torch.from_numpy(h5_vs_binary.binary_read(open(self.which_mem_bank + '/patch_size_' + str(patch_size) + '/memory_bank_patches_binary/' + str(cl) + '.bin','r'), gens[s].shape[1], I)).float()
                            mean = torch.from_numpy(np.load(self.which_mem_bank + '/patch_size_'+ str(patch_size) +'/means/' + str(cl) + '.npy'))
                        else:
                            nearest_real_patches = torch.from_numpy(h5_vs_binary.binary_read(open(self.which_mem_bank + '/patch_size_'+ str(patch_size) +'/memory_bank_patches_binary/0.bin', 'r'), gens[s].shape[1], I)).float()
                            mean = torch.from_numpy(np.load(self.which_mem_bank + '/patch_size_'+ str(patch_size) +'/means/0.npy'))
                    else:  # we loaded the memory banks
                        if str(cl) in self.mem_banks[s]:
                            nearest_real_patches = torch.from_numpy(self.mem_banks[s][str(cl)][I]).float()
                            mean = torch.from_numpy(self.means[s][str(cl)])
                        else:
                            nearest_real_patches = torch.from_numpy(self.mem_banks[s]['0'][I]).float()
                            mean = torch.from_numpy(self.means[s]['0'])

                    aff1 = self.affinities(gen_classes[s][cl].cuda(), ((nearest_real_patches - 0.5) * 2).cuda(), mean.cuda())
                    maxes1, indices = torch.max(aff1, 1, keepdim=True)

                    cx_loss -= torch.log(torch.mean(maxes1))

            loss_cx_full += cx_loss

        return loss_cx_full / batch_gen.shape[0]
