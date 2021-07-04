'''
Author: your name
Date: 2021-05-14 19:53:44
LastEditTime: 2021-05-23 00:08:51
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /dl_2021_hw3/dl_2021_hw3/attention_vis.py
'''
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.pyplot import cm
import os
import torch 
from data import Corpus
import pandas as pd
import matplotlib.ticker as ticker



def self_attention_matrix(seq, att_matrix):
    att_matrix = att_matrix.cpu().numpy()
    df = pd.DataFrame(att_matrix,columns=seq,index=seq)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)
    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    fontdict = {'rotation': 90} 
    ax.set_xticklabels([''] + list(df.columns), fontdict=fontdict)
    ax.set_yticklabels([''] + list(df.index))
    plt.savefig(f'./visual/attention_weight', bbox_inches='tight')
    

if __name__ == "__main__":
    data = torch.load('./best_att_weights.pt')
    corpus = Corpus()
    seq = []
    for i in data['data'][:,5][12:28]:
        seq.append(corpus.vocab.itos[i])
    weight_matrix = data['att_weight'][0][5,12:28,12:28]
    os.makedirs("./visual", exist_ok=True)
    self_attention_matrix(seq, weight_matrix)
    




