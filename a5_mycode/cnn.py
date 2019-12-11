#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    """
    impletement the CNN network in the step 3
    1-dimensional convolution neraul network
    """
    def __init__(self, embedding_char_size, embedding_word_size, max_word_length, kernels = 5):
        super(CNN, self).__init__()
        """
        @param: k: int, kernel size(or window size)
        @param: f(set to e_word): int, the number of the filters(number of output channels) 
                : set to equal e_word, final word embeddings size
        @param: max_word_length(m_word) : int
        """
        self.kernels = kernels
        self.embedding_char_size = embedding_char_size
        self.filters = embedding_word_size
        self.max_word_length = max_word_length
        self.conv1d = nn.Conv1d(in_channels = self.embedding_char_size, out_channels = self.filters, kernel_size = self.kernels)
        self.maxpool1d = nn.MaxPool1d(kernel_size = self.max_word_length - self.kernels + 1)
        
    def forward(self, x_reshaped):
        """
        impletement forward functin for CNN, this function should can opertate on batches of words
        
        @param: x_reshaped: 
            tensor of (batch_size, char_embedding_size(also called e_char), max_word_length)
        
        @return x_conv_out : tensor of (batch_size, e_word)
        
        first apply conv1d, then relu, finally maxpool to get x_conv_out
        """
        x_conv = self.conv1d(x_reshaped) #x_conv : tensor of (batch_size, filters, m_word-self.kernels+1)
        x_conv_out = self.maxpool1d(F.relu_(x_conv)) #tensor of (batch_size, filters=e_word, 1)
        x_conv_out = torch.squeeze(x_conv_out, dim = 2) #tensor of (batch_size, e_word)
        return x_conv_out
### END YOUR CODE


