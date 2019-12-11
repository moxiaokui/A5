#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn
import torch.nn.functional as F


### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    """impletement highway idea"""
    def __init__(self, embedding_word_size):
        """
        @param dropout: dropout rate
        """
        super(Highway, self).__init__()
        self.embedding_word_size = embedding_word_size
        self.x_proj = nn.Linear(self.embedding_word_size, self.embedding_word_size)
        self.x_gate = nn.Linear(self.embedding_word_size, self.embedding_word_size)
        
    def forward(self, x_conv_out):
        """
        impletement highway forward
        @param x_conv_out: Step 4, after max pooling, tensor of (m_sen, e_word) 
        
        @return x_highway:  after highway, shape of (m_sen, e_word)
        """
        x_p = F.relu(self.x_proj(x_conv_out))
        x_g = torch.sigmoid(self.x_gate(x_p))
        x_highway = x_p * x_g + (1-x_p) * (x_conv_out)
        return x_highway
    

    
    
    