#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code
        
        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.embedding_char_size = 50
        self.dropout_rate = 0.3
        self.max_word_length = 21
        self.embedding_word_size = embed_size
        
        pad_token_idx = vocab.char2id['<pad>']
        self.charEmbeddings = nn.Embedding(len(vocab.char2id), self.embedding_char_size, padding_idx = pad_token_idx)
        self.dropout = nn.Dropout(p = self.dropout_rate)
        
        #construct CNN
        self.CNN = CNN(self.embedding_char_size, self.embedding_word_size, self.max_word_length)
        
        #construct Highway
        self.highway = Highway(self.embedding_word_size)
        


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code


        ### YOUR CODE HERE for part 1j

        sentence_length, batch_size, _ = input.shape

        x_emb = self.charEmbeddings(input)
        #note: x_emb : tensor of (sentence_length, batch_size, max_word_length, e_char)
        
        x_emb = x_emb.view((sentence_length * batch_size, self.max_word_length, self.embedding_char_size)).transpose(1, 2)
        #now x_emb is : tensor of (sentence_length * batch_size, e_char, max_word_length)
        
        #cnn needs: input shape: x_reshaped:(batch_size, char_embedding_size(also called e_char), max_word_length)
        #           output shape: (batch_size, e_word)
        x_conv_out = self.CNN.forward(x_emb) 
        #x_conv_out : (batch_size * sentence_length, e_word=embed_size)
        x_highway = self.highway.forward(x_conv_out)
        x_word_emb = self.dropout(x_highway)
        #now x_word_emb is: tensor of (sentence_length * batch_size, embed_size)
        
        output = x_word_emb.view((sentence_length, batch_size, self.embedding_word_size))
    
        return output            

        ### END YOUR CODE