#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.

        super(CharDecoder, self).__init__()
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, padding_idx = 0)
        self.target_vocab = target_vocab

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        char_embedding = self.decoderCharEmb(input) #note : now input :tensor of (length, batch, char_Embedding_size)
        output, dec_hidden = self.charDecoder(char_embedding, dec_hidden)
        scores = self.char_output_projection(output)
        return scores, dec_hidden
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        
        scores, dec_hiden = self.forward(char_sequence[0: -1], dec_hidden) #scores: (length, batch, self.vocab.size)
        loss = nn.CrossEntropyLoss(ignore_index = 0, reduction = 'sum')

        #input: (batch_size, num_classes)--> (length+batch_size, self.vocab.size)   target:(batch_size)-->(length+batch_size)
        input = scores.view((-1, scores.shape[-1])) #input: (batch, length*self.vocab.size)
        target = char_sequence[1:].contiguous().view(-1)
        #output: (length+batch_size)
        #loss computes each batch's loss
        output = loss(input, target)
        return output



        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        output_word = []
        current_char_ind = self.target_vocab.start_of_word
        batch_size = initialStates[0].shape[1]
        dec_hidden = initialStates
        current_char = torch.tensor([[current_char_ind] * batch_size], dtype = torch.long, device = device)
        #current_char: (1, batch_size)
        for t in range(max_length):      
            scores, dec_hidden = self.forward(current_char, dec_hidden) #scores: (1, batch, self.vocab_size)
            current_char_ls = scores.argmax(-1).tolist()[0]
            output_word.append(current_char_ls)
            current_char = scores.argmax(-1)
        #output_word: list of length (max_length, batch_size) 
        decodeWord = torch.tensor(output_word, dtype = torch.long, device = device).transpose(0, 1).tolist() # (batch_size, max_length)
                    
        decodeWords = []
        for batch in decodeWord:
            if self.target_vocab.end_of_word in batch:
                del batch[batch.index(self.target_vocab.end_of_word):]
            s = ''
            for i in batch:
                s += self.target_vocab.id2char[i]
            decodeWords.append(s)

        return decodeWords

        ### END YOUR CODE

