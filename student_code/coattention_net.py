import torch.nn as nn
import torch
import ipdb


class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self,question_word_list_length,max_question_length,embedding_length):
        super().__init__()
        ############ 3.3 TODO
        self.get_word_embedding = nn.Embedding(question_word_list_length, embedding_length)
        self.get_unigram = nn.Conv1d(max_question_length, max_question_length, 1, stride=1, padding=0)
        self.get_bigram = nn.Conv1d(max_question_length, max_question_length, 2, stride=1, padding=1,dilation=2)       #Dilating to retain size
        self.get_trigram = nn.Conv1d(max_question_length, max_question_length, 3, stride=1, padding=2,dilation=2)      #Dilating to retain size
        self.tanh_activation = nn.Tanh()
        self.max_pool = nn.MaxPool2d((3, 1))
        self.lstm = nn.LSTM(input_size=embedding_length, hidden_size=embedding_length, num_layers=3, dropout=0.5,batch_first=True)
        ############ 

    def forward(self, image, question_encoding):
        ############ 3.3 TODO
        ipdb.set_trace()
        question_encoding_max,indices = torch.max(question_encoding,dim=2)        
        word_features = self.get_word_embedding(indices)
        unigram = self.tanh_activation(self.get_unigram(word_features))
        bigram = self.tanh_activation(self.get_bigram(word_features))
        trigram = self.tanh_activation(self.get_trigram(word_features))
        phrase_embedding = self.max_pool(torch.cat((unigram.unsqueeze(1), bigram.unsqueeze(1), trigram.unsqueeze(1)), 1).permute(0, 2, 1, 3))
        phrase_embedding = phrase_embedding.squeeze()
        _, (hn, cn) = self.lstm(phrase_embedding)

        #This is not complete
        ipdb.set_trace()
        ############ 
        raise NotImplementedError()

    # def alternating_coattention(self,X,g):

