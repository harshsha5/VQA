import torch.nn as nn
import torch.nn.functional as F
from external.googlenet.googlenet import GoogLeNet
import torch
import ipdb


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self,question_word_list_length,num_possible_answers):
        super().__init__()
	    ############ 2.2 TODO
        self.image_network = GoogLeNet(aux_logits=False)
        self.get_word_embedding = nn.Linear(question_word_list_length,1000) #1000 to ensure image and word get the same representative power
        self.fc = nn.Linear(2000, num_possible_answers)
	    ############

    def convert_one_hot2_bow(self,question_encoding):
        '''
        Input: question_encoding (26*question_vocab_length)
        Output: bag of word encoding of the question (1*question_vocab_length)
        '''
        return torch.clamp(torch.sum(question_encoding, dim=1), min=0, max=1)

    def forward(self, image, question_encoding):
	    ############ 2.2 TODO
        image_features = self.image_network(image)
        bow_embedding = self.convert_one_hot2_bow(question_encoding)
        word_features = self.get_word_embedding(bow_embedding)
        net_features = torch.cat((image_features,word_features), dim=1)
        return self.fc(net_features)
        # out = F.softmax(net_features,dim=1)
	    ############
