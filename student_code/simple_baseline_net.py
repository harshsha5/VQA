import torch.nn as nn
from external.googlenet.googlenet import GoogLeNet


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self):
        super().__init__()
	    ############ 2.2 TODO
        self.image_network = GoogLeNet()
	    ############

    def forward(self, image, question_encoding):
	    ############ 2.2 TODO

	    ############
        raise NotImplementedError()
