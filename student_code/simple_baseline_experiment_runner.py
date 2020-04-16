from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset
# from simple_baseline_net import SimpleBaselineNet
# from experiment_runner_base import ExperimentRunnerBase
# from vqa_dataset import VqaDataset
from torchvision import transforms as transforms
import torch
import torch.nn.functional as F
import ipdb


class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers, cache_location, lr, log_validation,writer):

        ############ 2.3 TODO: set up transform

        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        ############

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   transform=transform,
                                   ############ 2.4 TODO: fill in the arguments
                                   question_word_to_id_map=None,                  #Change Later
                                   answer_to_id_map=None,                         #Change Later
                                   ############
                                   )
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_train2014_{}.jpg", #image_filename_pattern="COCO_val2014_{}.jpg",
                                 transform=transform,
                                 ############ 2.4 TODO: fill in the arguments
                                 question_word_to_id_map=train_dataset.question_word_to_id_map,
                                 answer_to_id_map=train_dataset.answer_to_id_map                           
                                 ############
                                 )

        model = SimpleBaselineNet(train_dataset.question_word_list_length,train_dataset.answer_list_length)

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers)

        self.SGD_lr_word_embedding = 0.8
        self.SGD_lr_softmax_layer = 1e-2
        self.SGD_momentum = 0.9
        ############ 2.5 TODO: set up optimizer
        self.optimizer = torch.optim.SGD([{'params': self._model.get_word_embedding.parameters(), 'lr': self.SGD_lr_word_embedding},
                                                {'params': self._model.fc.parameters(), 'lr': self.SGD_lr_softmax_layer}
                                               ], momentum=self.SGD_momentum)
        ############
        self.grad_clip = 20
        self.softmax_weight_threshold = 20
        self.word_embedding_layer_weight_threshold = 1500
        self.lr_decrease_factor = 0.1
        self.lr_decay_freq = 5
        self.writer=writer

    def _optimize(self, predicted_answers, true_answer_ids):
        ############ 2.7 TODO: compute the loss, run back propagation, take optimization step.
        self.optimizer.zero_grad()
        loss = F.cross_entropy(input=predicted_answers,target=true_answer_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip)
        self.optimizer.step()
        self._model.get_word_embedding.weight.data.clamp_(min=-1*self.word_embedding_layer_weight_threshold, max=self.word_embedding_layer_weight_threshold)
        self._model.fc.weight.data.clamp_(min=-1*self.softmax_weight_threshold, max=self.softmax_weight_threshold)
        return loss
        ############
