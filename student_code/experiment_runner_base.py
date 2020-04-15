from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from collections import Counter
import ipdb


class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, log_validation=False):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 250  # Steps

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        self._log_validation = log_validation

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self):
        ############ 2.8 TODO
        # Should return your validation accuracy

        ############

        if self._log_validation:
            ############ 2.9 TODO
            # you probably want to plot something here
            pass

            ############
        raise NotImplementedError()

    def get_most_voted_answer(self,answers):
        '''
        Sum up all the ten answers one hot vector. The max value here is the most popular answer. So we get it's index
        '''
        summed_tensors = torch.sum(answers,dim=1)
        _,max_indices = torch.max(summed_tensors,dim=1)
        return max_indices

    def train(self):

        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                ############ 2.6 TODO
                # Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                # answer_logits = F.softmax(output,dim=1)
                predicted_answer = self._model(batch_data['image'],batch_data['questions']) # TODO
                ground_truth_answer = self.get_most_voted_answer(batch_data['answers']) # TODO
                ############

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    ############ 2.9 TODO
                    # you probably want to plot something here

                    ############

                # if current_step % self._test_freq == 0:
                #     self._model.eval()
                #     val_accuracy = self.validate()
                #     print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                #     ############ 2.9 TODO
                #     # you probably want to plot something here

                #     ############

                if (epoch+1) % self.lr_decay_freq==0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * self.lr_decrease_factor
