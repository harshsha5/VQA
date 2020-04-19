from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from collections import Counter
import random
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
        self._test_freq = 400  # Steps
        self._batch_size = batch_size

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

    def accuracy(self,logits, y):
        _, preds = torch.max(logits, 1)
        return (preds == y).float().mean()

    def get_question(self,question_one_hot):
        stro = ''
        for row in range(question_one_hot.shape[1]):
           val,ind = torch.max(question_one_hot[0,row,:],dim=0)
           if(val.item()!=1):
              continue
           else:
              question_in_words = [key  for (key, value) in self.question_dictionary.items() if value == ind.item()]
              if(len(question_in_words)!=0):
                 stro+= question_in_words[0] + ' '
        return stro + '?'

    def validate(self,current_step):
        ############ 2.8 TODO
        # Should return your validation accuracy
        num_batches_to_validate = 20   #This is for faster debugging. Remove Later
        batch_end_point = 100
        net_validation_loss = 0
        net_validation_accuracy = 0
        randomly_selected_batches = [random.randint(0,batch_end_point) for _ in range(num_batches_to_validate)] #Randomly select 30 V batches
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            if(batch_id>batch_end_point):
                break
            if(batch_id not in randomly_selected_batches):
                continue
            predicted_answer = self._model(batch_data['image'].to(self.device),batch_data['questions'].to(self.device)) # TODO
            ground_truth_answer = self.get_most_voted_answer(batch_data['answers'].to(self.device)) 
            net_validation_loss += self._optimize(predicted_answer, ground_truth_answer).cpu()
            net_validation_accuracy += self.accuracy(predicted_answer,ground_truth_answer).cpu()
            if(self._batch_size==1):
               _, preds = torch.max(predicted_answer, 1)
               answer_pred_in_words = [key  for (key, value) in self.answer_dictionary.items() if value == preds.item()]
               gt_pred_in_words = [key  for (key, value) in self.answer_dictionary.items() if value == ground_truth_answer.item()]
               print(self.get_question(batch_data['questions']))
               if(len(answer_pred_in_words) !=0):
                    print("Predicted Answer: ",answer_pred_in_words[0])
               if(len(gt_pred_in_words) !=0):
                    print("Ground_Truth Answer: ",gt_pred_in_words[0])
               print("=========================================================")
        ############
        avg_validation_loss = net_validation_loss/num_batches_to_validate
        avg_validation_accuracy = net_validation_accuracy/num_batches_to_validate
        if self._log_validation:
            ############ 2.9 TODO
            # you probably want to plot something here
            print("In log validation")
            self.writer .add_scalar('Average_Validation/Loss', avg_validation_loss, current_step)
            ############
        return avg_validation_accuracy

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
                predicted_answer = self._model(batch_data['image'].to(self.device),batch_data['questions'].to(self.device)) # TODO
                ground_truth_answer = self.get_most_voted_answer(batch_data['answers'].to(self.device)) # TODO
                ############

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    self.writer.add_scalar('Train/Loss', loss, current_step)
                    ############

                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate(current_step)
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    self.writer .add_scalar('Average_Validation/Accuracy', val_accuracy, current_step)
                    ############

                if (epoch+1) % self.lr_decay_freq==0:
                    print("Decreasing lr")
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * self.lr_decrease_factor

            torch.save(self._model.state_dict(), self.MODEL_SAVE_PATH)
