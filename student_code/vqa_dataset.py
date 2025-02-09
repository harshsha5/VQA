import torch
from torch.utils.data import Dataset
from external.vqa.vqa import VQA
import re
from collections import Counter
from PIL import Image
import torchvision
import os
import ipdb
import numpy as np

def convert_to_dict(tuple_list):
    d = {}
    for i,(key,value) in enumerate(tuple_list):
        d[key] = i
    return d

def save_dict2file(path,data):
    try:
        import cPickle as pickle
    except ImportError:  # python 3.x
        import pickle

    with open(path, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("dictionary saved!")

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern,
                 transform=None, question_word_to_id_map=None, answer_to_id_map=None, question_word_list_length=5746, answer_list_length=5216,
                 pre_encoder=None, cache_location=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self._vqa = VQA(annotation_file=annotation_json_file_path, question_file=question_json_file_path)
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern
        self._transform = transform
        self._max_question_length = 26
        self._num_answers_for_each_question = 10
        self.qIDs = self._vqa.getQuesIds()
        # self.imIDs = self._vqa.getImgIds()
        self.image_id_net_length = 12           #Check if this is the correct method
        self.question_dict_path = "Saved_Models/question_dict"
        self.answer_dict_path = "Saved_Models/answer_dict"

        # Publicly accessible dataset parameters
        self.question_word_list_length = question_word_list_length + 1
        self.unknown_question_word_index = question_word_list_length
        self.answer_list_length = answer_list_length + 1
        self.unknown_answer_index = answer_list_length
        self._pre_encoder = pre_encoder
        self._cache_location = cache_location
        if self._cache_location is not None:
            try:
                os.makedirs(self._cache_location)
            except OSError:
                pass

        #1.1-1.3 Task
        print("Number of Question ID's are: ",len(self._vqa.getQuesIds()))  #1.1
        # requested_Q_ID = 409380
        # # requested_Q_ID = 4870251
        # self._vqa.showQA(self._vqa.loadQA(requested_Q_ID))  #1.2
        # print("Image ID associated with Question ID ",requested_Q_ID, " is ",self._vqa.qa[requested_Q_ID]['image_id']) #1.2
        # print("Most voted answer is: ",self.get_most_voted_answer(requested_Q_ID)) #1.3

        # Create the question map if necessary
        if question_word_to_id_map is None:
            ############ 1.6 TODO
            question_list = []
            for question_id in self._vqa.getQuesIds():
                question_list.append(self._vqa.qqa[question_id]['question'])
            word_list = self._create_word_list(question_list)
            self.question_word_to_id_map = self._create_id_map(word_list,self.question_word_list_length)
            save_dict2file(self.question_dict_path,self.question_word_to_id_map)
            ############
        else:
            self.question_word_to_id_map = question_word_to_id_map

        # Create the answer map if necessary
        if answer_to_id_map is None:
            ############ 1.7 TODO
            answer_list = []
            for annotation in self._vqa.dataset['annotations']:
                for ans in annotation['answers']:
                    answer_list.append(ans['answer'])
            answer_list = self._clean_sentences(answer_list)       
            self.answer_to_id_map = self._create_id_map(answer_list,self.answer_list_length)
            save_dict2file(self.answer_dict_path,self.answer_to_id_map)
            ############
        else:
            self.answer_to_id_map = answer_to_id_map

    def get_most_voted_answer(self,question_id):
        cnt = Counter()
        annotation = self._vqa.loadQA(question_id)
        assert(len(annotation)==1)
        answers = annotation[0]['answers']
        for elt in answers:
            cnt[elt['answer']] += 1
        return cnt.most_common(1)[0][0]


    def _create_word_list(self, sentences):
        """
        Turn a list of sentences into a list of processed words (no punctuation, lowercase, etc)
        Args:
            sentences: a list of str, sentences to be splitted into words
        Return:
            A list of str, words from the split, order remained.
        """
        regex = re.compile('[,\.!?""'']')
        word_list = []
        ############ 1.4 TODO
        for sentence in sentences:
            sentence = regex.sub('', sentence)
            sentence = sentence.lower()
            sentence = sentence.split() 
            for word in sentence:
                word_list.append(word)
        return word_list

    def _clean_sentences(self,sentences):
        """
        Cleans a list of sentences into a list of processed sentences (no punctuation, lowercase, etc)
        Args:
            sentences: a list of str, sentences to be splitted into words
        Return:
            A list of str, cleaned sentences, order remained.
        """
        regex = re.compile('[,\.!?""'']')
        for i,sentence in enumerate(sentences):
            sentence = regex.sub('', sentence)
            sentence = sentence.lower()
            sentences[i] = sentence
        return sentences

    def _create_id_map(self, word_list, max_list_length):
        """
        Find the most common str in a list, then create a map from str to id (its rank in the frequency)
        Args:
            word_list: a list of str, where the most frequent elements are picked out
            max_list_length: the number of strs picked
        Return:
            A map (dict) from str to id (rank)
        """
        ############ 1.5 TODO
        dict_counter = Counter(word_list) 
        most_occur = dict_counter.most_common(max_list_length)
        return convert_to_dict(most_occur)
        ############

    # def encoder(self,input_sequence,tensor_length,id_map,encoder_type):
    #     for i,elt in enumerate(input_sequence):
    #         if(encoder_type=="question"):
    #             word_list = self._create_word_list([elt])
    #         elif(encoder_type=="answer"):
    #             word_list = elt
    #         intermediate_tensor = torch.zeros([1, tensor_length], dtype=torch.int32)

    #         for word in word_list:
    #             if word in id_map:
    #                 intermediate_tensor[0,id_map[word]] = 1
    #             else:
    #                 intermediate_tensor[0,tensor_length-1] = 1

    #         if(i==0):
    #             resultant_tensor = intermediate_tensor
    #         else:
    #             resultant_tensor = torch.cat((resultant_tensor,intermediate_tensor), dim=0)
    #     return resultant_tensor

    def encode_questions(self,question_list):
        '''
        Converts the  question_list to a tensor representation. Each question is represented as a self._max_question_length*_max_question_length tensor. 
        For each word a one hot vector is created (a row) in the above block. These are then stacked row-wise.
        '''
        for i,elt in enumerate(question_list):
            intermediate_tensor = torch.zeros([self._max_question_length, self.question_word_list_length])                
            word_list = self._create_word_list([elt])
            for j,word in enumerate(word_list):
                if word in self.question_word_to_id_map:
                    intermediate_tensor[j,self.question_word_to_id_map[word]] = 1
                else:
                    intermediate_tensor[j,self.question_word_list_length-1] = 1

            if(i==0):
                resultant_tensor = intermediate_tensor
            else:
                resultant_tensor = torch.cat((resultant_tensor,intermediate_tensor), dim=0)

        return resultant_tensor

    def encode_answers(self,answer_list):
        '''
        Converts the  answer_list to a tensor representation. Each question is represented as a self._num_answers_for_each_question*answer_list_length tensor. 
        For each word a one hot vector is created (a row) in the above block. These are then stacked row-wise.
        '''
        for i,elt in enumerate(answer_list):
            intermediate_tensor = torch.zeros([self._num_answers_for_each_question, self.answer_list_length])   
            elt = self._clean_sentences(elt)             
            for j,sentence in enumerate(elt):
                if sentence in self.answer_to_id_map:
                    intermediate_tensor[j,self.answer_to_id_map[sentence]] = 1
                else:
                    intermediate_tensor[j,self.answer_list_length-1] = 1

            if(i==0):
                resultant_tensor = intermediate_tensor
            else:
                resultant_tensor = torch.cat((resultant_tensor,intermediate_tensor), dim=0)

        return resultant_tensor
              
    def __len__(self):
        ############ 1.8 TODO
        return len(self._vqa.getQuesIds())
        ############

    def __getitem__(self, idx):
        """
        Load an item of the dataset
        Args:
            idx: index of the data item
        Return:
            A dict containing multiple torch tensors for image, question and answers.
        """

        ############ 1.9 TODO
        # figure out the idx-th item of dataset from the VQA API
        question_ID = self.qIDs[idx]
        image_id = self._vqa.qa[question_ID]['image_id']
        image_name = "0" * (self.image_id_net_length - len(str(image_id))) + str(image_id)
        ############

        if self._cache_location is not None and self._pre_encoder is not None:
            ############ 3.2 TODO
            # implement your caching and loading logic here
            encode_path = self._cache_location + '/' + image_name + '.npz'
            try:
                img = np.load(encode_path)['img']
            except FileNotFoundError:  
                fpath = self._image_dir + "/" + self._image_filename_pattern.replace("{}", str(image_name))
                img = Image.open(fpath)
                img =  img.convert("RGB")
                if(self._transform is not None):
                    img = self._transform(img)       #Make sure to_tensor is in the transform function
                else:
                    img = torchvision.transforms.ToTensor()(img)
                img = self._pre_encoder(img.unsqueeze(0))
                np.savez_compressed(encode_path, img=img.detach().numpy())
            ############
            #raise NotImplementedError()
        else:
            ############ 1.9 TODO
            # load the image from disk, apply self._transform (if not None)
            fpath = self._image_dir + "/" + self._image_filename_pattern.replace("{}", str(image_name))
            img = Image.open(fpath)
            img =  img.convert("RGB")
            if(self._transform is not None):
                img = self._transform(img)       #Make sure to_tensor is in the transform function
            else:
                img = torchvision.transforms.ToTensor()(img)
            ############

        ############ 1.9 TODO
        # load and encode the question and answers, convert to torch tensors
        # question_ID = self._vqa.getQuesIds(imgIds=[image_id])
        question_list,answer_list = self._vqa.get_QA(self._vqa.loadQA(question_ID))
        question_tensors = self.encode_questions(question_list)
        answer_tensors = self.encode_answers(answer_list)
        ############
        return {"image":img,
                "questions":question_tensors,
                "answers":answer_tensors}
