import unittest
import os
import sys
from torch.utils.data import DataLoader
from student_code.vqa_dataset import VqaDataset

sys.path.append(os.path.abspath('test/'))
import test_vqa_dataset as tvd

if __name__ == "__main__":
    t = tvd.TestVqaDataset()
    t.test_load_dataset()
