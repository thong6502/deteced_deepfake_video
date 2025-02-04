from torch.utils.data import Dataset
from PIL import Image
import os
from .config import TRANSFORM

class MyDataset(Dataset):
  def __init__(self, dataset_path ,transform = None):
    self.classes = sorted(os.listdir(dataset_path))
    self.img_paths = []
    self.labels = []
    self.transform = transform
    if self.transform is None:
      self.transform = TRANSFORM
      
    for idx, label in enumerate(self.classes):
      for file_name in os.listdir(os.path.join(dataset_path,label)):
        self.img_paths.append(os.path.join(dataset_path, label, file_name))
        self.labels.append(idx)

  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, index):
    img = Image.open(self.img_paths[index])
    img = self.transform(img)
    label = self.labels[index]
    return img, label
  
if __name__ == "__main__":
  import matplotlib.pyplot as plt
  dataset = MyDataset("../dataset")
  img, label = dataset[0]

  plt.imshow(img.permute(1,2,0))
  plt.title(dataset.classes[label])
  plt.show()