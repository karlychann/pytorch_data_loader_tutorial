import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os


def is_valid_image_file(filename):
  # Check file name extension
  valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
  if os.path.splitext(filename)[1].lower() not in valid_extensions:
    print(f"Invalid image file extension \"{filename}\". Skipping this file...")
  # Verify that image file is intact
  try:
    with Image.open(filename) as img:
      img.verify()  # Verify if it's an image
      return True
  except (IOError, SyntaxError) as e:
    print(f"Invalid image file {filename}: {e}")
    return False


class HymenopteraDataset(Dataset):
  def __init__(self, img_dir, transform=None, target_transform=None):
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform

    self.class_dict = {0: "ant", 1: "bee"}

    image_label_dict = {}
    class_counts = {"ant": 0, "bee": 0}
    for filename in os.listdir(img_dir):
      if is_valid_image_file(os.path.join(self.img_dir, filename)):
        last_char_before_ext = os.path.splitext(filename)[0][-1]
        if last_char_before_ext.isdigit() and int(last_char_before_ext) in self.class_dict.keys():
          img_class = int(last_char_before_ext)
          image_label_dict[filename] = img_class
          class_counts[self.class_dict[img_class]] += 1
          print("Image loaded:", filename)

    # self.items will look something like: [ ("im1.jpg", 0), ("img2.png", 1), ... ]
    self.items = list(image_label_dict.items())

    print("Class counts:", class_counts) 
    

  def __len__(self):
    return len(self.items)


  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.items[idx][0])
    # image = torchvision.io.read_image(img_path)
    image = Image.open(img_path)
    label = self.items[idx][1]

    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    
    return image, label

    

