import os
import shutil

# Define the paths
original_dataset_dir = 'hymenoptera_data'
new_dataset_dir = 'hymenoptera_data_reorg'

# Classes
classes = {'class_0': 'ant', 'class_1': 'bee'}

# Function to create directory if it doesn't exist
def create_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)

# Create new dataset structure
create_dir(new_dataset_dir)
for phase in ['train', 'val']:
  for class_id, class_name in classes.items():
    create_dir(os.path.join(new_dataset_dir, phase, class_name))

# Function to copy files to new structure
def reorganize_dataset(phase):
  phase_dir = os.path.join(original_dataset_dir, phase)
  for filename in os.listdir(phase_dir):
    if 'class_0' in filename:
      shutil.copy(os.path.join(phase_dir, filename),
                  os.path.join(new_dataset_dir, phase, 'ant', filename))
    elif 'class_1' in filename:
      shutil.copy(os.path.join(phase_dir, filename),
                  os.path.join(new_dataset_dir, phase, 'bee', filename))

# Reorganize the train and validation datasets
reorganize_dataset('train')
reorganize_dataset('val')
