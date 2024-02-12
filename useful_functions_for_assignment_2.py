import os
import pickle

####  Useful functions for exploring/indexing into directories #####

# Get the names of all of the items (files or directories) within a directory
# (here, assuming that you have these three items in /content/drive)
os.listdir("/content/drive/")  # returns ["car.png", "my_subfolder", "a_file.txt"]

# Get the file name from a string containing a path
os.path.basename("/content/drive/car.png")  # returns "car.png"

# Get the directory path from a string containing a file path
os.path.dirname("/content/drive/car.png")  # returns "/content/drive"

# Check if something is a file (returns True if it is a file, False if it is a directory or doesn't exist)
os.path.isfile("/content/drive/car.png")  # Returns True
os.path.isfile("/content/drive")  # Returns False

# Check if something is a directory (returns True if it is a directory, False if it is a file or doesn't exist)
os.path.isdir("/content/drive/car.png")  # Returns False
os.path.isdir("/content/drive")  # Returns True

# Safely construct directory paths
os.path.join("directory_1", "subdirectory_c", "list_of_dogs.txt")  # returns "directory_1/subdirectory_c/list_of_dogs.txt"




#### Useful functions for using dictionaries, lists, and pickle ####

# The biased_cars_1 dataset is stored as a python dictionary, in a pickle file. 

# Below, we make a new dictionary called "friend_dict". This is like a lookup table where you enter a key and receive a value. 
# In this example, our keys are the names of some friends. Each value is a Python list of 3 integers, which represent (in order) how many dogs, how many cats, and how many ferrets each friend has. 
# We assume that Lisa has 2 dogs, 1 cat, and 0 ferrets. Bart has no dogs, no cats, and 11 ferrets. 

friend_dict = {
    "Lisa": [2, 1, 0],
    "Bart": [0, 0, 11]
}

# We can find out how many of each type of pet Lisa has by doing the following: 
lisa_pets = friend_dict["Lisa"]  # lisa_pets will be set to [1, 1, 0]

# We can find out how many cats Lisa has by indexing into lisa_pets (to get the second value with index=1):
lisa_cat_count = lisa_pets[1]  # lisa_cat_count will be set to 1. 

# We can add a new key-value pair to our dictionary like so (for our friend Milhouse, who has one dog): 
friend_dict["Milhouse"] = [1, 0, 0]

# We can check if a given KEY is in our dictionary using the following: 
if "Stewie" in friend_dict:
  print("Stewie is in our dictionary")
else:
  print("Stewie is not in our dictionary")


# We can save our important data to a pickle file on the disk
file = open('my_friends_pets.p', 'wb')
pickle.dump(friend_dict, file)
file.close()

# We can then load it in a future python session
file = open('my_friends_pets.p', 'rb')
friend_dict = pickle.load(file)
file.close()
