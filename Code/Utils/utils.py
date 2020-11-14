import pandas as pd
from google.colab import output

# Check if classes are distributed equally between training and test set

def check_classes(train_dataset, test_dataset, val_dataset=None):
  labels = []
  labels_test = []
  labels_val = []
  images = []
  images_test = []
  images_val = []

  for image, label in train_dataset:
    labels.append(label)
    images.append(image)

  for image_test, label_test in test_dataset:
    labels_test.append(label_test)
    images_test.append(image_test)

  if val_dataset is not None:
    for image_val, label_val in val_dataset:
      labels_val.append(label_val)
      images_val.append(image_val)


  l = pd.Series(labels)
  l_test = pd.Series(labels_test)

  print(l.value_counts(sort=False))
  print(l_test.value_counts(sort=False))

  if val_dataset is not None:
    l_val = pd.Series(labels_val)
    print(l_val.value_counts(sort=False))
  
  return
  
