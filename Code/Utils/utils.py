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

# Compute the Kullback-Leibler divergence

def kl_divergence(p, q):
      return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def compute_kl(train_dataset, test_dataset):
  lambdas = []
  results = [[], [], [], []]

  for val in l.value_counts(sort=False):
    lambdas.append(val / len(train_dataset))

  DS = 0

  for ims, ls in train_dataset:
    p = 1 / l.value_counts(sort=False)[ls]
    for imt, lt in test_dataset:
      q = 1 / l.value_counts(sort=False)[lt]
      KLD = lambdas[ls] * kl_divergence(p, q)
      DS += KLD

  domain_shift = DS / (len(train_dataset * test_dataset))

  print(domain_shift)
  return

def compute_means(train_dataset):
  means_1 = []
  means_2 = []
  means_3 = []
  stds_1 = []
  stds_2 = []
  stds_3 = []

  for img, lab in train_dataset:      
    means_1.append(torch.mean(img[0]))
    means_2.append(torch.mean(img[1]))
    means_3.append(torch.mean(img[2]))
    stds_1.append(img[0])
    stds_2.append(img[1])
    stds_3.append(img[2])

  stds_1 = torch.cat((stds_1), 0)
  stds_2 = torch.cat((stds_2), 0)
  stds_3 = torch.cat((stds_3), 0)
  mean_1 = torch.mean(torch.tensor(means_1))
  mean_2 = torch.mean(torch.tensor(means_2))
  mean_3 = torch.mean(torch.tensor(means_3))
  std_1 = torch.std(stds_1)
  std_2 = torch.std(stds_2)
  std_3 = torch.std(stds_3)

  print("Means = [{:.4f}, {:.4f}, {:.4f}]".format(mean_1.item(), mean_2.item(), mean_3.item()))
  print("Stds = [{:.4f}, {:.4f}, {:.4f}]".format(std_1.item(), std_2.item(), std_3.item()))
  
  return

# Images distribution among domains/classes

def compute_distribution(classes, domains, train_dataset, test_dataset, val_dataset=None, class_dataset, domains_dataset):
  count_train_items = {}
  count_test_items = {}
  count_val_items = {}
  count_domain_items = {}
  count_class_items = {}

  for c in classes:
    count_train_items[c] = 0
    count_test_items[c] = 0
    count_class_items[c] = 0
    count_val_items[c] = 0

  for d in domains:
    count_domain_items[d] = 0

  for image, label in train_dataset:
    i = classes[label]
    count_train_items[i] += 1

  for image, label in test_dataset:
    i = classes[label]
    count_test_items[i] += 1

  if val_dataset is not None:

    for image, label in val_dataset:
      i = classes[label]
      count_val_items[i] += 1

  for image, label in class_dataset:
    i = classes[label]
    count_class_items[i] += 1

  for image, label in domains_dataset:
    i = domains[label]
    count_domain_items[i] += 1

  print("Domains")
  print(count_domain_items)
  print()
  print("Classes")
  print(count_class_items)
  print()
  print("Train dataset")
  print(count_train_items)
  print()
  print("Test dataset")
  print(count_test_items)
  
  if val_dataset is not None:
    print()
    print("Val dataset")
    print(count_val_items)
  
  return
