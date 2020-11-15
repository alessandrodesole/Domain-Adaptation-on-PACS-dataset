import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Compute and plot images distribution among domains/classes

def compute_plot_distribution(classes, domains, train_dataset, test_dataset, val_dataset=None, class_dataset, domains_dataset):
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
    
  index = np.arange(len(classes))
  plt.figure(figsize=(15, 8))
  p1 = plt.bar(index, count_train_items.values(), color='royalblue', width=0.5)
  #p2 = plt.bar(index, count_val_items.values(), bottom=list(count_train_items.values()), color='limegreen')
  #p2 = plt.bar(index, count_test_items.values(), bottom=list(count_train_items.values()), color='darkorange', width=0.5)
  plt.xlabel('Domains', fontsize=25, labelpad=15)
  plt.ylabel('Count', fontsize=25, labelpad=15)
  #plt.title('Number of images by domain in PACS', fontsize=30, pad=15)
  plt.title('Training and Test image distribution', fontsize=30, pad=15)
  plt.xticks(index, classes, fontsize=20)
  #plt.legend((p1[0], p2[0]), ('train', 'test'), prop={'size': 20})
  plt.show()

  if val_dataset is not None:
    index = np.arange(len(classes))
    plt.figure(figsize=(15, 8))
    p1 = plt.bar(index, count_train_items.values(), color='royalblue', width=0.5)
    p2 = plt.bar(index, count_val_items.values(), bottom=list(count_train_items.values()), color='darkorange', width=0.5)
    p3 = plt.bar(index, count_test_items.values(), bottom=list(count_train_items.values()), color='limegreen', width=0.5)
    plt.xlabel('Classes', fontsize=25, labelpad=15)
    plt.ylabel('Count', fontsize=25, labelpad=15)
    #plt.title('Number of domains by class in PACS', fontsize=30, pad=15)
    plt.title('Training, Validation and Test image distribution', fontsize=30, pad=15)
    plt.xticks(index, classes, fontsize=20)
    plt.legend((p1[0], p2[0], p3[0]), ('train', 'validation','test'), prop={'size': 20})
    plt.show()
  
  return

# Show images

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    #mean = np.array([0.5, 0.5, 0.5])
    #std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
# Functions to plot training results 

def plot_loss(loss_train):
  epochs = range(NUM_EPOCHS)
  loss_train = np.array(loss_train)
  plt.plot(epochs, loss_train, linestyle='-', color='b', label='Training loss')
  plt.title('Training loss', fontsize=15, fontweight='bold')
  plt.xlabel('Epochs', fontsize=10, labelpad=7)
  plt.ylabel('Loss', fontsize=10, labelpad=10)
  plt.legend()
  plt.grid()
  plt.show()

def plot_loss_val(loss_train, loss_val):
  epochs = range(NUM_EPOCHS)
  loss_train = np.array(loss_train)
  loss_val = np.array(loss_val)
  plt.plot(epochs, loss_train, linestyle='-', color='b', label='Training loss')
  plt.plot(epochs, loss_val, linestyle='-', color='darkorange', label='Validation loss')
  plt.title('Training and Validation loss', fontsize=15, fontweight='bold')
  plt.xlabel('Epochs', fontsize=10, labelpad=7)
  plt.ylabel('Loss', fontsize=10, labelpad=10)
  plt.legend()
  plt.grid()
  plt.show()

def plot_loss_DANN(train_class_losses, train_domain_losses, test_domain_losses):
  epochs = range(NUM_EPOCHS)
  train_class_losses = np.array(train_class_losses)
  train_domain_losses = np.array(train_domain_losses)
  test_domain_losses = np.array(test_domain_losses)
  plt.plot(epochs, train_class_losses, linestyle='-', color='b', label='Source Class loss')
  plt.plot(epochs, train_domain_losses, linestyle='-', color='g', label='Source Domain loss')
  plt.plot(epochs, test_domain_losses, linestyle='-', color='r', label='Target Domain loss')
  plt.title('DANN losses', fontsize=15, fontweight='bold')
  plt.xlabel('Epochs', fontsize=10, labelpad=7)
  plt.ylabel('Loss', fontsize=10, labelpad=10)
  plt.legend()
  plt.grid()
  plt.show()

def plot_accuracy(acc_train):
  epochs = range(NUM_EPOCHS)
  acc_train = np.array(acc_train)
  plt.plot(epochs, acc_train, linestyle='-', color='b', label='Training accuracy')
  plt.title('Training accuracy', fontsize=15, fontweight='bold')
  plt.xlabel('Epochs', fontsize=10, labelpad=7)
  plt.ylabel('Accuracy', fontsize=10, labelpad=10)
  plt.legend()
  plt.grid()
  plt.show()

def plot_accuracy_val(acc_train, acc_val):
  epochs = range(NUM_EPOCHS)
  acc_train = np.array(acc_train)
  acc_val = np.array(acc_val)
  plt.plot(epochs, acc_train, linestyle='-', color='b', label='Training accuracy')
  plt.plot(epochs, acc_val, linestyle='-', color='darkorange', label='Validation accuracy')
  plt.title('Training and Validation accuracy', fontsize=15, fontweight='bold')
  plt.xlabel('Epochs', fontsize=10, labelpad=7)
  plt.ylabel('Accuracy', fontsize=10, labelpad=10)
  plt.legend()
  plt.grid()
  plt.show()
