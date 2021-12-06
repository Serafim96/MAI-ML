import numpy as np
import pandas as pd
import os
from time import time

import matplotlib.pyplot as plt 
import seaborn as sns 


class NeuralNetwork:
  def __init__(self, layers = []):
    self.layers = layers
  

  def add(self, layer):
    self.layers.append(layer)
  

  def forward(self, x):
    for layer in self.layers:
      x = layer.forward(x)
    return x
  

  def backward(self, z):
    for layer in self.layers[::-1]:
      z = layer.backward(z)
    return z
  

  def update(self, learning_rate: float):
    for layer in self.layers:
      layer.update(learning_rate)



class NeuralNetworkLayer:
  def __init__(self, n_inputs:int, n_outputs: int, activation_func):
    self.activation_func = activation_func
    self.weights = np.random.normal(0, 1.0 / np.sqrt(n_inputs), (n_outputs, n_inputs))
    self.bias = np.zeros((1, n_outputs))
    self.d_weights = np.zeros_like(self.weights)
    self.d_bias = np.zeros_like(self.bias)


  def forward(self, x):
    self.x = x
    pred = np.dot(x, self.weights.T)
    pred = pred + self.bias
    outputs = self.activation_func.forward(pred)
    return outputs
  

  def backward(self, dz):
    dz = self.activation_func.backward(dz)
    dx = np.dot(dz, self.weights)
    d_weights = np.dot(dz.T, self.x)
    d_bias = dz.sum(axis=0)
    self.d_weights = d_weights
    self.d_bias = d_bias
    return dx


  def update(self, learning_rate: float):
    self.weights -= learning_rate * self.d_weights
    self.bias -= learning_rate * self.d_bias


class ActivationFunction:
  def __init__(self, func: str='softmax'):
    self.func = func
    self.__set_forward__(func)
    self.__set_backward__(func)


  def forward(self, z):
    self.forward(self, z)
  

  def backward(self, dp):
    self.backward(self, dp)
      

  def __set_forward__(self, func):
    if self.func == 'softmax':
      self.forward = self.__softmax_forward__
    elif self.func == 'tanh':
      self.forward = self.__tanh_forward__
    elif self.func == 'sigmoid':
      self.forward = self.__sigmoid_forward__
    else:
      self.forward = self.__softmax_forward__
        

  def __set_backward__(self, func):
    if self.func == 'softmax':
      self.backward = self.__softmax_backward__
    elif self.func == 'tanh':
      self.backward = self.__tanh_backward__
    elif self.func == 'sigmoid':
      self.backward = self.__sigmoid_backward__
    else:
      self.backward = self.__softmax_backward__


  def __sigmoid_forward__(self, x):
    self.sigmoid_x = x
    a = (1/(1+np.exp(-x)))
    self.sigmoid_a = a
    return a


  def __sigmoid_backward__(self, dy):
    return ((self.sigmoid_a) * (1 - self.sigmoid_a)) * dy


  def __softmax_backward__(self, dp):
    p = self.forward(self.softmax_z)
    pdp = p * dp
    result = pdp - p * pdp.sum(axis=1, keepdims=True)
    return result


  def __softmax_forward__(self, z):
    self.softmax_z = z
    zmax = z.max(axis=1, keepdims=True)
    expz = np.exp(z - zmax)
    Z = expz.sum(axis=1, keepdims=True)
    probabilities = expz / Z
    return probabilities


  def __tanh_forward__(self, z):
    y = np.tanh(z)
    self.tanh_y = y
    return y
  
  def __tanh_backward__(self, dy):
    return (1.0 - self.tanh_y**2) * dy



class LossFunction:
  def __init__(self, func: str='cross-entropy'):
    self.func = func
    self.__set_forward__(func)
    self.__set_backward__(func)


  def forward(self, p, y):
    self.forward(self, p, y)
  

  def backward(self, loss):
    self.backward(self, loss)
      

  def __set_forward__(self, func):
    if self.func == 'cross-entropy':
      self.forward = self.__cross_entropy_forward__
    else:
      self.forward = self.__cross_entropy_forward__
        

  def __set_backward__(self, func):
    if self.func == 'cross-entropy':
      self.backward = self.__cross_entropy_backward__
    else:
      self.backward = self.__cross_entropy_backward__


  def __cross_entropy_forward__(self, p, y):
    self.p = p
    self.y = y
    p_of_y = p[np.arange(len(y)), y]
    self.log_prob = np.log(p_of_y)
    result = -self.log_prob.mean()
    return result
  

  def __cross_entropy_backward__(self, loss):
    dlog_softmax = np.zeros_like(self.p)
    dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0 / len(self.y)
    result = dlog_softmax / self.p
    return result



class NeuralNetworkTrainer:
  def __init__(self, mini_batch_size=4, learning_rate=0.01):
    # self.network = network
    self.mini_batch_size = mini_batch_size
    self.learning_rate = learning_rate
  

  def train_epoch(self, net, x_train, y_train, loss):
    for i in range(0, len(x_train), self.mini_batch_size):
      x_batch = x_train[i:i + self.mini_batch_size]
      y_batch = y_train[i:i + self.mini_batch_size]

      pred = net.forward(x_batch)
      losses = loss.forward(pred, y_batch)
  
      d_pred = loss.backward(losses)
      dx = net.backward(d_pred)
      net.update(self.learning_rate)


  def train_epoch_by_batches(self, net: NeuralNetwork, x_train, y_train, x_test, y_test, 
                             n_epoch: int, loss: LossFunction, n_batches=5):
    metrics = NeuralNetworkMetrics()
    
    train_accurancy = np.empty((n_epoch + 1, 3))
    train_accurancy[:] = np.NAN
    valid_accurancy = np.empty((n_epoch + 1, 3))
    valid_accurancy[:] = np.NAN
    start_time = time()
    
    x_batches = np.split(x_train, n_batches)
    y_batches = np.split(y_train, n_batches)
    for epoch in range(n_epoch + 1):
      if epoch != 0:
        for i in range(n_batches):
          self.train_epoch(net, x_batches[i], y_batches[i], loss)
    
      t_loss, t_accuracy = metrics.get_avg_loss_accurancy(net, x_train, y_train, loss, n_batches)
      train_accurancy[epoch, :] = [epoch, t_loss, t_accuracy]

      v_loss, v_accuracy = metrics.get_avg_loss_accurancy(net, x_test, y_test, loss, n_batches)
      valid_accurancy[epoch, :] = [epoch, v_loss, v_accuracy]
      
      if epoch == 0:
        print(f'Initial train loss={t_loss:.3f}, accuracy={t_accuracy:.3f}')
        print(f'Initial valid loss={v_loss:.3f}, accuracy={v_accuracy:.3f}')
    end_time = time()

    train_time = end_time - start_time

    print(f'Final train loss={train_accurancy[-1, 1]:.3f}, accuracy={train_accurancy[-1, 2]:.3f}')
    print(f'Final valid loss={valid_accurancy[-1, 1]:.3f}, accuracy={valid_accurancy[-1, 2]:.3f}')

    return train_accurancy, valid_accurancy, train_time



class NeuralNetworkMetrics:
  def get_avg_loss_accurancy(self, net, x, y, loss, n_batches=5):
    sum_losses = np.zeros((n_batches,))
    sum_accurancy = np.zeros((n_batches,))

    x_batches = np.split(x, n_batches)
    y_batches = np.split(y, n_batches)

    for i in range(n_batches):
      losses, accurancy = self.__get_loss_mean_accurancy__(net, x_batches[i], y_batches[i], loss)
      sum_losses[i] = losses
      sum_accurancy[i] = accurancy

    return sum_losses.mean(), sum_accurancy.mean()


  def __get_loss_mean_accurancy__(self, net, x, y, loss):
    pred = net.forward(x)
    losses = loss.forward(pred, y)
    pred_max = np.argmax(pred, axis=1)
    accurancy = (pred_max == y).mean()
    return losses, accurancy


  def get_predicts(self, net, x, y, loss, n_batches=5):
    predicts = []
    x_batches = np.split(x, n_batches)
    y_batches = np.split(y, n_batches)

    for i in range(n_batches):
      predicts.append(self.__get_predicts__(net, x_batches[i], y_batches[i], loss))

    return predicts


  def __get_predicts__(self, net, x, y, loss):
    pred = net.forward(x)
    losses = loss.forward(pred, y)
    pred_max = np.argmax(pred, axis=1)
    # accurancy = (pred_max == y).mean()
    return pred_max



# -------------------------------------------------------------------------------------------------------
# функция построения матрицы ошибок (соответствия)
def draw_confusion_matrix(df_confusion, title: str, img_dir: str, img_format: str, description: str):
  fig, ax = plt.subplots(figsize=(12, 6))
  sns.heatmap(df_confusion, annot=True, cmap="RdYlBu", linewidths=.5, fmt='.2f', ax=ax);
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  ax.text(x=1.2, y=0.5, s=description, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=props)

  plt.title(title)
  plt.savefig(os.path.join(img_dir, f'{title}{img_format}'), bbox_inches='tight')
  plt.show()


def plot_confusion_matrix(net: NeuralNetwork, x_test, y_test, loss: LossFunction, 
                          n_batches: int, metrics: NeuralNetworkMetrics, 
                          db_name: str, img_dir: str, img_format: str, description: str):
  y_pred = metrics.get_predicts(net, x_test, y_test, loss, n_batches)
  y_pred =  np.array(y_pred).flatten()

  if not os.path.isdir(img_dir):
    os.mkdir(img_dir)

  df_y_test = pd.Series(y_test, name='Actual Class')
  df_y_pred = pd.Series(y_pred, name='Predicted Class')

  df_confusion = pd.crosstab(df_y_test, df_y_pred)
  title = f'{db_name}. Confussion Matrix (not normalized)'
  draw_confusion_matrix(df_confusion, title, img_dir, img_format, description)
  
  df_confusion = df_confusion / df_confusion.sum(axis=1)
  title = f'{db_name}. Confussion Matrix (normalized)'
  draw_confusion_matrix(df_confusion, title, img_dir, img_format, description)



# функция обучения и построения графика
def train_and_plot(net: NeuralNetwork, x_train, y_train, x_test, y_test, 
                   trainer: NeuralNetworkTrainer, n_epoch: int, 
                   loss: LossFunction, n_batches: int, 
                   db_name: str, img_dir: str, img_format: str):
  
  title = f'{db_name}. MPL Train and Validation'
  img_name = f'{title}{img_format}'

  train_accurancy, valid_accurancy, train_time \
        = trainer.train_epoch_by_batches(net, x_train, y_train, x_test, y_test, n_epoch, loss, n_batches)
  
  x = train_accurancy[:, 0]
  y = (train_accurancy[:, 2], valid_accurancy[:, 2])

  description = f'Database: {db_name}\
  \nLayers: {len(net.layers)}\
  \nActivation Funcs: {[x.activation_func.func for x in net.layers]}\
  \nEpoches: {n_epoch}\
  \nBatches: {n_batches}\
  \nMini Batch Size: {trainer.mini_batch_size}\
  \nLearning Rate: {trainer.learning_rate:.3f}\
  \nLoss Function: {loss.func}\
  \nTrain Time: {train_time:.3f} sec.\
  \nFinal Train Loss: {train_accurancy[-1, 1]:.3f}\
  \nFinal Train Accuracy: {train_accurancy[-1, 2]:.3f}\
  \nFinal Valid Loss: {valid_accurancy[-1, 1]:.3f}\
  \nFinal Valid Accuracy: {valid_accurancy[-1, 2]:.3f}'

  fig, ax = plt.subplots(figsize=(10, 8))
  ax.plot(x, y[0], 'k--', label='training accuracy')
  ax.plot(x, y[1], 'g-', label='validation accuracy')
  ax.legend(loc='lower right')
  ax.set_ylabel('Acurancy')
  ax.set_xlabel('Epoches')
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  ax.text(x=1.2, y=0.5, s=description, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=props)

  ax.set_title(f'{title}')
  
  if not os.path.isdir(img_dir):
    os.mkdir(img_dir)

  plt.savefig(os.path.join(img_dir, img_name), bbox_inches='tight')
  plt.show()
  return description