import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

import tensorflow as tf
import tensorflow_datasets as tfds

class MNIST():
  def __init__(self, normalize: bool = True):
    tf.config.set_visible_devices([], 'GPU')
    (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

    if normalize:
      self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
    
    self.x_train = tf.convert_to_tensor(self.x_train)
    self.y_train = tf.convert_to_tensor(self.y_train)
    self.x_test = tf.convert_to_tensor(self.x_test)
    self.y_test = tf.convert_to_tensor(self.y_test)

  def args_convo2d(self, args: dict):
    default_args = {'filters': 10,
                    'kernel_size': (2, 2),
                    'strides': (1, 1),
                    'padding': 'valid',
                    'data_format': None,
                    'dilation_rate': (1, 1),
                    'groups': 1,
                    'activation': 'relu',
                    'use_bias': True,
                    'kernel_initializer': 'glorot_uniform',
                    'bias_initializer': 'zeros',
                    'kernel_regularizer': None,
                    'bias_regularizer': None,
                    'activity_regularizer': None,
                    'kernel_constraint': None,
                    'bias_constraint': None
                    }
    if len(args.items())!=0:
      for arg, value in args.items():
        default_args[arg] = value
    return default_args
  
  def args_dense(self, args: dict):
    default_args = {'units': 10,
                    'activation': 'relu',
                    'use_bias': True,
                    'kernel_initializer': 'glorot_uniform',
                    'bias_initializer': 'zeros',
                    'kernel_regularizer': None,
                    'bias_regularizer': None,
                    'activity_regularizer': None,
                    'kernel_constraint': None,
                    'bias_constraint': None,
                    }
    if len(args.items())!=0:
      for arg, value in args.items():
        default_args[arg] = value
    return default_args

  def args_flatten(self, args):
    default_args = {'data_format': None}
    if len(args.items())!=0:
      for arg, value in args.items():
        default_args[arg] = value
    return default_args
  
  def args_maxpool2d(self, args):
    default_args = {'pool_size': (2, 2),
                    'strides': None,
                    'padding': 'valid',
                    'data_format': None
                    }
    if len(args.items())!=0:
      for arg, value in args.items():
        default_args[arg] = value
    return default_args

  def args_drop(self, args):
    default_args = {'rate': 0.2, 
                    'noise_shape': None, 
                    'seed': None
                    }
    if len(args.items())!=0:
      for arg, value in args.items():
        default_args[arg] = value
    return default_args
  
  def initializeConvNetwork(self, layers: dict):
    self.model = tf.keras.Sequential()
    cnt = 0
    for name, args in layers:
      if name=='convo2d':
        args = self.args_convo2d(args)
        if cnt==0:
          self.model.add(tf.keras.layers.Conv2D(
              filters=args['filters'],
              kernel_size=args['kernel_size'],
              strides=args['strides'],
              padding=args['padding'],
              data_format=args['data_format'],
              dilation_rate=args['dilation_rate'],
              groups=args['groups'],
              activation=args['activation'],
              use_bias=args['use_bias'],
              kernel_initializer=args['kernel_initializer'],
              bias_initializer=args['bias_initializer'],
              kernel_regularizer=args['kernel_regularizer'],
              bias_regularizer=args['bias_regularizer'],
              activity_regularizer=args['activity_regularizer'],
              kernel_constraint=args['kernel_constraint'],
              bias_constraint=args['bias_constraint'],
              input_shape=(28,28, 1)
              ))
        else:
          self.model.add(tf.keras.layers.Conv2D(
            filters=args['filters'],
            kernel_size=args['kernel_size'],
            strides=args['strides'],
            padding=args['padding'],
            data_format=args['data_format'],
            dilation_rate=args['dilation_rate'],
            groups=args['groups'],
            activation=args['activation'],
            use_bias=args['use_bias'],
            kernel_initializer=args['kernel_initializer'],
            bias_initializer=args['bias_initializer'],
            kernel_regularizer=args['kernel_regularizer'],
            bias_regularizer=args['bias_regularizer'],
            activity_regularizer=args['activity_regularizer'],
            kernel_constraint=args['kernel_constraint'],
            bias_constraint=args['bias_constraint']
            ))
      
      elif name=='dense':
        args = self.args_dense(args)
        self.model.add(tf.keras.layers.Dense(
            units=args['units'],
            activation=args['activation'],
            use_bias=args['use_bias'],
            kernel_initializer=args['kernel_initializer'],
            bias_initializer=args['bias_initializer'],
            kernel_regularizer=args['kernel_regularizer'],
            bias_regularizer=args['bias_regularizer'],
            activity_regularizer=args['activity_regularizer'],
            kernel_constraint=args['kernel_constraint'],
            bias_constraint=args['bias_constraint']
        ))

      elif name=='flatten':
        args = self.args_flatten(args)
        self.model.add(tf.keras.layers.Flatten(
            data_format=args['data_format']
        ))
      
      elif name=='maxpool2d':
        args = self.args_maxpool2d(args)
        self.model.add(tf.keras.layers.MaxPool2D(
            pool_size=args['pool_size'],
            strides=args['strides'],
            padding=args['padding'],
            data_format=args['data_format']
        ))
      
      elif name=='drop':
        args = self.args_drop(args)
        self.model.add(tf.keras.layers.Dropout(
            rate=args['rate'],
            noise_shape=args['noise_shape'],
            seed=args['seed']
        ))
      
      else:
        raise ValueError(f'No layer found with name: {name}')
      
      cnt += 1
    
    self.model.add(tf.keras.layers.Dense(10, activation='softmax'))
    print(self.model.summary())
    # return self.model

  def compile(self, learning_rate):
    with open("basicCNN_summary.txt", "w") as fp:
      with redirect_stdout(fp):
          self.model.summary()
    
    self.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
  
  def fit(self, batch_size=None, epochs=1, verbose='auto',callbacks=None, 
          validation_split=0.0, shuffle=True, validation_steps=None, 
          validation_batch_size=None, validation_freq=1, max_queue_size=10, 
          workers=-1, use_multiprocessing=True):
    history = self.model.fit(x=self.x_train, y=self.y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=verbose,
                   callbacks=callbacks,
                   validation_split=validation_split,
                   shuffle=shuffle,
                   validation_steps=validation_steps,
                   validation_batch_size=validation_batch_size,
                   validation_freq=validation_freq,
                   max_queue_size=max_queue_size,
                   workers=workers,
                   use_multiprocessing=use_multiprocessing)
    return history
  
  def save(self, name: str):
    self.model.save(name+'.keras')

  def visualize(self, seq: Union[int, None] = None):
    if seq==None:
      seq = self.random
    
    plt.imshow(self.mnist_images[seq])
    plt.show()
    print(self.mnist_labels[seq])

  def evaluate(self):
    loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=2)
    return "Trained model, accuracy: {:5.2f}%".format(100 * acc)
  
  def load_model(self, path: str):
    self.model = tf.keras.models.load_model(path+'.keras')

  def predict(self, x: np.array, workers=-1, use_multiprocessing=True):
    if x.shape != (28, 28):
      raise ValueError(f"Expected an array of shape (28, 28), but recieved an array of shape {x.shape}")
    
    max_val = np.max(x)
    if max_val > 1:
      x = x/255.0

    x = np.expand_dims(x, axis=0)
    x = tf.convert_to_tensor(x)
    
    value = self.model.predict(
        x=x,
        workers=workers,
        use_multiprocessing=use_multiprocessing
        )
    
    value = np.argmax(value)
    return value

