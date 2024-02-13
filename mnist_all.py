from csv import Error
from basicCNN import mnist
import pandas as pd
import plotly.express as px
import pandas as pd
import numpy as np
import json
from io import BytesIO
from ViT.model import VisionTrnasformer
from PIL import Image

import anvil.server

class Handler():
  def __init__(self):
    self.mn = mnist.MNIST()
    self.mn.load_model("basicCNN/basicCNN")

    self.vit = VisionTrnasformer()
    self.vit.load_model("ViT/ViT_Mnist")
    
  def get_history(self, model: str, type: str):
    path = f"{model}/history_{type}.json"
    with open("basicCNN/history_val.json", 'r') as fp:
        data = json.load(fp)
    return data
  
  def evaluate(self, model: str):
    if model=='basicCNN':
      result = self.mn.evaluate()
    elif model=='vit':
        result = self.vit.evaluate()
    else:
       raise ValueError(f"{model} not found.")
    return result
  
  def convert_image(self, file):
    try:
      file = file.get_bytes()
      im_df = pd.read_csv(BytesIO(file), header=None)
      im_df = np.array(im_df)

      if im_df.shape != (28, 28):
        raise ValueError(f"Expected file of shape(28, 28), but recieved with shape {im_df.shape}")
      
      max_val = np.max(im_df)
      if max_val < 1:
        im_df = im_df*255.0
      
      self.im_df = im_df.copy()
      
      image = Image.fromarray(self.im_df)
      image = image.convert("L")
      bs = BytesIO()
      image.save(bs, format='JPEG')

      return anvil.BlobMedia("image/jpeg", bs.getvalue(), name='input')
    except Error as e:
      return e
  
  def predict(self, model: str):
    try:
      if model=='basicCNN':
        val = self.mn.predict(self.im_df)
      elif model=='vit':
        df = np.expand_dims(self.im_df, axis=0)
        df = self.vit.preprocess_data(df)[0]
        val = self.vit.predict(df)
      else:
        raise ValueError(f"No model found with name {model}")
      return val
    except Error as e:
      return e


handler = Handler()

@anvil.server.callable
def get_history(model, type):
  return handler.get_history(model, type)

@anvil.server.callable
def evaluate(model):
  return handler.evaluate(model)

@anvil.server.callable
def image(file):
  return handler.convert_image(file)

@anvil.server.callable
def predict(model):
  return handler.predict(model)

anvil.server.connect("server_O5OPT6KC6T6WXPJUMOM7IJUG-VJ5KJZ4SWIN2DSBR")
anvil.server.wait_forever()