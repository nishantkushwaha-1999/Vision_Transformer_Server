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
      self.file = file.get_bytes()
      im_df = pd.read_csv(BytesIO(self.file), header=None)
      im_df = np.array(im_df)

      if im_df.shape != (28, 28):
        raise ValueError(f"Expected file of shape(28, 28), but recieved with shape {im_df.shape}")
      
      image = Image.fromarray(im_df, '1')
      bs = BytesIO()
      image.save(bs, format='JPEG')

      return anvil.BlobMedia("image/jpeg", bs.getvalue(), name='input')
    except Error as e:
      return e
       
h = Handler()
print(h.evaluate('vit'))

@anvil.server.callable
def image(file):
  return h.convert_image(file)

# @anvil.server.callable
# def get_history(model: str, type: str):
#   path = f"{model}/history_{type}.json"
#   with open("basicCNN/history_val.json", 'r') as fp:
#       data = json.load(fp)
#   return data

# @anvil.server.callable
# def evaluate_bCNN():
#     result = mn.evaluate()
#     return result

# @anvil.server.callable
# def image(file):
#    try:
#       file = file.get_bytes()
#       file = pd.read_csv(BytesIO(file), header=None)
#       df = np.array(df)
      
#       val = mn.predict(df)
#       return val
    
#     except Error as e:
#       return e
   

# @anvil.server.callable
# def predict_basicCNN(file):
#     try:
#       file = file.get_bytes()
#       df = pd.read_csv(BytesIO(file), header=None)
#       df = np.array(df)
      
#       val = mn.predict(df)
#       return val
    
#     except Error as e:
#       return e

# @anvil.server.callable
# def predict_vit(file):
#     try:
#       file = file.get_bytes()
#       df = pd.read_csv(BytesIO(file), header=None)
#       df = np.array(df)
#       df = np.expand_dims(df, axis=0)
#       df = vit.preprocess_data(df)[0]
#       print(df.shape)
      
#       val = vit.predict(df)
#       return val
    
#     except Error as e:
#       return e

anvil.server.connect("server_O5OPT6KC6T6WXPJUMOM7IJUG-VJ5KJZ4SWIN2DSBR")
anvil.server.wait_forever()