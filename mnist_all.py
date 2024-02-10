from csv import Error
from types import NoneType
from basicCNN import mnist
import pandas as pd
import plotly.express as px
import pandas as pd
import numpy as np
import json
from io import BytesIO
from ViT.model import VisionTrnasformer

import anvil.server

mn = mnist.MNIST()
mn.load_model("basicCNN/basicCNN")

vit = VisionTrnasformer()
vit.load_model("ViT/ViT_Mnist")

@anvil.server.callable
def get_history(model: str, type: str):
  path = f"{model}/history_{type}.json"
  with open("basicCNN/history_val.json", 'r') as fp:
      data = json.load(fp)
  return data

@anvil.server.callable
def evaluate_bCNN():
    result = mn.evaluate()
    return result

@anvil.server.callable
def predict_basicCNN(file):
    try:
      file = file.get_bytes()
      df = pd.read_csv(BytesIO(file), header=None)
      df = np.array(df)
      
      val = mn.predict(df)
      return val
    
    except Error as e:
      return e

@anvil.server.callable
def predict_vit(file):
    try:
      file = file.get_bytes()
      df = pd.read_csv(BytesIO(file), header=None)
      df = np.array(df)
      df = np.expand_dims(df, axis=0)
      df = vit.preprocess_data(df)[0]
      print(df.shape)
      
      val = vit.predict(df)
      return val
    
    except Error as e:
      return e

anvil.server.connect("server_O5OPT6KC6T6WXPJUMOM7IJUG-VJ5KJZ4SWIN2DSBR")
anvil.server.wait_forever()