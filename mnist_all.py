from basicCNN import mnist
import pandas as pd
import plotly.express as px
import pandas as pd
import numpy as np
import json
from io import BytesIO

import anvil.server

mn = mnist.MNIST()
mn.load_model("basicCNN/basicCNN")

@anvil.server.callable
def history_all_data():
    with open("basicCNN/history_val.json", 'r') as fp:
        data = json.load(fp)
    return data

@anvil.server.callable
def history_all_data():
    with open("basicCNN/history_all.json", 'r') as fp:
        data = json.load(fp)
    return data

@anvil.server.callable
def evaluate_bCNN():
    result = mn.evaluate()
    return result

@anvil.server.callable
def predict(file):
    # with open(strfile, 'r') as fp:
    file = file.get_bytes()
    df = pd.read_csv(BytesIO(file))
    return np.array(df)

anvil.server.connect("server_O5OPT6KC6T6WXPJUMOM7IJUG-VJ5KJZ4SWIN2DSBR")
anvil.server.wait_forever()
