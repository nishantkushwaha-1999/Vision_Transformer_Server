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

# mn = mnist.MNIST()
# mn.load_model("basicCNN/basicCNN")

vit = VisionTrnasformer()
# vit.load_model("ViT/ViT_Mnist")

print(vit.hyperparams)