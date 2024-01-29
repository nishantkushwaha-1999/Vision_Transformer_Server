from basicCNN import mnist
import json

import anvil.server

mn = mnist.MNIST()
mn.load_model("basicCNN/basicCNN")

@anvil.server.callable
def evaluate():
    result = mn.evaluate()
    return result

def history_all():
    with open("basicCNN/history_val.json", 'r') as fp:
        data = json.load(fp)
    return data

anvil.server.connect("server_O5OPT6KC6T6WXPJUMOM7IJUG-VJ5KJZ4SWIN2DSBR")
anvil.server.wait_forever()
