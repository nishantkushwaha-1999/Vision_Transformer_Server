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

anvil.server.connect("server_4FTKWSNOYMLKR7NPWYBUZPTJ-3G6ST77Q4OVSWPLF")
anvil.server.wait_forever()
