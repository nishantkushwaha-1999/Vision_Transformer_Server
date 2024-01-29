from basicCNN import mnist
import json

# import anvil.server

# mn = mnist.MNIST()
# mn.load_model("basicCNN/basicCNN")

# @anvil.server.callable
# def evaluate():
#     result = mn.evaluate()
#     return result

def history_all():
    with open("basicCNN/history_val.txt", 'r') as fp:
        data = fp.readlines()
    data = json.loads(data[0])
    print(data)

history_all()

# anvil.server.connect("server_4FTKWSNOYMLKR7NPWYBUZPTJ-3G6ST77Q4OVSWPLF")
# anvil.server.wait_forever()
