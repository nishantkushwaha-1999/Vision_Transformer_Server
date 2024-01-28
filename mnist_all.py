import mnist

import anvil.server

mn = mnist.MNIST()
mn.load_model("basicCNN")

@anvil.server.callable
def evaluate():
    result = mn.evaluate()
    return result

anvil.server.connect("server_4FTKWSNOYMLKR7NPWYBUZPTJ-3G6ST77Q4OVSWPLF")
anvil.server.wait_forever()
