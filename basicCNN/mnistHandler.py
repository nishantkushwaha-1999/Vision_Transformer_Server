import mnist
import json

mn = mnist.MNIST()

lyrs = [('convo2d', {'filters':15, 'kernel_size':(2, 2), 'strides':(1, 1), 'activation':'relu'}),
        ('convo2d', {'filters':15, 'kernel_size':(2, 2), 'strides':(1, 1), 'activation':'relu'}),
        ('convo2d', {'filters':15, 'kernel_size':(2, 2), 'strides':(1, 1), 'activation':'relu'}),
        ('maxpool2d', {'strides': (2, 2)}),
        ('drop', {'rate': 0.3}), 
        
        ('convo2d', {'filters':25, 'kernel_size':(3, 3), 'strides':(1, 1), 'activation':'relu'}),
        ('convo2d', {'filters':25, 'kernel_size':(3, 3), 'strides':(1, 1), 'activation':'relu'}),
        ('convo2d', {'filters':25, 'kernel_size':(3, 3), 'strides':(1, 1), 'activation':'relu'}),
        ('maxpool2d', {'strides': (2, 2)}),
        ('drop', {'rate': 0.3}),
        
        ('flatten', {}),
        ('dense', {'units':512, 'activation':'relu'}),
        ('dense', {'units':256, 'activation':'relu'}),
        ('dense', {'units':128, 'activation':'relu'})]

mn.initializeConvNetwork(layers=lyrs)
mn.compile(0.001)
history_val = mn.fit(batch_size=128, validation_split=0.2, epochs=20)

with open("history_val.json", "w") as fp:
	json.dump(history_val.history, fp)
	# fp.writelines(str(history_val.history))

mn.initializeConvNetwork(layers=lyrs)
mn.compile(0.001)
history_all = mn.fit(batch_size=128, epochs=20)

with open("history_all.json", "w") as fp:
	json.dump(history_all.history, fp)
	# fp.writelines(str(history_all.history))

mn.save('basicCNN')

mn.evaluate()
