import json
from model import VisionTrnasformer
from contextlib import redirect_stdout

ViT = VisionTrnasformer()
x_train, y_train, x_test, y_test = ViT.load_mnist()

patch_rows = 4
patch_columns = 4
embedding_dim = 128
key_dim = 128
value_dim = 128
n_encoders = 4
n_heads = 6
lr = 0.001
dropout_rate = 0.3

x_train = ViT.preprocess_data(x_train, patch_rows=patch_rows, patch_columns=patch_columns)
x_test = ViT.preprocess_data(x_test, patch_rows=patch_rows, patch_columns=patch_columns)

model = ViT.initialize(patch_rows=patch_rows, patch_columns=patch_columns, 
                       embedding_dim=embedding_dim, img_shape_x=28, img_shape_y=28,
                   n_encoders=n_encoders, n_heads=n_heads, key_dim=key_dim, 
                       value_dim=value_dim, dropout_rate=dropout_rate, num_classes=10)

with open("ViT_summary.txt", "w") as fp:
    with redirect_stdout(fp):
        model.summary()

ViT.compile(lr)
history_val = ViT.fit(x=x_train, y=y_train, batch_size=1024, validation_split=0.2, epochs=300)

with open("history_val.json", "w") as fp:
	json.dump(history_val.history, fp)