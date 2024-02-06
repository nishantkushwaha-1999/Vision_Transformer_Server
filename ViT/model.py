import numpy as np
import tensorflow as tf

class ClassToken(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls
    
class VisionTrnasformer():
    def __init__(self):
        pass

    def load_mnist(self, normalize=True):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

        if normalize:
            self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        
        self.x_train = tf.convert_to_tensor(self.x_train)
        self.y_train = tf.convert_to_tensor(self.y_train)
        self.x_test = tf.convert_to_tensor(self.x_test)
        self.y_test = tf.convert_to_tensor(self.y_test)
        return(self.x_train, self.y_train, self.x_test, self.y_test)

    def preprocess_data(self, data, patch_rows, patch_columns):
        if type(data)!=np.ndarray:
            data = data.numpy()
        flatten_images = np.zeros((data.shape[0],patch_rows*patch_columns, 
                                   int((data.shape[1]*data.shape[1])/(patch_rows*patch_columns))))
        # print(flatten_images.shape)
        helper = int(data.shape[1]/patch_rows)
        for i in range(data.shape[0]):
            ind = 0
            for row in range(patch_rows):
                for col in range(patch_columns):
                    # print(data[i,(row*helper):((row+1)*helper),(col*helper):((col+1)*helper)].ravel())
                    flatten_images[i,ind,:] = data[i,
                                                   (row*helper):((row+1)*helper),
                                                   (col*helper):((col+1)*helper)].ravel()
                    ind += 1
        return tf.convert_to_tensor(flatten_images)
    
    def initialize(self, patch_rows, patch_columns, embedding_dim, img_shape_x, img_shape_y,
                   n_encoders, n_heads, key_dim, value_dim, dropout_rate, num_classes):
        self.patch_rows = patch_rows
        self.patch_columns = patch_columns
        self.embedding_dim = embedding_dim
        self.img_shape_x = img_shape_x
        self.img_shape_y = img_shape_y
        self.block_size = int((self.img_shape_y*self.img_shape_x)/(self.patch_rows*self.patch_columns))

        inputlayer = tf.keras.layers.Input((self.patch_rows*self.patch_columns, self.block_size))
        n_patches = tf.keras.layers.Input(self.patch_rows*self.patch_columns)

        embeddings = tf.keras.layers.Embedding(input_dim=self.patch_rows*self.patch_columns, 
                                                output_dim=self.embedding_dim)(n_patches)
        
        projection = tf.keras.layers.Dense(self.embedding_dim)(inputlayer)

        vect = projection + embeddings
        
        token = ClassToken()(vect)
        vect = tf.keras.layers.Concatenate(axis=1)([token, vect])

        for _ in range(n_encoders):
            norm_lyr = tf.keras.layers.LayerNormalization()(vect)
            mha = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=key_dim, value_dim=value_dim)(norm_lyr, norm_lyr, norm_lyr)
            skip_conn1 = tf.keras.layers.add([vect, mha])
            norm_lyr = tf.keras.layers.LayerNormalization()(skip_conn1)
            mlp = tf.keras.layers.Dense(self.embedding_dim, activation='gelu')(norm_lyr)
            mlp = tf.keras.layers.Dropout(dropout_rate)(mlp) 
            mlp = tf.keras.layers.Dense(int(self.embedding_dim/2))(mlp)
            mlp = tf.keras.layers.Dropout(dropout_rate)(mlp)
            # mlp = tf.keras.layers.Dense(int(self.embedding_dim/4))(mlp)
            # mlp = tf.keras.layers.Dropout(dropout_rate)(mlp)
            mlp = tf.keras.layers.Dense(self.embedding_dim)(mlp)
            mlp = tf.keras.layers.Dropout(dropout_rate)(mlp)
            vect = tf.keras.layers.Add()([mlp, skip_conn1])
        
        norm_lyr = tf.keras.layers.LayerNormalization()(vect)
        class_token = norm_lyr[:,0,:]
        clas = tf.keras.layers.Dense(num_classes,activation='softmax')(class_token)
        self.model = tf.keras.models.Model([inputlayer, n_patches], clas)
        return self.model
    
    def fit(self, x, y, batch_size=None, epochs=1, verbose='auto',callbacks=None, 
          validation_split=0.0, shuffle=True, validation_steps=None, 
          validation_batch_size=None, validation_freq=1, max_queue_size=10, 
          workers=-1, use_multiprocessing=True):
        
        pos_feed = np.array([list(range(self.patch_rows*self.patch_columns))]*x.shape[0])
        history = self.model.fit(x=[x, pos_feed], y=y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    callbacks=callbacks,
                    validation_split=validation_split,
                    shuffle=shuffle,
                    validation_steps=validation_steps,
                    validation_batch_size=validation_batch_size,
                    validation_freq=validation_freq,
                    max_queue_size=max_queue_size,
                    workers=workers,
                    use_multiprocessing=use_multiprocessing)
        return history
    
    def compile(self, learning_rate):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            )
    
    def save(self, name: str):
        self.model.save(name+'.keras')

    def predict(self, x: np.array, workers=-1, use_multiprocessing=True):
        if x.shape != (28, 28):
            raise ValueError(f"Expected an array of shape (28, 28), but recieved an array of shape {x.shape}")
    
        x = np.expand_dims(x, axis=0)
        x = tf.convert_to_tensor(x)
        
        pos_feed = np.array([list(range(self.patch_rows*self.patch_columns))]*x.shape[0])
        value = self.model.predict(
            x=[x, pos_feed],
            workers=workers,
            use_multiprocessing=use_multiprocessing
            )
        
        value = np.argmax(value)
        return value
    
    def evaluate(self, x, y):
        pos_feed = np.array([list(range(self.patch_rows*self.patch_columns))]*x.shape[0])
        loss, acc = self.model.evaluate(x=[x, pos_feed], y=y, verbose=2)
        return "Trained model, accuracy: {:5.2f}%".format(100 * acc)
