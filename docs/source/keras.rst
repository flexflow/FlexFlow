:tocdepth: 1
****************
Keras Interface
****************

FlexFlow provides a drop-in replacement for TensorFlow Keras. Running an existing Keras program on the FlexFlow backend only requires a few lines of changes to the program. The detailed instructions are as follows:

1. Replace the Keras header files
=================================

Redirect the program to import Keras functions from FlexFlow by using the following import header lines::

    from flexflow.keras.models import Model, Sequential
    from flexflow.keras.layers import Input, Dense, Conv2D, ...
    from flexflow.keras.callbacks import Callback, ...

2. Modify the main Keras program
================================

FlexFlow requires a Keras program to wrap its model construction in a Python function called ``top_level_task()``. This allows FlexFlow to automatically parallelize DNN training across all GPUs on all compute nodes. For example, the following code snippet shows parallelizing AlexNet training in FlexFlow:: 

    def top_level_task():
      model = Sequential()
      model.add(Conv2D(filters=64, input_shape=(3,229,229), kernel_size=(11,11), strides=(4,4), padding=(2,2), activation="relu"))
      model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
      model.add(Conv2D(filters=192, kernel_size=(5,5), strides=(1,1), padding=(2,2), activation="relu"))
      ## More lines for model construction
      model.add(Activation("softmax"))
      ## Model compilation
      model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
      ## Model training
      (x_train, y_train) = cifar10.load_data()
      model.fit(x_train, y_train, epochs=30)

    if __name__ == "__main__":
      top_level_task()

More FlexFlow Keras examples are available on `GitHub <https://github.com/flexflow/FlexFlow/tree/master/examples/python/keras>`_.
