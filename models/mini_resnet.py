import tensorflow as tf

class CNNBlock(tf.keras.Model):
    def __init__(self, kernels, kernel_size=2):
        super(CNNBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(kernels, kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(kernels, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.concatenate = tf.keras.layers.Concatenate()
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.concatenate([x, inputs])
        return self.act(x)

class DNNBlock(tf.keras.Model):
    def __init__(self, units):
        super(DNNBlock, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.fc2 = tf.keras.layers.Dense(units)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.concatenate = tf.keras.layers.Concatenate()
    
    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.concatenate([x, inputs])
        return self.act(x)

class ResNet(tf.keras.Model):
    def __init__(self, num_classes, kernels=128, kernel_size=3):
        super(ResNet, self).__init__()
        self.cnn_block = CNNBlock(kernels, kernel_size)
        self.flatten = tf.keras.layers.Flatten()
        self.dnn_block1 = DNNBlock(128)
        self.dnn_block2 = DNNBlock(128)
        self.out = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs):
        x = self.cnn_block(inputs)
        x = self.flatten(x)
        x = self.dnn_block1(x)
        x = self.dnn_block2(x)
        return self.out(x)