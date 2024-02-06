import tensorflow as tf
from tensorflow import random
import os
import sys

def build_model(op_type):
    if (op_type == "conv2d"):
            input_shape = (5, 5, 1)
            x_train = tf.random.normal(input_shape, mean = 0.0, stddev = 1.0)

            in_layer = tf.keras.layers.Input(shape=input_shape)
            conv2d_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3)) (in_layer)
##            print (" Shape is " + str(conv2d_layer.output_shape))
            model = tf.keras.Model(in_layer, conv2d_layer)
            model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
            for layer in model.layers:
                print(layer.output_shape)
            model.summary()
            #model.fit(x_train, y_train, epochs=5, batch_size=32)

            export_dir = os.path.join(os.getcwd(), op_type)
            os.makedirs(export_dir, exist_ok=True)
            tf.saved_model.save(model, export_dir)
            converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
            tflite_model = converter.convert()
            tflite_file = os.path.join(export_dir, 'conv2d_model.tflite')
            with open(tflite_file, 'wb') as f:
                f.write(tflite_model)

            print(f"model saved at {export_dir}")

    if (op_type == "relu6"):
            fit_model = False

            input_shape = (5, 5, 1)
            x_train = tf.random.normal(input_shape, mean = 0.0, stddev = 1.0)

            in_layer = tf.keras.layers.Input(shape=input_shape)
            #conv2d_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3)) (in_layer)
            relu_layer = tf.keras.layers.ReLU(6.0) (in_layer)
            model = tf.keras.Model(in_layer, relu_layer)

            model.compile(optimizer='adam',  loss='categorical_crossentropy', metrics=['accuracy'])

            x_shape = (128,1)
            x_train  = tf.random.normal(x_shape, mean = 0.0, stddev = 1.0)

            y_shape = (1,)
            y_train  = tf.random.uniform(y_shape, minval=0, maxval=1,dtype=tf.int32)
            model.build(x_shape)

            model.summary()

            if fit_model:
                model.fit(x_train, y_train, epochs=2)

            export_dir = os.path.join(os.getcwd(), op_type)
            os.makedirs(export_dir, exist_ok=True)
            tf.saved_model.save(model, export_dir)
            converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
            tflite_model = converter.convert()
            tflite_file = os.path.join(export_dir, 'relu6_model.tflite')
            with open(tflite_file, 'wb') as f:
                f.write(tflite_model)

            print(f"model saved at {export_dir}")
    if (op_type == "relu"):
            fit_model = False

            input_shape = (5, 5, 1)
            x_train = tf.random.normal(input_shape, mean = 0.0, stddev = 1.0)

            in_layer = tf.keras.layers.Input(shape=input_shape)
            #conv2d_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3)) (in_layer)
            relu_layer = tf.keras.layers.ReLU() (in_layer)
            model = tf.keras.Model(in_layer, relu_layer)

            model.compile(optimizer='adam',  loss='categorical_crossentropy', metrics=['accuracy'])

            x_shape = (128,1)
            x_train  = tf.random.normal(x_shape, mean = 0.0, stddev = 1.0)

            y_shape = (1,)
            y_train  = tf.random.uniform(y_shape, minval=0, maxval=1,dtype=tf.int32)
            model.build(x_shape)

            model.summary()

            if fit_model:
                model.fit(x_train, y_train, epochs=2)

            export_dir = os.path.join(os.getcwd(), op_type)
            os.makedirs(export_dir, exist_ok=True)
            tf.saved_model.save(model, export_dir)
            converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
            tflite_model = converter.convert()
            tflite_file = os.path.join(export_dir, 'relu_model.tflite')
            with open(tflite_file, 'wb') as f:
                f.write(tflite_model)

            print(f"model saved at {export_dir}")

    if (op_type == "logistic"):
            input_shape = (5, 5, 1)
            x_train = tf.random.normal(input_shape, mean = 0.0, stddev = 1.0)

            in_layer = tf.keras.layers.Input(shape=input_shape)
            logistic_layer = tf.keras.activations.sigmoid(in_layer)
            model = tf.keras.Model(in_layer, logistic_layer)
            model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
            for layer in model.layers:
                print(layer.output_shape)
            model.summary()

            export_dir = os.path.join(os.getcwd(), op_type)
            os.makedirs(export_dir, exist_ok=True)
            tf.saved_model.save(model, export_dir)
            converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
            tflite_model = converter.convert()
            tflite_file = os.path.join(export_dir, 'logistic_model.tflite')
            with open(tflite_file, 'wb') as f:
                f.write(tflite_model)

            print(f"model saved at {export_dir}")

    if (op_type == "mul"):
            input1 = tf.keras.layers.Input(shape=(32,))
            input2 = tf.keras.layers.Input(shape=(32,))
            multiplied = tf.keras.layers.multiply([input1, input2])
            model = tf.keras.models.Model(inputs=[input1, input2], outputs=multiplied)

            model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
            for layer in model.layers:
                print(layer.output_shape)
            model.summary()

            export_dir = os.path.join(os.getcwd(), op_type)
            os.makedirs(export_dir, exist_ok=True)
            tf.saved_model.save(model, export_dir)
            converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
            tflite_model = converter.convert()
            tflite_file = os.path.join(export_dir, 'mul_model.tflite')
            with open(tflite_file, 'wb') as f:
                f.write(tflite_model)

            print(f"model saved at {export_dir}")

def main():
    print(f"building {sys.argv[1]}")
    build_model(sys.argv[1])

main()
