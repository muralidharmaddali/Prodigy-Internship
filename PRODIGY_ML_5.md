# PRODIGY_ML_5

import tensorflow as tf from tensorflow import keras from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory( 'path_to_dataset', target_size=(64, 64), batch_size=32, class_mode='categorical', subset='training' )

validation_generator = train_datagen.flow_from_directory( 'path_to_dataset', target_size=(64, 64), batch_size=32, class_mode='categorical', subset='validation' )

model = keras.Sequential([ Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)), MaxPooling2D((2, 2)), Conv2D(64, (3, 3), activation='relu'), MaxPooling2D((2, 2)), Flatten(), Dense(128, activation='relu'), Dense(num_classes, activation='softmax') # Replace num_classes with the number of gesture classes ])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit( train_generator, validation_data=validation_generator, epochs=10 # Adjust as needed )

test_generator = train_datagen.flow_from_directory( 'path_to_test_dataset', target_size=(64, 64), batch_size=32, class_mode='categorical' )

test_loss, test_acc = model.evaluate(test_generator) print(f'Test accuracy: {test_acc}')
