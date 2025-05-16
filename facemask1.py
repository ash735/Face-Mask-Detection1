import os
import numpy as np
import cv2
import datetime
import json
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Check if the model already exists. If yes, load it; otherwise, train it.
if os.path.exists('mymodel.h5'):
    print("Loading saved model...")
    model = load_model('mymodel.h5')
    
    # If training history was saved, load and print the best epoch details.
    if os.path.exists('training_history.json'):
        with open('training_history.json', 'r') as f:
            history_data = json.load(f)
        best_epoch = np.argmax(history_data['val_accuracy'])
        best_val_accuracy = history_data['val_accuracy'][best_epoch]
        print("Best Epoch from saved history:", best_epoch + 1)  # 1-indexed for readability
        print("Best Validation Accuracy from saved history:", best_val_accuracy)
    else:
        print("No training history found.")
else:
    print("Training model from scratch...")
    # Build the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Data generators for training and testing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(150, 150),
        batch_size=16,
        class_mode='binary'
    )
    
    test_set = test_datagen.flow_from_directory(
        'test',
        target_size=(150, 150),
        batch_size=16,
        class_mode='binary'
    )
    
    # Set up ModelCheckpoint to save the best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        'mymodel.h5', 
        monitor='val_accuracy', 
        verbose=1, 
        save_best_only=True, 
        mode='max'
    )
    
    # Train the model and capture the training history
    history = model.fit(
        training_set,
        epochs=10,
        validation_data=test_set,
        callbacks=[checkpoint]
    )
    
    # Save the training history to a JSON file for future use
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)
    
    # Determine which epoch had the best validation accuracy
    best_epoch = np.argmax(history.history['val_accuracy'])
    best_val_accuracy = history.history['val_accuracy'][best_epoch]
    
    print("Best Epoch:", best_epoch + 1)  # Converting to 1-indexed count
    print("Best Validation Accuracy:", best_val_accuracy)

# Live Mask Detection
# Use DirectShow backend to avoid MSMF issues
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        # Extract the face region and prepare it for prediction
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg', face_img)
        test_img = image.load_img('temp.jpg', target_size=(150, 150, 3))
        test_img = image.img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis=0)
        pred = model.predict(test_img)[0][0]

        # Threshold: >= 0.5 is "NO MASK", otherwise "MASK"
        if pred >= 0.5:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(frame, 'NO MASK', ((x+w)//2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame, 'MASK', ((x+w)//2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Display current date and time on the frame
        datet = str(datetime.datetime.now())
        cv2.putText(frame, datet, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Live Mask Detection', frame)
    
    # Break if the window is closed or 'q' is pressed
    if cv2.getWindowProperty('Live Mask Detection', cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()