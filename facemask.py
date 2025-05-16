import os
import numpy as np
import cv2
import datetime
import json
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# File paths for the model and training history
MODEL_PATH = "mymodel.keras"
HISTORY_PATH = "training_history.json"

# Check if a trained model already exists; if yes, load it (along with training history if available)
if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = load_model(MODEL_PATH)
    
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as f:
            history_data = json.load(f)
        best_epoch = np.argmax(history_data['val_accuracy'])
        best_val_accuracy = history_data['val_accuracy'][best_epoch]
        print("Best Epoch from saved history:", best_epoch + 1)
        print("Best Validation Accuracy from saved history:", best_val_accuracy)
    else:
        print("No training history found.")
else:
    print("Training model from scratch using MobileNetV2 backbone...")
    
    # Define the dataset directories for training and testing
    train_dir = 'train'
    test_dir = 'test'
    
    # Data augmentation and preprocessing for the training and test sets
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    # Create data generators using flow_from_directory (expects subfolders for each class)
    training_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    test_set = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    # Load the MobileNetV2 base model with pre-trained ImageNet weights and exclude the top layers
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(224, 224, 3)))
    
    # Construct the head of the model that will be placed on top of the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    
    # Place the head FC model on top of the base model
    model = Model(inputs=baseModel.input, outputs=headModel)
    
    # Freeze the layers of the base model so they are not updated during the initial training
    for layer in baseModel.layers:
        layer.trainable = False
    
    # Initialize hyperparameters and compile the model
    INIT_LR = 1e-4
    EPOCHS = 10  # Adjust the number of epochs if desired
    opt = Adam(learning_rate=INIT_LR)  # Removed decay parameter since it's ignored
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    # Set up a ModelCheckpoint callback to save the best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        MODEL_PATH, 
        monitor='val_accuracy', 
        verbose=1, 
        save_best_only=True, 
        mode='max'
    )
    
    # Train the model
    H = model.fit(
        training_set,
        steps_per_epoch=len(training_set),
        validation_data=test_set,
        validation_steps=len(test_set),
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )
    
    # Save the training history to a JSON file for future reference
    with open(HISTORY_PATH, 'w') as f:
        json.dump(H.history, f)
    
    # Determine and print the best epoch based on validation accuracy
    best_epoch = np.argmax(H.history['val_accuracy'])
    best_val_accuracy = H.history['val_accuracy'][best_epoch]
    print("Best Epoch:", best_epoch + 1)
    print("Best Validation Accuracy:", best_val_accuracy)
    
    # Optionally, plot the training loss and accuracy over the epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")
    plt.show()

# ----------------- Live Mask Detection -----------------
print("Starting live mask detection...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame using the Haar cascade classifier
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        # Extract the face ROI, resize to 224x224, and preprocess for prediction
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = img_to_array(face_img)
        face_img = preprocess_input(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        
        # Predict mask/no mask using the trained model
        pred = model.predict(face_img)[0]
        label = "Mask" if np.argmax(pred) == 0 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
    # Display current date and time on the frame
    datet = str(datetime.datetime.now())
    cv2.putText(frame, datet, (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Live Mask Detection", frame)
    
    # Exit if the window is closed or if 'q' is pressed
    if cv2.getWindowProperty("Live Mask Detection", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
