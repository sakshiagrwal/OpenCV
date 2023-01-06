import os
import cv2

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Specify the directory containing the photos
photos_dir = 'photos'

# Create a dictionary to store the face images for each person
face_images = {}

# Loop through all of the photos in the directory
for filename in os.listdir(photos_dir):
    # Load the photo
    photo = cv2.imread(os.path.join(photos_dir, filename))

    # Convert the photo to grayscale
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    # Detect faces in the photo
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Loop through all of the faces detected in the photo
    for (x, y, w, h) in faces:
        # Get the person's name from the filename
        name = filename.split('.')[0]

        # If this is the first time we've encountered this person, create a new folder for them
        if name not in face_images:
            os.makedirs(name)
            face_images[name] = []

        # Add the file path of the photo to the person's list of face images
        face_images[name].append(os.path.join(photos_dir, filename))

# Move the photos to the appropriate folders
for name in face_images:
    for photo_path in face_images[name]:
        # Get the filename of the photo
        filename = os.path.basename(photo_path)

        # Create the destination file path
        dest_path = os.path.join(name, filename)

        # Move the photo to the destination folder
        os.rename(photo_path, dest_path)

# Print the number of face images for each person
for name in face_images:
    print(f'{name}: {len(face_images[name])} face images')
