#!/usr/bin/env python


import glob

import cv2


def main():

    model = cv2.dnn.readNetFromCaffe(
        'objects_data/MobileNetSSD_deploy.prototxt',
        'objects_data/MobileNetSSD_deploy.caffemodel')
    blob_height = 300
    color_scale = 1.0/127.5
    average_color = (127.5, 127.5, 127.5)
    confidence_threshold = 0.5
    labels = ['avión', 'bicicleta', 'ave', 'barco', 'botella', 'autobús',
              'coche', 'gato', 'silla', 'vaca', 'mesa de comedor', 'perro',
              'caballo', 'moto', 'persona', 'planta en maceta', 'oveja',
              'sofá', 'tren', 'tv o pantalla']

    for input_filename in glob.iglob(r'resources/*.jpg'):

        image = cv2.imread(input_filename)
        output_filename = input_filename.replace('resources/', 'objects_output/')

        h, w = image.shape[:2]
        aspect_ratio = w/h

        # Detect objects in the image.

        blob_width = int(blob_height * aspect_ratio)
        blob_size = (blob_width, blob_height)

        blob = cv2.dnn.blobFromImage(
            image, scalefactor=color_scale, size=blob_size,
            mean=average_color)

        model.setInput(blob)
        results = model.forward()

        # Iterate over the detected objects.
        for object in results[0, 0]:
            confidence = object[2]
            if confidence > confidence_threshold:

                # Get the object's coordinates.
                x0, y0, x1, y1 = (object[3:7] * [w, h, w, h]).astype(int)

                # Get the classification result.
                id = int(object[1])
                label = labels[id - 1]

                # Draw a blue rectangle around the object.
                cv2.rectangle(image, (x0, y0), (x1, y1),
                              (255, 0, 0), 2)

                # Draw the classification result and confidence.
                text = '%s (%.1f%%)' % (label, confidence * 100.0)
                cv2.putText(image, text, (x0, y0 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        cv2.imwrite(output_filename, image)


if __name__ == '__main__':
    main()
