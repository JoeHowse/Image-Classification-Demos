#!/usr/bin/env python


import glob

import cv2
import numpy
import PIL
import torchvision


def main():

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    THRESHOLD = 0.75

    LABELS = [
        '__background__', 'persona', 'bicicleta', 'coche', 'motocicleta', 'avión',
        'autobús', 'tren', 'camión', 'barco', 'semáforo', 'hidrante', 'N/A',
        'señal de stop', 'parquímetro', 'banco', 'ave', 'gato', 'perro', 'caballo',
        'oveja', 'vaca', 'elefante', 'oso', 'cebra', 'jirafa', 'N/A', 'mochila',
        'paraguas', 'N/A', 'N/A', 'bolso', 'corbata', 'maleta', 'frisbee', 'esquís',
        'snowboard', 'pelota deportiva', 'barrilete', 'bate de béisbol',
        'guante de béisbol', 'patineta', 'tabla hawaiana', 'raqueta de tenis',
        'botella', 'N/A', 'copa de vino', 'vaso', 'tenedor', 'cuchillo', 'cuchara',
        'tazón', 'plátano', 'manzana', 'sándwich', 'naranja', 'brócoli',
        'zanahoria', 'perro caliente', 'pizza', 'donut', 'pastel', 'silla', 'sofá',
        'planta en maceta', 'cama', 'N/A', 'mesa de comedor', 'N/A', 'N/A',
        'inodoro', 'N/A', 'tv', 'portátil', 'ratón', 'control remoto', 'teclado',
        'celular', 'microondas', 'horno', 'tostadora', 'fregadero', 'refrigerador',
        'N/A', 'libro', 'reloj', 'florero', 'tijeras', 'osito de peluche',
        'secador de pelo', 'cepillo de dientes'
    ]

    COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
              (0, 255, 255), (255, 255, 0), (255, 0, 255)]

    for input_filename in glob.iglob(r'resources/*.jpg'):

        image = cv2.imread(input_filename)
        output_filename = input_filename.replace(
            'resources/', 'coco_segmentation_output/')

        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])
        tensor = transform(PIL.Image.open(input_filename))

        # Detect instances in the image.
        predictions = model([tensor])[0]
        scores = list(predictions['scores'].detach().numpy())
        scores = [score for score in scores if score > THRESHOLD]
        masks = (predictions['masks'] > 0.5).squeeze().detach().cpu().numpy()
        labels = [LABELS[i] for i in list(predictions['labels'].numpy())]
        boxes = [[(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))]
                 for box in list(predictions['boxes'].detach().numpy())]

        # Visualize the results.
        for i, (score, mask, label, box) in enumerate(zip(
                scores, masks, labels, boxes)):

            if score < THRESHOLD:
                continue

            color = COLORS[i % len(COLORS)]

            # Draw the mask.
            colored_mask = numpy.zeros_like(image)
            colored_mask[mask==1] = color
            cv2.addWeighted(image, 1.0, colored_mask, 0.5, 0.0, image)

            # Draw the box.
            #cv2.rectangle(image, box[0], box[1], color, 2)

            # Draw the label and score.
            text = '%s (%.1f%%)' % (label, score * 100.0)
            cv2.putText(image, text, (box[0][0], box[0][1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imwrite(output_filename, image)


if __name__ == '__main__':
    main()
