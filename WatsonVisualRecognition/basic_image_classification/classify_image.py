#!/usr/bin/env python


from __future__ import print_function
import json
import os
from ibm_watson import VisualRecognitionV3, ApiException


def main():

    try:

        service = VisualRecognitionV3(
            '2018-03-19',
            url=os.environ['WATSON_VISUAL_RECOGNITION_URL'],
            iam_apikey=os.environ['WATSON_VISUAL_RECOGNITION_KEY'])

        image_path = os.path.abspath('resources/image.jpg')
        with open(image_path, 'rb') as image_file:
            classify_result = service.classify(
                images_file=image_file,
                threshold='0.1',
                classifier_ids=['default']).get_result()
            print(json.dumps(classify_result, indent=2))

    except ApiException as ex:
        print(ex)


if __name__ == '__main__':
    main()
