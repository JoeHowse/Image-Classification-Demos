#!/usr/bin/env python


from __future__ import print_function
import json
import os
from watson_developer_cloud import VisualRecognitionV3, WatsonApiException


def main():

    classifier_name = 'dogs'

    try:

        service = VisualRecognitionV3(
            '2018-03-19',
            url=os.environ['WATSON_VISUAL_RECOGNITION_URL'],
            iam_apikey=os.environ['WATSON_VISUAL_RECOGNITION_KEY'])

        list_classifiers_result = service.list_classifiers().get_result()
        for classifier in list_classifiers_result['classifiers']:
            if classifier['name'] == classifier_name:
                print('Found pre-existing "%s" classifier:' % classifier_name)
                print(json.dumps(classifier, indent=2))
                return

        beagle_path = os.path.abspath('resources/training/beagle.zip')
        husky_path = os.path.abspath('resources/training/husky.zip')
        golden_retriever_path = os.path.abspath(
            'resources/training/golden-retriever.zip')
        cats_path = os.path.abspath('resources/training/cats.zip')
        with open(beagle_path, 'rb') as beagle, \
                open(husky_path, 'rb') as husky, \
                open(golden_retriever_path, 'rb') as golden_retriever, \
                open(cats_path, 'rb') as cats:
            create_classifier_result = service.create_classifier(
                classifier_name,
                beagle_positive_examples=beagle,
                husky_positive_examples=husky,
                goldenretriever_positive_examples=golden_retriever,
                negative_examples=cats).get_result()
            print('Created new "%s" classifier:' % classifier_name)
            print(json.dumps(create_classifier_result, indent=2))

    except WatsonApiException as ex:
        print(ex)


if __name__ == '__main__':
    main()
