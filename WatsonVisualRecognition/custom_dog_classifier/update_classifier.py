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
        found_right_classifier = False
        for classifier in list_classifiers_result['classifiers']:
            if classifier['name'] == classifier_name:
                found_right_classifier = True
                break
        if not found_right_classifier:
            print('Could not find "%s" classifier.' % classifier_name)
            return
        if classifier['status'] != 'ready':
            print('Found "%s" classifier but it is not ready to use:' % classifier_name)
            print(json.dumps(classifier, indent=2))
            return
        print('Found "%s" classifier, which is ready to use:' % classifier_name)
        print(json.dumps(classifier, indent=2))

        dalmatian_path = os.path.abspath('resources/training/dalmatian.zip')
        more_cats_path = os.path.abspath('resources/training/more-cats.zip')
        with open(dalmatian_path, 'rb') as dalmatian, \
             open(more_cats_path, 'rb') as more_cats:
            update_classifier_result = service.update_classifier(
                classifier['classifier_id'],
                dalmatian_positive_examples=dalmatian,
                negative_examples=more_cats).get_result()
            print('Updated "%s" classifier:' % classifier_name)
            print(json.dumps(update_classifier_result, indent=2))

    except WatsonApiException as ex:
        print(ex)


if __name__ == '__main__':
    main()
