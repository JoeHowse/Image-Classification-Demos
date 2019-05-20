#!/usr/bin/env python


from __future__ import print_function
import json
import os
from ibm_watson import VisualRecognitionV3, ApiException


def get_metrics(service, classifier_id, images_paths, correct_label):

    num_images = 0
    num_hits = 0
    num_misses = 0
    total_hit_score = 0.0
    total_miss_score = 0.0

    for images_path in images_paths:
        images_absolute_path = os.path.abspath(images_path)
        with open(images_absolute_path, 'rb') as images_file:
            classify_result = service.classify(
                images_file=images_file,
                threshold='0.0',
                classifier_ids=[classifier_id]).get_result()
            print('Classification result:')
            print(json.dumps(classify_result, indent=2))
            for image in classify_result['images']:
                best_label = 'none'
                best_score = 0.0
                for classification in image['classifiers'][0]['classes']:
                    score = classification['score']
                    if score > best_score:
                        best_score = score
                        if score > 0.6:
                            best_label = classification['class']
                num_images += 1
                if best_label == correct_label:
                    num_hits += 1
                    total_hit_score += best_score
                else:
                    num_misses += 1
                    total_miss_score += best_score

    hit_rate = num_hits / float(num_images)
    if num_hits > 0:
        mean_hit_score = total_hit_score / num_hits
    else:
        mean_hit_score = float('nan')
    if num_misses > 0:
        mean_miss_score = total_miss_score / num_misses
    else:
        mean_miss_score = float('nan')
    return (hit_rate, mean_hit_score, mean_miss_score)


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

        classifier_id = classifier['classifier_id']

        metrics_for_label = {}
        metrics_for_label['beagle'] = get_metrics(
            service, classifier_id, ['resources/test/beagle_0.zip'], 'beagle')
        metrics_for_label['dalmatian'] = get_metrics(
            service, classifier_id, ['resources/test/dalmatian_0.zip'], 'dalmatian')
        metrics_for_label['golden retriever'] = get_metrics(
            service, classifier_id, ['resources/test/golden-retriever_0.zip'],
            'goldenretriever')
        metrics_for_label['husky'] = get_metrics(
            service, classifier_id, ['resources/test/husky_0.zip'], 'husky')
        metrics_for_label['none (max score <= 0.6)'] = get_metrics(
            service, classifier_id, ['resources/test/cats_0.zip'], 'none')

        print('\nMetrics for each label:')
        for label, metrics in metrics_for_label.items():
            hit_rate, mean_hit_score, mean_miss_score = metrics
            print('%s:\n  hit rate = %.5f\n  mean score for hits = %.5f\n  mean score for misses = %.5f' % (label, hit_rate, mean_hit_score, mean_miss_score))

    except ApiException as ex:
        print(ex)


if __name__ == '__main__':
    main()
