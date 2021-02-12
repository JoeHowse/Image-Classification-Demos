#!/usr/bin/env python


from __future__ import print_function

import json
import os
import timeit

from ibm_watson import VisualRecognitionV3, ApiException


def classify_images_file(service, filename, language='en', threshold=0.1):
    try:
        absolute_path = os.path.abspath(filename)
        with open(absolute_path, 'rb') as images_file:
            classify_result = service.classify(
                images_file=images_file,
                threshold='{0}'.format(threshold),
                classifier_ids=['default'],
                accept_language=language).get_result()
            return classify_result
    except ApiException as ex:
        print(ex)
        return None


def tag_file(filename, tags):

    xml = '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" ' \
          '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">' \
          '<plist version="1.0"><array>{0}</array></plist>'.format(
              ''.join(map(lambda s:'<string>{0}</string>'.format(s), tags)))

    command = 'xattr -w com.apple.metadata:_kMDItemUserTags ' \
              '\'{0}\' \'{1}\''.format(xml, filename)

    os.system(command)


def untag_file(filename):

    command = 'xattr -d com.apple.metadata:_kMDItemUserTags ' \
              '\'{0}\''.format(filename)

    os.system(command)


def main():

    run_offline = False

    """Supported languages:
    'ar': Arabic
    'de': German
    'en': English
    'es': Spanish
    'fr': French
    'it': Italian
    'ja': Japonese
    'ko': Korean
    'pt-br': Portuguese (Brazil)
    'zh-cn': Chinese (China)
    'zh-tw': Chinese (Taiwan)
    """
    language = 'es'

    images_dir = 'resources'
    images_zip = '{0}.zip'.format(images_dir)
    cache_json = 'classify_result.json'

    service = VisualRecognitionV3(
        '2018-03-19',
        url=os.environ['WATSON_VISUAL_RECOGNITION_URL'],
        iam_apikey=os.environ['WATSON_VISUAL_RECOGNITION_KEY'])

    if run_offline:
        classify_result = None
    else:
        t0 = timeit.default_timer()
        os.system('zip -r {0} {1} -x "{1}/.*" -x "{1}/LICENSE.txt"'.format(
            images_zip, images_dir))
        start_time = timeit.default_timer()
        classify_result = classify_images_file(service, images_zip, language)
        end_time = timeit.default_timer()
        print('Query to Watson ran in %.3f seconds' % (end_time - start_time))
        os.system('rm {0}'.format(images_zip))

    if classify_result is None:
        with open(cache_json) as f:
            classify_result = json.load(f)
    else:
        with open(cache_json, 'w') as f:
            json.dump(classify_result, f, indent=2)

    for image in classify_result['images']:

        filename = os.path.join(images_dir, os.path.split(image['image'])[1])

        classes = image['classifiers'][0]['classes']
        score_max = max([c['score'] for c in classes])
        score_threshold = 0.75 * score_max
        tags = [c['class'] for c in classes if c['score'] > score_threshold]

        print('Tags for {0}: {1}'.format(filename, tags))

        tag_file(filename, tags)


if __name__ == '__main__':
    main()
