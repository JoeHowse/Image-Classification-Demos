#!/usr/bin/env python


from __future__ import print_function
import glob
import json
import os

from ibm_watson import NaturalLanguageUnderstandingV1, ApiException
from ibm_watson.natural_language_understanding_v1 import \
    Features, ConceptsOptions, EmotionOptions, EntitiesOptions, KeywordsOptions, \
    RelationsOptions, SemanticRolesOptions, SentimentOptions, CategoriesOptions, \
    SyntaxOptions


def main():

    try:

        service = NaturalLanguageUnderstandingV1(
            version='2018-03-16',
            url=os.environ['WATSON_NATURAL_LANGUAGE_UNDERSTANDING_URL'],
            iam_apikey=os.environ['WATSON_NATURAL_LANGUAGE_UNDERSTANDING_KEY'])

        concepts = ConceptsOptions()
        emotion = EmotionOptions()
        entities = EntitiesOptions()
        keywords = KeywordsOptions()
        metadata = None
        relations = RelationsOptions()
        semantic_roles = SemanticRolesOptions()
        sentiment = SentimentOptions()
        categories = CategoriesOptions()
        syntax = None

        features = Features(concepts, emotion, entities, keywords, metadata, relations,
                            semantic_roles, sentiment, categories, syntax)

        text_file_paths = glob.glob('input/*.txt')
        for i in range(len(text_file_paths)):

            with open(text_file_paths[i], 'r') as text_file:
                text = text_file.read()

            response = service.analyze(text=text, features=features).get_result()

            with open('output/description_%d.txt' % i, 'w') as description_file:
                print(json.dumps(response, indent=2), file=description_file)

            with open('output/keywords_%d.csv' % i, 'w') as keywords_file:
                print('{0},{1},{2}'.format('text', 'count', 'relevance'),
                      file=keywords_file)
                for keyword in response['keywords']:
                    text = keyword['text']
                    count = keyword['count']
                    relevance = keyword['relevance']
                    print('{0},{1},{2}'.format(text, count, relevance),
                          file=keywords_file)

    except ApiException as ex:
        print(ex)


if __name__ == '__main__':
    main()
