# Watson Natural Language Understanding Python Samples

This sample shows how to use the Watson Developer Cloud Python SDK to access the Watson Visual Recognition service. It has been tested on Mac OS 10.11 (El Capitan) with Python 3.5 and Watson Developer Cloud Python SDK 3.0.4.

By Nummist Media (Joseph Howse).

## Setting up dependencies

1. Register for an IBM Cloud account at https://console.bluemix.net/registration.

2. Create a Watson Natural Language Understanding resource at https://cloud.ibm.com/catalog/services/natural-language-understanding.

3. Go to https://cloud.ibm.com/resources, select the Watson Natural Language Understanding resource that you have just created in step [2], and find the resource's API key and URL.

3. Define an environment variable based on your 44-digit API key for the Watson Natural Language Understanding service: `WATSON_NATURAL_LANGUAGE_UNDERSTANDING_KEY=12345678901234567890123456789012345678901234`.

4. Define another environment variable based on the URL for the Watson Natural Language Understanding service: `WATSON_NATURAL_LANGUAGE_UNDERSTANDING_URL=https://gateway.watsonplatform.net/natural-language-understanding/api`.

6. Install the Watson Developer Cloud Python SDK using pip: `pip install --upgrade ibm-watson`. Alternatively, use easy_install: `easy_install --upgrade ibm-watson`. For troubleshooting, see the instructions at https://github.com/watson-developer-cloud/python-sdk.
