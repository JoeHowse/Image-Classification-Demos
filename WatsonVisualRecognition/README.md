# Watson Visual Recognition Python Samples

These samples show how to use the Watson Developer Cloud Python SDK to access the Watson Visual Recognition service.  As well as basic usage of the default classifier, the samples cover training and use of custom classifiers.  They have been tested on macOS 10.15 (Catalina) with Python 3.5 and Watson Developer Cloud Python SDK 3.0.4.

The `basic_image_classification` and `custom_dog_classifier` samples are adapted by Nummist Media (Joseph Howse) from IBM samples.

The `image_tagger_for_mac` sample is by Nummist Media (Joseph Howse) with file-tagging code adapted from a Brett Terpstra sample.

## Setting up dependencies

1. Register for an IBM Cloud account at https://console.bluemix.net/registration.
2. Create a Watson Visual Recognition resource at https://console.bluemix.net/catalog/services/watson-vision-combined.
3. Go to https://cloud.ibm.com/resources, select the Watson Visual Recognition resource that you have just created in step [2], and find the resource's API key and URL.
4. Define an environment variable based on your 44-digit API key for the Watson Visual Recognition service: `WATSON_VISUAL_RECOGNITION_KEY=12345678901234567890123456789012345678901234`.
5. Define another environment variable based on the URL for the Watson Visual Recognition service: `WATSON_VISUAL_RECOGNITION_URL=https://gateway.watsonplatform.net/visual-recognition/api`.
6. Install the Watson Developer Cloud Python SDK using pip: `pip install --upgrade ibm-watson`. Alternatively, use easy_install: `easy_install --upgrade ibm-watson`. For troubleshooting, see the instructions at https://github.com/watson-developer-cloud/python-sdk.

## Finding the samples

Each of the following subfolders contains a different sample:

- `basic_image_classification` contains a sample script, `classify_image.py`, which classifies a single image using the default classifier. It is adapted from IBM's sample at https://github.com/watson-developer-cloud/python-sdk/blob/master/examples/visual_recognition_v3.py.
- `custom_dog_classifier` contains a set of sample scripts that should be run in the following order: `create_classifier.py`, `update_classifier.py`, and `classify_images.py`. They train and test a custom classifier for various breeds of dogs. They are adapted from IBM's sample at https://console.bluemix.net/docs/services/visual-recognition/tutorial-custom-classifier.html#creating-a-custom-model. `log.txt` contains a sample of `classify_images.py`'s console output, including the hit rate, mean score for hits, and mean score for misses for each label in the model.
- `image_tagger_for_mac` contains a a sample script, `tag_images.py`, which classifies the images in the `resources` folder using the default classifier and then tags the image files using macOS Finder tags. The file-tagging code is adapted from Brett Terpstra's sample at https://brettterpstra.com/2017/08/22/tagging-files-from-the-command-line/. `classify_results.json` contains a sample of the JSON response that `tag_images.py` gets from Watson Visual Recogntion. `untag_images.sh` cleans the `resources` folder by removing any tags (and other macOS extended file attributes) from the files therein.