"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


# DO NOT CHANGE the function name
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification.
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    
    # first set up some basic information
    metadata = AppMetadata(
        name="DSWT Reader",
        description="Text reader app for dynamic scenes with text",  # briefly describe what the purpose and features of the app
        app_license="Apache 2.0",  # short name for a software license like MIT, Apache2, GPL, etc.
        identifier="dswt-reader",  # should be a single string without whitespaces. If you don't intent to publish this app to the CLAMS app-directory, please use a full IRI format.
        url="https://github.com/clamsproject/app-dswt-reader",  # a website where the source code and full documentation of the app is hosted
        # (if you are on the CLAMS team, this MUST be "https://github.com/clamsproject/app-dswt-reader"
        # (see ``.github/README.md`` file in this directory for the reason)
        analyzer_version='0.8.1', # use this IF THIS APP IS A WRAPPER of an existing computational analysis algorithm
        # (it is very important to pinpoint the primary analyzer version for reproducibility)
        # (for example, when the app's implementation uses ``torch``, it doesn't make the app a "torch-wrapper")
        # (but, when the app doesn't implementaion any additional algorithms/model/architecture, but simply use API's of existing, for exmaple, OCR software, it is a wrapper)
        # if the analyzer is a python app, and it's specified in the requirements.txt
        # this trick can also be useful (replace ANALYZER_NAME with the pypi dist name)
        # analyzer_version=[l.strip().rsplit('==')[-1] for l in open('requirements.txt').readlines() if re.match(r'^ANALYZER_NAME==', l)][0],
        analyzer_license="Apache 2.0",  # short name for a software license
    )
    # and then add I/O specifications: an app must have at least one input and one output

    metadata.add_input(DocumentTypes.VideoDocument)
    in_tf = metadata.add_input(AnnotationTypes.TimeFrame, label="credits")
    in_tf.add_description('')

    out_td = metadata.add_output(DocumentTypes.TextDocument, **{'@lang': 'en'})
    out_td.add_description('')
    out_td = metadata.add_output(AnnotationTypes.Alignment)
    out_td.add_description('')

    ## Alignment?
    
    # (optional) and finally add runtime parameter specifications
    metadata.add_parameter(name='xThreshold', description='A relative threshold value (0-1) for the x-coordinate to determine how close the text blocks need to be horizontally to be grouped together.',
                           type='number', default=0.04)
    metadata.add_parameter(name='yLimit',
                           description='A relative value (0-1) for the y-coordinate: if it is vertically farther apart than this value, it is considered a separate group even if the x-coordinate difference is within the x_threshold.',
                           type='number', default=0.045)
    metadata.add_parameter(name='yThreshold',
                           description='A relative threshold value (0-1) for the y-coordinate to determine how close the text blocks need to be vertically to be grouped together.',
                           type='number', default=0.00919)
    metadata.add_parameter(name='firstNTimepoints', description='The initial number of timepoints to sample from the beginnign of the timeframe',
                           type='integer', default=20)
    metadata.add_parameter(name='initialInterval',
                           description='The initial interval to sample timepoints from the beginning of the timeframe (in milliseconds)',
                           type='integer', default=1000)

    # metadta.add_parameter(more...)
    
    # CHANGE this line and make sure return the compiled `metadata` instance
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
