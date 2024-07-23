"""
DELETE THIS MODULE STRING AND REPLACE IT WITH A DESCRIPTION OF YOUR APP.

app.py Template

The app.py script does several things:
- import the necessary code
- create a subclass of ClamsApp that defines the metadata and provides a method to run the wrapped NLP tool
- provide a way to run the code as a RESTful Flask service


"""

import argparse
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from math import floor, ceil
from typing import Tuple, Sequence

import numpy as np
import torch
from clams import ClamsApp, Restifier
from doctr.models import ocr_predictor
from lapps.discriminators import Uri
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh

# Imports needed for Clams and MMIF.
# Non-NLP Clams applications will require AnnotationTypes

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes

# For an NLP tool we need to import the LAPPS vocabulary items
from lapps.discriminators import Uri


class DswtReader(ClamsApp):

    def __init__(self):
        super().__init__()
        # default docTR configs:
        # det_arch='db_resnet50' (keeping it)
        # reco_arch='crnn_vgg16_bn',
        # pretrained=False,
        # paragraph_break=0.035, (keeping it)
        # assume_straight_pages=True
        # detect_orientation=False,
        self.reader = ocr_predictor(det_arch='db_resnet50', reco_arch='parseq',
                                    pretrained=True,
                                    paragraph_break=0.035,
                                    assume_straight_pages=False, detect_orientation=True)
        if torch.cuda.is_available():
            self.gpu = True
            self.reader = self.reader.cuda().half()
        else:
            self.gpu = False


    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory.
        # When using the ``metadata.py`` leave this do-nothing "pass" method here.
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._annotate
        raise NotImplementedError

def get_app():
    """
    This function effectively creates an instance of the app class, without any arguments passed in, meaning, any
    external information such as initial app configuration should be set without using function arguments. The easiest
    way to do this is to set global variables before calling this.
    """
    # for example:
    # return DswtReader(create, from, global, params)
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    # if get_app() call requires any "configurations", they should be set now as global variables
    # and referenced in the get_app() function. NOTE THAT you should not change the signature of get_app()
    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
