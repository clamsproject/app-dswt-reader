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
from typing import Tuple, Sequence, Any, List

import numpy as np
import torch
from clams import ClamsApp, Restifier
from doctr.models import ocr_predictor
from lapps.discriminators import Uri
from mmif.utils import video_document_helper as vdh

# Imports needed for Clams and MMIF.
# Non-NLP Clams applications will require AnnotationTypes

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes
import mmif
from collections import defaultdict
import editdistance

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
        # same as a reader in docTRWrapper
        self.reader = ocr_predictor(det_arch='db_resnet50', reco_arch='parseq',
                                    pretrained=True,
                                    paragraph_break=0.035,
                                    assume_straight_pages=False, detect_orientation=True)
        if torch.cuda.is_available():
            self.gpu = True
            self.reader = self.reader.cuda().half()
        else:
            self.gpu = False

        self.tp2sentences_final = {}
        self.TP2bb = defaultdict(dict)
        self.pa2txt = {}


    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory.
        # When using the ``metadata.py`` leave this do-nothing "pass" method here.
        pass

    def read_mmif(self, mmif_file) -> mmif.Mmif:
        """return mmif object from mmif file"""
        with open(mmif_file, 'r') as file:
            mmif_obj = mmif.Mmif(file.read())
            return mmif_obj

    def find_overlap(self, list1: list[str], list2: list[str]):
        """
        Find overlab between two lists of sentences
        For example,
        :param list1: [sentence1, sentence2, sentence3, sentence4]
        :param list2: [sentence3, sentence4, ...]
        :return: [sentence3, sentence4]
        """
        for i in range(len(list1)):
            if editdistance.eval(list1[i], list2[0]) / len(list1[i]) < 0.5:
                overlap_len = len(list1) - i
                return list1[i:], list2[:overlap_len]
        return [], []

    def resampleTP_at_best_interval(self, video_doc: Document, start_TP_annotation: Annotation, end_TP_annotation: Annotation, n=20, step=1000):
        """
        Resample the timepoints between start and end in targets at the best interval.
        :param video_doc: video document
        :param start_TP_annotation: annotation of starting timepoint
        :param end_TP_annotation: annotation of ending timepoint
        :param n: The initial number of scenes to sample in order to find the best interval, which will be the first
                n timepoints from the beginning.
        :param step: depends on the unit of the timestamp. Default = 1000 for millisecond.
        :return: a list of resampled timepoints
        """
        # Assume default TimePoint unit is ms (step=1000).
        target_start_timestamp = start_TP_annotation.get("timePoint")
        target_end_timestamp = end_TP_annotation.get("timePoint")
        TP_annotation = start_TP_annotation
        timeUnit = TP_annotation.get("timeUnit")
        fps = video_doc.get("fps")

        # 1. create a list of first n timestamp.
        first_n_timestamp = []
        for i in range(n):
            curr = target_start_timestamp + i * step
            first_n_timestamp.append(curr)
            if curr + step > target_end_timestamp:
                break

        # 2. Run docTR reader on a scene(image) of each timepoint in the first_n_timestamp. Split the result text
        # into sentences and store them in a list, then create a dictionary that maps each timepoint to a list of
        # sentences.
        tp2sentences = {}
        for TP in first_n_timestamp:
            tp_frame_index = vdh.convert(TP, timeUnit, "frame", fps)
            image: np.ndarray = vdh.extract_frames_as_images(video_doc, [tp_frame_index], as_PIL=False)[0]
            result = self.reader([image])
            sentences = []
            for block in result.pages[0].blocks:
                for line in block.lines:
                    sentences.append(line.render())
            tp2sentences[TP] = sentences

        # 3. Find the best interval that minimizes overlaps while ensuring nothing is missed by incrementing the
        # timepoint interval one by one and comparing the sentences of two consecutive timepoints.
        # This part used to be a function find_best_interval(sorted_tp, map_tp2sentences):
        best_interval = len(first_n_timestamp) - 1
        for j in range(1, len(first_n_timestamp)):
            num_overlap = 0
            for i in range(len(first_n_timestamp) - j):
                prev = first_n_timestamp[i]
                cur = first_n_timestamp[i + j]
                overlap1, overlap2 = self.find_overlap(tp2sentences[prev], tp2sentences[cur])
                if len(overlap1) > 0:
                    num_overlap += 1
            if num_overlap == 0:
                best_interval = j - 1
                break
            elif num_overlap == 0 and j == 0:
                best_interval = 1
                break

        # 4. Resample the timepoints between start and end in targets at the best interval found. Return them as a list.
        list_TP = []
        for TP in range(target_start_timestamp, target_end_timestamp, step * best_interval):
            list_TP.append(TP)
        if target_end_timestamp not in list_TP:
            list_TP.append(target_end_timestamp)

        return list_TP

    def scene_w_multicolumn(self, y_threshold=0.00919):
        """
        Find all scenes with text with multiple columns using coordinates of bounding box of each paragraph and group the consecutive scenes together.
        :return: A list of lists where each sublist contains consecutive scenes with multi columns of text grouped together.
        """
        sorted_tp = sorted(list(self.TP2bb.keys()))
        scene_w_multicolumn = []
        prev = False
        for tp in sorted_tp:
            data = self.TP2bb[tp]
            # When grouping paragraphs with similar y-coordinates of their bounding box start points, if there is a
            # group with more than one element, it is considered that there are two or more parallel columns or text
            # blocks within that scene.
            grouped_by_y = self.group_keys_by_starting_y(data, y_threshold)
            loop_break = False
            for list_pa in grouped_by_y:
                if len(list_pa) > 1 and not prev:
                    scene_w_multicolumn.append([tp])
                    prev = True
                    loop_break = True
                    break
                elif len(list_pa) > 1 and prev:
                    scene_w_multicolumn[-1].append(tp)
                    prev = True
                    loop_break = True
                    break
                else:
                    loop_break = False
            if not loop_break:
                prev = False

        return scene_w_multicolumn

    def read_text_from_scenes(self, list_tp: list, video_doc: Document, timeunit, fps):
        """
        Run docTR reader on the scenes of video_doc from timepoints in the list_TP and fill the following attributes
        based on the text read.:
        self.tp2sentences_final = {}
        self.TP2bb = defaultdict(dict)
        self.pa2txt = {}
        """
        idx = 1
        for TP in list_tp:
            tp_frame_index = vdh.convert(TP, timeunit, "frame", fps)
            image: np.ndarray = vdh.extract_frames_as_images(video_doc, [tp_frame_index], as_PIL=False)[0]
            result = self.reader([image])
            pa2bb = {}
            sentences = []
            for block in result.pages[0].blocks:
                pa_id = "pa_" + str(idx)
                pa2bb[pa_id] = self.rel_coordinate_pair(block.geometry)
                self.pa2txt[pa_id] = block.render()
                idx += 1
                for line in block.lines:
                    sentences.append(line.render())
            self.TP2bb[TP] = pa2bb
            self.tp2sentences_final[TP] = sentences

    def read_multiple_blocks_in_order(self, tp_w_multicolumns_list: list, x_threshold: float, y_limit: float):

        def group_keys_by_starting_x(data: dict, x_threshold: float, y_limit: float):
            """
            A function that groups paragraphs with similar x-coordinates of the starting point (x1) of bounding boxes,
            given as [[x1, y1], [x2, y2]], where [x1, y1] is the top-left corner and [x2, y2] is the bottom-right corner.
            :param data: a dictionary of [[x1, y1], [x2, y2]], e.g.: {'pa_204': [[0.462, 0.78], [0.835, 0.100]]}
            :param x_threshold: A relative threshold value (0-1) for the x-coordinate to determine how close the text
                                blocks need to be horizontally to be grouped together.
            :param y_limit: A relative value (0-1) for the y-coordinate
                          : if it is vertically farther apart than this value, it is considered a separate group
                             even if the x-coordinate difference is within the x_threshold.
            :return: A list of keys (paragraph ids) grouped by similar x-coordinates of their bounding box start points.
            """
            grouped_keys = []
            visited = set()

            for key1 in data:
                if key1 in visited:
                    continue
                group = [key1]
                visited.add(key1)
                value1_x = data[key1][0][0]
                value1_y = data[key1][0][1]

                for key2 in data:
                    if key2 in visited:
                        continue
                    value2_x = data[key2][0][0]
                    value2_y = data[key2][0][1]

                    # We will check if the y-coordinate difference between the last element in the group and the current
                    # element is within a specified limit. To add the constraint that the last element in a group
                    # should not differ too much in the y-coordinate from the current element.
                    if abs(value1_x - value2_x) <= x_threshold and abs(data[group[-1]][1][1] - value2_y) <= y_limit:
                        group.append(key2)
                        visited.add(key2)

                grouped_keys.append(group)

            return grouped_keys

        def get_first_x_coordinate(data: dict, key):
            return data[key][0][0]

        def get_last_y_coordinate(data: dict, key):
            return data[key][-1][1]

        def merge_groupings(grouped1: list[list], grouped2: list[list], data1: dict, data2: dict, x_threshold: float):
            """
            When two or more consecutive scenes contain text with multiple columns, there is a possibility that the
            text continues within the same column. Group paragraphs within a scene classified as multi-column into
            lists based on similar x-coordinates, and then merge lists with similar x-coordinates from consecutive
            scenes. The order of elements within each list is preserved.

            It finds the best matching group in grouped1 for each group in grouped2 by comparing the x-coordinates of
            the starting point of paragraphs. If there are multiple matches in grouped1, the group with the largest
            y-coordinate of the ending point (= the group of paragraphs that ends at the very bottom of those multiple
            matches) is chosen.

            :return: a merged list of grouped paragraphs (grouped1 and grouped2) and their corresponding coordinate dictionary
            """
            merged_groups = []
            for group1 in grouped1:
                first_x1 = get_first_x_coordinate(data1, group1[0])
                max_y1 = get_last_y_coordinate(data1, group1[-1])
                merged_group = group1.copy()
                for group2 in grouped2:
                    first_x2 = get_first_x_coordinate(data2, group2[0])
                    if abs(first_x1 - first_x2) <= x_threshold:
                        # Find the group in grouped1 with the largest y-coordinate
                        best_group = None
                        best_y = -float('inf')
                        for g in grouped1:
                            if abs(get_first_x_coordinate(data1, g[0]) - first_x2) <= x_threshold:
                                last_y = get_last_y_coordinate(data1, g[-1])
                                if last_y > best_y:
                                    best_y = last_y
                                    best_group = g
                        # If the current group1 is the best group, merge group2 into it
                        if best_group == group1:
                            merged_group.extend(group2)
                merged_groups.append(merged_group)
            return merged_groups

        # A group-merge loop for handling cases where three or more consecutive scenes with multi-column text need to be merged.
        data1 = self.TP2bb[tp_w_multicolumns_list[0]]
        grouped1 = group_keys_by_starting_x(data1, x_threshold, y_limit)
        if len(tp_w_multicolumns_list) > 1:
            for i in range(len(tp_w_multicolumns_list) - 1):
                tp2 = tp_w_multicolumns_list[i + 1]
                data2 = self.TP2bb[tp2]
                grouped2 = group_keys_by_starting_x(data2, x_threshold, y_limit)
                result = merge_groupings(grouped1, grouped2, data1, data2, x_threshold)
                data1.update(data2)
                grouped1 = result

        return grouped1, data1

    def group_keys_by_starting_y(self, data: dict, y_threshold):
        """
        A function that groups paragraphs with similar x-coordinates of the starting point (y1) of bounding boxes,
        given as [[x1, y1], [x2, y2]], where [x1, y1] is the top-left corner and [x2, y2] is the bottom-right corner.
        :param data: a dictionary of [[x1, y1], [x2, y2]], e.g.: {'pa_204': [[0.462, 0.78], [0.835, 0.100]]}
        :param threshold: relative value 0-1
        :return: A list of keys (paragraph ids) grouped by similar y-coordinates of their bounding box start points.
        """
        grouped_keys = []
        visited = set()
        for key1 in data:
            if key1 in visited:
                continue
            group = [key1]
            visited.add(key1)
            value1 = data[key1][0][1]

            for key2 in data:
                if key2 in visited:
                    continue
                value2 = data[key2][0][1]

                if abs(value1 - value2) <= y_threshold:
                    group.append(key2)
                    visited.add(key2)

            grouped_keys.append(group)

        return grouped_keys

    def read_paragraphs_from_grouped_ids(self, grouped_ids:list[list], pa_id2txt: dict):
        """
        Split the text of all paragraphs within the lists of the grouped_id list into sentences,
        then put them in order into a single list and return that list.
        :param grouped_ids: a list of lists of paragraph ids grouped based on their bounding box coordinates.
        :param pa_id2txt: a dictionary mapping paragraph ids to their text (str).
        :return: a list of sentences(str)
        """
        sentences = []
        for id_list in grouped_ids:
            for id in id_list:
                sentences += pa_id2txt[id].split("\n")
            sentences[-1] = sentences[-1] + "\n"
        return sentences

    def rel_coordinate_pair(self, coords: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Assumes the passed shape is a rectangle, represented by top-left and bottom-right corners,
        and compute floor and ceiling based on the geometry.
        :param coords: a sequence of 4 tuples representing the bounding box coordinates generated by docTR reader.
        :return coords: a tuple of 2 coordinate tuples represented by top-left and bottom-right corners.
        """
        xs = [x for x, _ in coords]
        ys = [y for _, y in coords]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return (x1, y1), (x2, y2)

    def process_concatenation(self, multicolumn_list_TP, x_threshold=0.0117, y_limit=0.1838):
        """
        This function reads text from consecutive scenes with multiple columns of text in column order.
        It uses the earliest timepoint as the representative and updates the timepoint-to-sentences dictionary as
        <representative timepoint>: [sentences read in order]. Then, it concatenates the sentences in timepoint order
        and returns them. While concatenating the sentences, it checks for overlaps again to ensure there are no
        duplicates.
        :param multicolumn_list_TP: A list of lists where each sublist contains consecutive scenes with multi columns of text grouped together.
        :param x_threshold: A threshold value for the x-coordinate to determine how close the text blocks need to be horizontally to be grouped together.
        :param y_limit: A relative value (0-1) for the y-coordinate
                      : if it is vertically farther apart than this value, it is considered a separate group  even if
                        the x-coordinate difference is within the x_threshold.
        :return: a single string of the concatenated sentences.
        """
        # read text from consecutive scenes (in a TP_list) with multiple columns of text in column order.
        for TP_list in multicolumn_list_TP:
            representative_tp = TP_list[0]
            for tp in TP_list:
                # Only the representative timepoint is kept in the final version of timepoint-to-sentences dict.
                if tp != representative_tp:
                    del self.tp2sentences_final[tp]
            grouped_pa_ids, pa2bb = self.read_multiple_blocks_in_order(TP_list, x_threshold, y_limit)
            sentences = self.read_paragraphs_from_grouped_ids(grouped_pa_ids, self.pa2txt)
            self.tp2sentences_final[representative_tp] = sentences

        # concatenate the sentences in timepoint order
        sorted_tp = sorted(list(self.tp2sentences_final.keys()))
        concatenated_text = self.tp2sentences_final[sorted_tp[0]]
        # check for overlaps again
        for i in range(len(sorted_tp) - 1):
            prev_tp = sorted_tp[i]
            curr_tp = sorted_tp[i + 1]
            overlap1, overlap2 = self.find_overlap(self.tp2sentences_final[prev_tp], self.tp2sentences_final[curr_tp])
            concatenated_text += self.tp2sentences_final[curr_tp][len(overlap1):]

        return "\n".join(concatenated_text)


    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._annotate
        if self.gpu:
            self.logger.debug("running app on GPU")
        else:
            self.logger.debug("running app on CPU")
        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        input_view: View = mmif.get_views_for_document(video_doc.properties.id)[-1]

        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)

        for timeframe in input_view.get_annotations(AnnotationTypes.TimeFrame):
            # Initialize attributes every timeframe.
            self.tp2sentences_final = {}
            self.TP2bb = defaultdict(dict)
            self.pa2txt = {}
            # get first and last timepoints in the timeframe that is classified as a dynamic credit.

            # TODO
            # start_time = timeframe.get("start")
            # end_time = timeframe.get("end")

            target_start_id = timeframe.get("targets")[0]
            target_end_id = timeframe.get("targets")[-1]
            if Mmif.id_delimiter not in target_start_id:
                target_start_id = f'{input_view.id}{Mmif.id_delimiter}{target_start_id}'
            if Mmif.id_delimiter not in target_end_id:
                target_end_id = f'{input_view.id}{Mmif.id_delimiter}{target_end_id}'

            start_TP_annotation = mmif[target_start_id] # consider this as a representative
            end_TP_annotation = mmif[target_end_id]
            timeUnit = start_TP_annotation.get("timeUnit")
            fps = video_doc.get("fps")

            # Resample timepoints at the optimal interval
            list_TP = self.resampleTP_at_best_interval(video_doc, start_TP_annotation, end_TP_annotation, parameters['first_n_timepoints'], parameters['initial_interval'])
            # Read texts from the scenes in the timepoints resampled
            self.read_text_from_scenes(list_TP, video_doc, timeUnit, fps)
            # Find the scenes (timepoints) with multiple columns
            multicolumn_list_TP = self.scene_w_multicolumn(parameters['y_threshold'])

            # concatenate sentences considering reading orders in the texts with multiple columns
            result_text = self.process_concatenation(multicolumn_list_TP, parameters['x_threshold'], parameters['y_limit'])
            print(result_text)

            # Save the resulted text as a textdocument in the new view and align it to the corresponding timeframe.
            text_document: Document = new_view.new_textdocument(result_text)
            new_view.new_annotation(AnnotationTypes.Alignment, source = timeframe.long_id, target = text_document.long_id)

        return mmif



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
    app = DswtReader()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
