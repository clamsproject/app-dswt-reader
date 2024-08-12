import jiwer
from jiwer import wer, cer
import mmif
import argparse
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def read_mmif(mmif_file) -> mmif.Mmif:
    """return mmif object from mmif file"""
    with open(mmif_file, 'r') as file:
        mmif_obj = mmif.Mmif(file.read())
        return mmif_obj


def get_dswt_output(mmif_file) -> str:
    mmif_obj = read_mmif(mmif_file)
    for view in mmif_obj.views:
        if "dswt" in view.metadata.app:
            for annotation in view.annotations:
                if "TextDocument" in str(annotation.at_type) and "text" in annotation.properties:
                    return annotation.get("text").value
    return ""

def read_text_from_file(file_path):
    """
    Read text from a file and return it as a string.

    :param file_path: The path to the text file
    :return: The content of the file as a string
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    return content

def remove_empty_strings(input_list):
    """
    Remove empty strings and newline characters from a list of strings.

    :param input_list: The input list of strings
    :return: A new list with empty strings and newline characters removed
    """
    return [string for string in input_list if string not in ["", "\n"]]


def preprocessing(input_string):
    list_sentences = remove_empty_strings(input_string.split("\n"))
    list_sentences = jiwer.Strip()(list_sentences)
    list_sentences = jiwer.RemoveMultipleSpaces()(list_sentences)
    joined_sentences = "\n".join(list_sentences)
    return joined_sentences


def calculate_bleu(gold_text, predicted_text):
    """
    Calculate the BLEU score for a predicted text compared to the gold text.

    :param predicted_text: The predicted text (hypothesis)
    :param gold_text: The gold text (reference)
    :return: The BLEU score
    """
    predicted_text = preprocessing(predicted_text)
    predicted_text = predicted_text.replace("\n", " ")
    gold_text = preprocessing(gold_text)
    gold_text = gold_text.replace("\n", " ")

    # Tokenize the texts into lists of words
    gold = [gold_text.split()]  # BLEU expects a list of references
    predicted = predicted_text.split()

    # Use smoothing function to handle cases with small datasets
    smoothing = SmoothingFunction().method1

    # Calculate BLEU score
    bleu_score = sentence_bleu(gold, predicted, smoothing_function=smoothing)

    return bleu_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", help="File path to gold annotated txt file", type=str, required=True)
    parser.add_argument("--predicted_mmif", help="File path to output Mmif file with TextDocument from dswt-reader", type=str, required=True)
    parser.add_argument("--visualization", help="visualize the alignment between prediction and gold text", default = False)

    parsed_args = parser.parse_args()

    # Define the predicted text and the gold annotation
    gold_annotation = read_text_from_file(parsed_args.gold)
    predicted_text = get_dswt_output(parsed_args.predicted_mmif)

    # Preprocess the text
    gold_sentences = preprocessing(gold_annotation)
    predicted_sentences = preprocessing(predicted_text)

    if parsed_args.visualization:
        out = jiwer.process_words(gold_sentences, predicted_sentences)
        print(jiwer.visualize_alignment(out))

    # Calculate the WER and CER
    wer_score = wer(gold_sentences, predicted_sentences)
    cer_score = cer(gold_sentences, predicted_sentences)

    # Calculate BLEU score
    BLEU_score = calculate_bleu(gold_sentences, predicted_sentences)

    print(f"WER: {wer_score:.4f}")
    print(f"CER: {cer_score:.4f}")
    print(f"BLEU score: {BLEU_score:.4f}")
