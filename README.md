# Dswt Reader

## Description
A text reader for scenes with dynamic text using docTR's ocr_predictor as the OCR module.

## Input

The wrapper takes a [`VideoDocument`]('https://mmif.clams.ai/vocabulary/VideoDocument/v1/') with SWT 
[`TimeFrame`]('https://mmif.clams.ai/vocabulary/TimeFrame/v3/') annotations. Specifically, it uses the property `timePoint` of the first and last timepoints of the `target` in the TimeFrame classified as `credits` by the swt app.

The classification of whether each scene with text is credits or not is assumed to be perfectly handled by the swt-app. 

## Output

For each TimeFrame classified as credits, a single TextDocument is generated and added to the MMIF as a new view. 
The text value of the TextDocument stores the text extracted from the dynamic credits in the best possible reading order, considering the positional arrangement of text blocks or columns within each scene.

The best reading order is usually with job titles followed by the corresponding name (or names).

The TextDocument is aligned to the TimeFrame.


## User instruction

- General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

- [The documentation for docTR](https://mindee.github.io/doctr/modules/io.html#document-structure), the OCR model used in this app.

- The `examples/gold_transcriptions/` folder contains gold annotations, formatted as follows:

  - Each Job Title and corresponding Names are listed as follows, based on the placement of the text within the scenes. Each Job title-Names pair is separated by two newlines (`\n\n`):
    - `<Job title> <name>`
    or
    - `<Job title>\n<names>`
    or
    - `<Job title> <name>\n<names>`

  - Logo Part: Logos are annotated using `<Logo>` or `<Logos>`.
  - Other texts are transcribed based on their placement within the scene.
  
- Start and end timePoint (in ms) of annotated TimeFrames in each example video
  - sample_video_1.mp4
  `<start>: 0, <end>: 282000`
  - sample_video_2.mp4
  `<start>: 10000, <end>: 250000`
  - sample_video_3.mp4
  `<start>: 52000, <end>: 295000`
  


### System requirements

- Requires mmif-python[cv] for the `VideoDocument` helper functions
- Requires GPU to run docTR model at a reasonable speed
- Please refer to the requirements.txt for the required libraries and their version information. 

### Configurable runtime parameter

For the full list of parameters, please refer to the app metadata from the [CLAMS App Directory](https://apps.clams.ai) or the [`metadata.py`](metadata.py) file in this repository.
