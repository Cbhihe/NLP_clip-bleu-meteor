## Repo's contents

--------------------------------------------------------------------

### Notebook "01_clip_lexical-embedding_cpu-only_prod.ipynb"

- loads CLIP v1.0 in a Python 3.7.0 virtual environment.
- computes natural language embeddings for tokens of interest using the CLIP transformer model.

CLIP is the Contrastive Language-Image Pre-Training neural network trained on a variety of (image, text) pairs, first reported by Radford et al (2020) and applied by Hessel et al (2021).
Given a digitized image, CLIP claims it can predict the most relevant related caption, without having been specifically fine-tuned for the task, in a way similar to the zero-shot capabilities of GPT-2 and 3.

--------------------------------------------------------------------

### Notebook "02_clip_zero-shot-classifcation_cpu-only_prod.ipynb"

- loads CLIP v1.0 (Torch v1.7.1) in a Python 3.7.0 virtual environment.
- computes natural language embeddings for tokens of interest using the CLIP transformer model.
- implementats one-shot classification of annotation-image pairs.
- operates on selected images files on the condition that, after object detection, a well formed '*.xml' PASCAL-VOC metadata file specifies at least 3 bounding-boxes (bbx).
- classifies selected images according to the labels used during CLIP training.
- results are saved in file: '../Sgoab/Data/Images/clip_lab-class.tsv'

--------------------------------------------------------------------

### Notebook "03_clip_blue_meteor_similarity-scoring_prod.ipynb"

- extracts visual relationships as caption-seeds from VISREL analysis results for selected images, i.e. either manually annotated images (training dataset images) or never-seen-before images (test dataset images) processed by the Matterport Mask-RCNN  object detection transfer learning model,
- post-processes caption-seeds so they conform to the CLIP textual input requirements,
- computes CLIP image-summary feature-vector embeddings similarities between image and either reference or candidate summaries.
- computes BLEU and METEOR n-gram based similarities between reference and candidate summaries.
- tabulates thus obtained results to ease the visual comparison between the scoring schemes.


#### Sequence of actions:
**A)** Loads contents from the specific zifile './Data_git/crowd_set.zip' in-memory. The zip file contains a sample of 6 images as demo data for the open-sourced provided code.<BR>
    - Run existence checks on files  '<basename>.jpg', '<basename>_<infix>.xml' and '<basename>_<infix>_vrd.json', where <infix>  may take the values in {'test','train'} as explained further down:
        - Files '<basename>.jpg' are previously processed color images randomly sampled from the published DEArt dataset ([1])[https://zenodo.org/record/6984525] of mainly Christian paintings from the 12th to 18th European century period.
        - Files '<basename>_<infix>.xml' and '<basename>_<infix>.json' contain detected objects' bounding box (bbx) metadata for each image previously processed by a RCNN transfer-learning model [2]. The transfer-learning model was fine-tuned on the above referenced DEArt collection of paintings.
        - Files '<basename>_<infix>_vrd.json' contain further enrichements based on objects' bbxes' positions and labels analysis. Those enrichments take the form of image compositional annotations in relation to detected objects and objects' visual relationaship annotations for pairs of objects with overlapping or touching bbxes. Those annotation were obtained by means of a stand-alone post-processor, VISREL, a domain-specific pictorial semantics heuristics to be open-sourced shortly at https://www.github.com/Cbhihe/. Of specific interest to us is the dictionary value associated with the key "annot_txt" in the VISREL decoder output files. They constitute a set of caption-seeds, short specially crafted sentences of the form (S,V,O), or (S,sV,), where S=Subject, V=Verb, O=Object and sV=situation-Verb (similar but not exactly conformant to the definition of stative verbs).

**B)** Recover and clean up specially crafted caption-seeds from the json file '<basename>_<infix>_vrd.json' at key "annot_txt".
    - Just as for bbx annotations files, caption-seeds produced by VISREL and contained in files '<basename>_<infix>_vrd.json' have two flavors according to they <infix> value. The infix 'test' indicates that object detection was performed on an image never seen before by the RCNN transfer learning object detection model [2]. Those object detection annotations in the form of bbxes were then adjusted/corrected by hand in a semi-supervised approach meant to subsequently include the same image in the expanded training dataset. The resulting (modified) bounding box annotations correspond to files identified by the infix "train".


**C)** Implement CLIP to compute and save the image and textual summary features' cosine similarity between each image and available summaries for that image.<BR>
    - Available summaries include:
        - human references summaries collected as part of an internal caption crowd-sourcing campaign, to which 8 persons contributed.
        - "test" machine candidate summaries
        - "train" machine candidate summaries obtained as described above from the modified "test" machine candidates.


**D)** Implement METEOR and BLEU n-gram based similarity scoring using Porter-stemming, lemmatization aided by Wordnet lexical disambiguation for correct determination of lemmata.

***References:***<BR>
[1]   Reshetnikov A., Marinescu MC., More Lopez J.,  DEArt: Dataset of European Art (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6984525 (Oct. 2022).

[2]  Reshetnikov A. A faster RCNN transfer learning model trained on the DEArt dataset of paintings from the12th to 18th European century period. Private communication (Oct. 2022)


### Licensing terms and copyright

A substantial portion of the following code, as originally made available by its authors, is protected by the terms and conditions of the MIT license.

    Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

----------------------------------------------------------------------------

    Copyright (c) 2020,2021 Cedric Bhihe

The OpenAI software made available in this repository was extended with pre- and post-processing steps. The corresponding code is available for free under the terms of the Gnu GPL License version 3 or later.

In short, you are welcome to use and distribute the code of this repo as long as you always quote its author's name and the Licensing terms in full.

Consult the licensing terms and conditions in License.md on this repo.

----------------------------------------------------------------------------

***To contact the repo owner for any issue, please open an new thread under the [Issues] tab on the source code repo.***

### Guidelines to run the code

This iPython notebook must be run in a Python virtual environment, running Python v3.7.1. This is a prerequisite so the proper versions of Torch 1.7.1+cpu and TorchVision 0.8.2+cpu can be invoked to run the CLIP 1.0 inference engine on test images. A succint installation description is scripted below for a Linux host, assuming that:

- your interactive terminal session shell is `bash` or `sh`.
- you already setup a Python 3.7.0 virtual environment, in directory `/path/to/my_directory`.
- you know how to handle the command line interface on the terminal.

#### Setting up and registering a custom iPython kernel

What follows applies to CPU-only constrained installations. For CUDA-enabled machines, refer to `https://github.com/OpenAI/CLIP`.

- Assuming you have configured `pyenv` (on your favorite Linux host) to manage multiple Python virtual environments with specific package requirements, choose the directory in which to setup your Python virtual environment and install your iPython kernel utility package `ipykernel`:

```
      $ cd /path/to/my_directory
      $ pyenv local 3.7.1
      $ python -m pip install ipykernel  # or "ipykernel==4.10.0"
```
- Install every required packages in the virtual environment directory `/path/to/my_directory` (see following section for that).

```
      $ python -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
      $ python -m pip install ftfy regex tqdm
      $ python -m  pip install git+https://github.com/openai/CLIP.git
```

- Make the new custom iPython kernel, clip1.0, available in interactive Python sessions:
```
      $ cd /path/to/my_directory
      $ ipython kernel install --user --name clip1.0 --display-name "Python3.7.1 (clip1.0)"     # or
      $ python -m ipykernel install --user --name clip1.0 --display-name "Python3.7.1 (clip1.0)"
      $ jupyter notebook        # launch an iPython session based on 'notebook' server
```

- Select the special virtual environment kernel ***Python 3.7.1 (clip1.0)*** under the `New notebook` button in the top-right region of the browser page.

### Package requirements

Package requirements are detailed below. For a quick demo also install `Pillow==8.3.2` and dependencies.

- Install all required packages in the virtual environment directory "/path/to/my_directory", with:
```
    $ cd /path/to/my_directory
    $ python -m pip freeze <<- 'EOF'
                clip @ git+https://github.com/openai/CLIP.git@04f4dc2ca1ed0acc9893bd1a3b526a7e02c4bb10ftfy
                Cython==0.29.1
                h5py==2.9.0
                ftfy==5.5.1
                matplotlib==3.0.2
                numpy==1.17.3
                Pillow==8.3.2
                pyyaml==5.1
                regex==2021.8.3
                requests==2.20.1
                torch==1.7.1+cpu
                torchaudio==0.7.2
                torchvision==0.8.2+cpu
                tqdm==4.38.0
                scipy==1.2.0
                zipfile37==0.1.3
    EOF
```

Jupyter environment requirements include:
```
                ipykernel==6.6.0
                ipython==7.30.1
                ipython_genutils==0.2.0
                ipywidgets==7.6.5
                jupyter_client==7.1.0
                jupyter_core==4.9.1
                nbclient==0.5.9
                nbconvert==6.3.0
                nbformat==5.1.3
                notebook==5.7.4
                traitlets==5.1.1
```
 ... and starting the jupyter notebook from the sytem's jupyter's instance with:
```
    $ /usr/bin/jupyter notebook 01_clip_cpu_classify
```
#### Known issues
\- Launching the notebook by relying on the local environment's shims, with:

    $ jupyter notebook 01_clip_cpu_classify

may fail under Pyenv with a "Segmentation fault". It is likely an iPython issue related to jupyter. To avoid it, launch either notebook from a more recent python version, and select iyour custom built 3.7.1 iPython kernel from the notebook at first launch.
