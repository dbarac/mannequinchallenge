# Mannequin Challenge Code and Trained Models

This repository contains inference code for models trained on the Mannequin
Challenge dataset introduced in the CVPR 2019 paper "[Learning the Depths of
Moving People by Watching Frozen People](https://mannequin-depth.github.io/)."

This is not an officially supported Google product.

## Setup

The code is based on PyTorch. The code has been tested with PyTorch 1.1 and Python 3.6. 

We recommend setting up a `virtualenv` environment for installing PyTorch and
the other necessary Python packages. The [TensorFlow installation
guide](https://www.tensorflow.org/install/pip) may be helpful (follow steps 1
and 2) or follow the `virtualenv` documentation.

Once your environment is set up and activated, install the necessary packages:

```
(pytorch)$ pip install torch torchvision scikit-image h5py
```

The model checkpoints are stored on Google Cloud and may be retrieved by running:

```
(pytorch)$ ./fetch_checkpoints.sh
```

## Single-View Inference

Our test set for single-view inference is the [DAVIS
2016](https://davischallenge.org/davis2016/code.html) dataset. Download and
unzip it by running:

```
(pytorch)$ ./fetch_davis_data.sh
```

Then run the DAVIS inference script:

```
(pytorch)$ python test_davis_videos.py --input=single_view
```

Once the run completes, visualizations of the output should be
available in `test_data/viz_predictions`.

## Full Model Inference

The full model described in the paper requires several additional inputs: the
human segmentation mask, the depth-from-parallax buffer, and (optionally) a
human keypoint buffer. We provide a preprocessed version of the [TUM
RGBD](https://vision.in.tum.de/data/datasets/rgbd-dataset) dataset that includes
these inputs. Download (~9GB) and unzip it using the script:

```
(pytorch)$ ./fetch_tum_data.sh
```

To reproduce the numbers in Table 2 of the paper, run:

```
(pytorch)$ python test_tum.py --input=single_view
(pytorch)$ python test_tum.py --input=two_view
(pytorch)$ python test_tum.py --input=two_view_k
```

Where `single_view` is the variant _I_ from the paper, `two_view` is the variant _IDCM_, and `two_view_k` is the variant _IDCMK_. The script prints running averages of the various error metrics as it runs. When the script completes, the final error metrics are shown.

## Dodatno

### Verzije paketa
Dodan je requirements.txt file sa verzijama korištenih python paketa za lakšu instalaciju. Na listi je i opencv-python paket koji se koristi u skripti za pretvaranje videa u pojedine frame-ove.
```
pip install -r requirements.txt
```
### Testiranje modela na odabranim video datotekama

Dodana skripta video2frames.py video datoteke dijeli na .jpeg frame-ove koji se koriste kao input za model. Frame-ovi se spremaju u test_data/user_data/.
```
(pytorch)$ python video2frames.py test-video.mp4
```
Nakon toga se pokreće skripta test_user_videos.py
```
 (pytorch)$ python test_user_videos.py --input=single_view
```
Rezultati će biti u test_data/viz_predictions/user_data/

## Acknowledgements

If you find the code or results useful, please cite the following paper:

```
@article{li2019learning,
  title={Learning the Depths of Moving People by Watching Frozen People},
  author={Li, Zhengqi and Dekel, Tali and Cole, Forrester and Tucker, Richard
    and Snavely, Noah and Liu, Ce and Freeman, William T},
  journal={arXiv preprint arXiv:1904.11111},
  year={2019}
}
```
