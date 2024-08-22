# Fluorescent Neuronal Cells

## Quick Start

This dataset contains 283 images of mice brain slices obtained through fluorescent microscopy, plus the correspondent
ground-truth masks for semantic segmentation and object detection.

The folder `all_images/images` contains the original images acquired during the experiment described in [1], while the
correspondent ground-truth masks were generated in [2].

## Terms of Use

All images in this archive are licensed under the Creative Commons Attribution Share Alike 4.0 International License,
available at: https://creativecommons.org/licenses/by-sa/4.0/legalcode

Specific copyright statements are reported separately for the images and the masks.

When using this dataset, please cite the work that introduced it [2].

## Abstract

By releasing this dataset, we aim at providing a new testbed for computer vision techniques using Deep Learning. The main peculiarity is the shift from the domain of "natural images" proper of common benchmark dataset to biological imaging.
We anticipate that the advantages of doing so could be two-fold: i) fostering research in biomedical-related fields - for which popular pre-trained models perform typically poorly - and ii) promoting methodological research in deep learning by addressing peculiar requirements of these images.
Possible applications include but are not limited to *semantic segmentation*, *object detection* and *object counting*.

The data consist of 283 high-resolution pictures (1600x1200 pixels) of mice brain slices acquired through a fluorescence microscope. The final goal is to individuate and count neurons highlighted in the pictures by means of a marker, so to assess the result of a biological experiment. 

The corresponding ground-truth labels were generated through a hybrid approach involving semi-automatic and manual semantic segmentation. The result consists of black (0) and white (255) images having pixel-level annotations of where the stained neurons are located. For more information, please refer to [2].

## Description

The images depict neurons of interest as objects of different size and shape appearing as yellow-ish spots of variable
brightness and saturation over a composite, generally darker background. For more details refer to *Dataset* section in [2].

A summary table with average measures<strong>*</strong> of main cell characteristics is reported below:

|   area  | minor axis | major axis | equivalent diameter | maximum feret diameter | mean diameter |
|:-------:|:----------:|:----------:|:-------------------:|:----------------------:|:-------------:|
| 1206.43 |    29.39   |    50.43   |        36.50        |          55.34         |     47.42     |

<div class="alert alert-block alert-info">
<strong>*</strong> obtained with <i>skimage 0.18.1</i> python package
</div>  

### Challenges

Several relevant challenges are present:

- variability in brightness and contrast causes some fickleness in the pictures overall appearance
- cells exhibit varying saturation levels due to the natural fluctuation of the fluorescent emission properties
- substructures of interest have a fluid nature, so the shape of the stained cells may change significantly
- artifacts, bright biological structures -- like neurons' filaments -- and non-marked cells are present
- cells are sometimes clumping together and/or overlapping each other
- broad shift in the number of target cells from image to image, from no stained cells to several dozens

All of these factors make the segmentation/recognition task harder, sometimes creating borderline cases that lead to a
subjective interpretation.

## Ground-truth labels

Since generating annotations requires a great effort in terms of time and human resources, we resorted to an automatic
procedure to speed up the labeling. We started from a large subset of 252 images and applied gaussian blurring to remove
noise. The cleaned images were then subjected to a thresholding operation based on automatic histogram shape-based
methods. The goal was to obtain a loose selection of the objects that may seem good candidates to be labeled as neuronal
cells. After that, acknowledged operators reviewed the results to discard the false positives introduced with the
previous procedure, taking care of excluding irrelevant artifacts and misleading biological structures. The remaining 31
images were segmented manually by domain experts. We included significant pictures with peculiar traits -- such as
artifacts, filaments and crowded areas -- in the latter set to have highly reliable masks for the most challenging
examples.

The list of manually segmented images is reported below, all the remaining were obtained from the semi-automatic
labeling procedure:

*Mar31bS2C1R2_VLPAGr_200x_y.png*, *Mar33bS1C4R2_DMl_200x_y.png*, *Mar40S1C2R2_DMl_200x_o.png*, *Mar41S3C1R1_DMl_200x_o.png*,
*MAR38S1C3R1_LHR_20_o.png*, *MAR38S1C3R1_DML_20_o.png*, *Mar42S2C2R2_DMr_200x_o.png*, *Mar36bS1C6R2_DMl_200x_y.png*,
*39.png*, *Mar40S1C2R2_DMr_200x_o.png*, *Mar41S3C1R1_DMr_200x_o.png*, *Mar33bS2C1R1_DMl_200x_y.png*, *Mar43S1C5R3_DMr_200x_o.png*,
*Mar37S1C2R1_DMr_200x_o.png*, *Mar40S3C4R2_VLPAGr_200x_o.png*, *37.png*, *Mar37S1C2R1_DMl_200x_o.png*, *Mar36bS1C6R2_DMr_200x_y.png*,
*MAR55S3C2R2_VLPAGL_20_o.png*, *MAR39S2C2R2_DMR_200x_o.png*, *38.png*, *MAR55S3C2R2_VLPAGR_20_o.png*, *Mar31bS2C3R4_DMr_200x_y.png*,
*Mar32bS2C2R2_DMl_200x_y.png*, *MAR39S2C2R2_DML_200x_o.png*, *MAR52S2C1R3_LHL_20_o.png*, *Mar42S2C4R2_VLPAGr_200x_o.png*,
*MAR55S1C5R3_DMR_20_o.png*, *MAR38S1C3R1_DMR_20_o.png*, *Mar33bS1C4R2_DMr_200x_y.png*, *Mar41S3C3R3_VLPAGl_200x_o.png'*

## Fundings

The collection of original images was supported by funding from the University of Bologna (RFO 2018) and the
European Space Agency (Research agreement collaboration 4000123556).

## Ethical approval

All the experiments were conducted following approval by the National Health Authority (decree: No.141/2018 -
PR/AEDB0.8.EXT.4), in accordance with the DL 26/2014 and the European Union Directive 2010/63/EU, and under the
supervision of the Central Veterinary Service of the University of Bologna. All efforts were made to minimize the number
of animals used and their pain and distress.

## References

[[1]](https://www.nature.com/articles/s41598-019-51841-2) Hitrec, T., Luppi, M., Bastianini, S., Squarcio, F.,
Berteotti, C., Martire, V.L., Martelli, D., Occhinegro, A., Tupone, D., Zoccoli, G. and Amici, R., 2019. Neural control
of fasting-induced torpor in mice. Scientific reports, 9(1), pp.1-12.

[[2]](https://doi.org/10.1038/s41598-021-01929-5) Morelli, R., Clissa, L., Amici, R., Cerri, M., Hitrec, T.,
Luppi, M., Rinaldi, L., Squarcio, F. and Zoccoli, A., 2021. Automating cell counting in fluorescent microscopy through 
deep learning with c-ResUnet. Scientific reports, (in press).