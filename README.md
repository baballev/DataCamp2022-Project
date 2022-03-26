# Multilabel Food Products Classification Challenge

Authors: Vincent Josse, Pierre Piovesan, Adame Ben Friha, Romin Durand, Benjamin Cohen & Huan Trochon

## Datacamp project 2022 for Master Data Science IP Paris

### Challenge Description



![OpenFoodFacts](https://world.openfoodfacts.org/images/misc/openfoodfacts-logo-en-178x150.png)

This challenge consists ion classifying images from the [OpenFoodFacts](https://world.openfoodfacts.org/) database by attributing them their tags. There are 30 tags for this challenge, and each image correspond to one or multiple tags. In other words, this challenge is a Multilabel Classification Task.


### Installation Instructions

After cloning, run

```
pip install -r requirements.txt
```
and then
```
python download_data.py
```

This will install the necessary packages and create the folders `data/train` and `data/test`. It will also download the images from `data_locations_public.csv` and then create 2 separate `labels.csv` files. You need approximately 2 GB disk space to download every images, and it can take about 2 hours with a decent connection to retrieve every image because we are using a single process. There are 50,000 training images and 5,000 test images of different sizes.


### Metrics

The challenge is ranked based on the average between F1-score and accuracy.


### Submission

To test your submission, you can run 
```
ramp-test --submission starting_kit
```

If you just want to debug and check that your script executes completely, you can run
```
ramp-test --submission starting_kit --quick-test
```
