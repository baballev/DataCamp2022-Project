# Multilabel Food Products Classification Challenge

Authors: Vincent Josse, Pierre Piovesan, Adame Ben Friha, Romin Durand, Benjamin Cohen & Huan Trochon

Datacamp project 2022 for Master Data Science IP Paris


After cloning, run

```
pip install -r requirements.txt
```
and then
```
python download_data.py
```

This will install the necessary packages and create the folders `data/train` and `data/test`. It will also download the images from `data_locations_public.csv` and then create 2 separate `labels.csv` files. You need approximately 2 GB disk space to download every images, and it can take about 2 hours with a decent connection to retrieve every image because we are using a single process. There are 50,000 training images and 5,000 test images.
