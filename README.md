# Multilabel Food Products Classification Challenge

Authors: Vincent Josse, Pierre Piovesan, Adame Ben Friha, Romin Durand, Benjamin Cohen & Huan Trochon

## Datacamp project 2022 for Master Data Science IP Paris

### Challenge Description



![OpenFoodFacts](https://world.openfoodfacts.org/images/misc/openfoodfacts-logo-en-178x150.png)

This challenge consists in classifying images from the [OpenFoodFacts](https://world.openfoodfacts.org/) database by attributing them their tags. There are 30 tags for this challenge, and each image correspond to one or multiple tags. In other words, this challenge is a Multilabel Classification Task.

### Data Description

Open Food Facts is a collaborative project aimed at building a free and open database of food products marketed worldwide. The media sometimes compare it to Wikipedia because of its collaborative way of working and the use of free licenses.

For each product listed, we find, in particular, a generic name of the product, its quantity, the packaging (cardboard, frozen, etc.), the brand or brands of the product (in particular the main brand and any subsidiary brands), the food category of the product to allow comparisons, places of manufacture or processing, stores and countries where the product is for sale, list of ingredients and possible traces (for allergies, food bans or any specific diet), food additives detected from the latter, and nutritional information.

The French government even promotes its use on official websites and has encouraged the creation of similar projects in cosmetics, Open Beauty Facts, or animal food, Open Pet Food Facts.

[OpenFoodFacts.data.gouv.fr](https://www.data.gouv.fr/fr/organizations/open-food-facts/?msclkid=d83e9bc2ace511ecb71758e750feb774)

### Installation Instructions

After cloning, run

```
pip install -r requirements.txt
```
and then
```
python download_data.py
```

This will install the necessary packages and create the folders `data/train` and `data/test`. It will also download the images from `data_locations_public.csv` and then create 2 separate `labels.csv` files. You need approximately 2 GB disk space to download every images, and it can take about 2 hours with a decent connection to retrieve every image because we are using a single process. There are 50,000 training images and 5,000 test images of different sizes. Please note that the images are of varying quality and thus the dataset contains "noisy" inputs. Grayscale images should be filtered while training.


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
