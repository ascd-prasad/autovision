Autovision - Car Detection Project
=================================

Traditional way of Car identification relies on manual Human Intelligence.
As intelligent and observant a human could be, it is practically difficult & inefficient to distinguish between the wide variety of vehicle makes and models. It becomes a laborious and time-consuming task for a human observer to monitor and observe the multitude of screens and record the incoming or outgoing makes and models or to even spot the make and model being looked for.

The task of detecting Car can be automated with Artificial Intelligence by leveraging the Convolutional Neural Nets.

Here Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe. 

### Step-by-step walks through the solution

The Cars dataset contains Training and Test images of 196 classes of cars. Captured the below details from Training and Test images:

![train-test](https://github.com/ascd-prasad/autovision/blob/main/static/train-test.jpg)

 
Annotations files contains bounding box details for test images and train images.

![Annotatons](https://github.com/ascd-prasad/autovision/blob/main/static/Annotatons.jpg)

Merging both Dataset and Annotations into one data frame. Final Dataset sample:

![final-dataset](https://github.com/ascd-prasad/autovision/blob/main/static/final-dataset.jpg)

Here is the word cloud represents the high presence of keywords from car names and make file. Dataset has more presence of Sedan, SUV, Coupe, and convertible type models.

![cloud](https://github.com/ascd-prasad/autovision/blob/main/static/cloud.jpg)

Year 2012 is having the most number of car makes from all company models.

![year](https://github.com/ascd-prasad/autovision/blob/main/static/year-bar.jpg)

Sedan is having high presence in our data followed by SUV, convertible, Coupe, and hatchback models. Type-R models are very less in our data.

![type](https://github.com/ascd-prasad/autovision/blob/main/static/car-type.jpg)
