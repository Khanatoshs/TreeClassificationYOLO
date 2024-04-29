# TreeClassificationYOLO
Thesis code for classification of trees with YOLO

In order to run the file it's necessary to prepare the configuration file.



* Config file

First duplicate the config_example.ini file. Then change the name of the copy to config.ini
Inside the config.ini file will be the different paremeters needed to run all the scripts.

Whenever a list is needed it is only necessary to separate the values with the character ,  no need for brackets.

If possible use absolute paths in order to reduce errors

As of Now the config file has sections for the scripts

imageprocess.py

shapecheck.py

data_split.py


More may be added later

* Yaml file

The file example.yaml contains an example on how to create a yaml file to run the yolo.
First it is needed to input the folder where the folders for the images and the lables are located.
Next the relative path to the training images and the path to the validation images.

The next section of the yaml file is the categories that are going to be identified. 
In the example only the categories of larch and cedar are included but they can be changed as needed.
The number must match the one written on the labels text file.