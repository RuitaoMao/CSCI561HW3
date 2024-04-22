# CSCI561HW3
A project with neural network using backprop built from scratch (without layer libraries such as tensorflow) to predict NY housing market. And the training time limit is only 5 mins with only one kernel (multithreading is not allowed).

**Basic description:**

The original data file contains 16 features about certain housing propertys. Using these features, the model should be trained to predict the number of bedrooms in each property. The label data file contains the number of bedrooms for each property.

The features' data structures includes integers (such as price, propertysqft) and strings (such as type, brokertitle, address).

After exploring the 16 features, it's found that three integer value features: "price", "propertysqft", and "bath" are strongly related to the outcome. Although "longtitude" and "latitude" are also integer type, they does not provide a strong relationship to the outcome without importing some GIS libraries.
As for the String types features, it is found that "type" and "sublocality" are some features relatively important to the outcome. Therefore, this project designed some mapping structures to normalize these string values and used Onehot technique to add these features into the input nodes.


