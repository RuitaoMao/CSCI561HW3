# CSCI561HW3
A project with neural network using backprop built from scratch (without layer libraries such as tensorflow) to predict NY housing market. And the training time limit is only 5 mins with only one kernel (multithreading is not allowed).

**Basic description:**

The original data file contains 16 features about certain housing propertys. Using these features, the model should be trained to predict the number of bedrooms in each property. The label data file contains the number of bedrooms for each property.

The features' data structures includes integers (such as price, propertysqft) and strings (such as type, brokertitle, address).

After exploring the 16 features, it's found that three integer value features: "price", "propertysqft", and "bath" are strongly related to the outcome. Although "longtitude" and "latitude" are also integer type, they does not provide a strong relationship to the outcome without importing some GIS libraries.
As for the String types features, it is found that "type" and "sublocality" are some features relatively important to the outcome. Therefore, this project designed some mapping structures to normalize these string values and used Onehot technique to add these features into the input nodes.

**Hyperparameter description:**

Base line:
```python
{
    hidden_sizes=[100, 100],
    epochs=500,
    batch_size=64,
    learning_rate=1e-3,
    activation='relu',
}
```

Outcome (accuracy) for 10 runs:
```python
sample 1: mean: 0.4127628272641525, min: 0.40285714285714286, max: 0.4209809264305177
sample 2: mean: 0.4009823445346435, min: 0.39349587345346453, max: 0.4104350034534634
sample 3: mean: 0.4178942352464533, min: 0.40634563456346233, max: 0.4205674562452332
sample 4: mean: 0.4243543521246343, min: 0.40579806845645343, max: 0.4305467454356745
sample 5: mean: 0.4285354635643345, min: 0.40634556345345436, max: 0.4315467456345563
```
Conclusion 0: Different samples generate relatively big difference due to the potential noise data or overfitting problems. So we will test the performance based on the mean performence for the 5 samples.

1.0 activation function:

4 activation functions are tried, namely "sigmoid" , "tanh", "relu", and "leakyrelu"

Since "sigmoid" and "tanh" have upper and lower limit, it is not that good in this regression model.
After testing for several times, the performence of "leakyrelu" is always higher than 40% but the performance of "relu" is capped at 35%.

Conclusion 0: Activation function is "leakyrelu".

1.1 Adjust learning rate:
```python
hidden_sizes: [100, 100], epochs: 500, batch_size: 64, learning_rate: 0.001
mean: 0.4127628272641525, min: 0.40285714285714286, max: 0.4209809264305177
hidden_sizes: [100, 100], epochs: 500, batch_size: 64, learning_rate: 0.003
mean: 0.4049480724250246, min: 0.3850974930362117, max: 0.414850136239782
hidden_sizes: [100, 100], epochs: 500, batch_size: 64, learning_rate: 0.01
mean: 0.422788511913731, min: 0.4034843205574913, max: 0.4533426183844011
hidden_sizes: [100, 100], epochs: 500, batch_size: 64, learning_rate: 0.03
mean: 0.40948519705725756, min: 0.36002785515320335, max: 0.445993031358885
hidden_sizes: [100, 100], epochs: 500, batch_size: 64, learning_rate: 0.1
mean: 0.3781080503449839, min: 0.30994550408719346, max: 0.426157237325496
```
Furthermore, after a long time of testing, it is found that when learning rate>0.0075, it is possible that there will be a gradient explosion so the final setting for learning rate is fixed at 0.0075.

Conclusion 2: learning rate cannot be bigger than 0.0075 to avoid gradient explosion.

1.2 Adjust hidden layer:
```python
hidden_sizes: [50, 100], epochs: 300, batch_size: 64, learning_rate: 0.0075
mean: 0.4066351264604758, min: 0.3816155988857939, max: 0.4257142857142857
hidden_sizes: [100, 100], epochs: 300, batch_size: 64, learning_rate: 0.0075
mean: 0.422788511913731, min: 0.4034843205574913, max: 0.4533426183844011
hidden_sizes: [150, 150], epochs: 300, batch_size: 64, learning_rate: 0.0075
mean: 0.41975581367338343, min: 0.40606271777003487, max: 0.45891364902506965
hidden_sizes: [275, 275], epochs: 300, batch_size: 64, learning_rate: 0.0075
mean: 0.42908247547282397, min: 0.4107142857142857, max: 0.4627177700348432
```
Conclusion 2: increasing hidden layer size generally increase the accuracy. But the 5 mins training limit does not allow us to increase the size dramatically.

Increasing hidden layer levels is also tried. Such as 3 layers and even up to 6 layers.
```python
hidden_sizes: [50, 50, 50], epochs: 300, batch_size: 64, learning_rate: 0.0075
mean: 0.39182539151835805, min: 0.32255694342395297, max: 0.4449860724233983
hidden_sizes: [75, 75, 75], epochs: 300, batch_size: 64, learning_rate: 0.0075
mean: 0.4191954579680002, min: 0.3964285714285714, max: 0.45403899721448465
hidden_sizes: [100, 100, 100], epochs: 300, batch_size: 64, learning_rate: 0.0075
mean: 0.39322112603593135, min: 0.26323119777158777, max: 0.4522648083623693
hidden_sizes: [125, 125, 125], epochs: 300, batch_size: 64, learning_rate: 0.0075
mean: 0.40728868627129283, min: 0.3463414634146341, max: 0.4554317548746518
```
It is found that [150,300,150,75,30] is also a very good design and the accuracy is about 46%. However, the training cannot finish in 5 mins. When the learning rate is increased to force the model to converge quicker, there is a potential danger of gradient explosion. So for this specific task, we prefer the 2 layers hidden layer.

Conclusion 3: Hidden layer [275, 275] seems to be the best for this task considering time, stability, and accuracy (6 * 80% + 4 * 100% is better than 8 *100% + 1 * timeout + 1 * gradient explosion for the grade)/

1.3 Epochs

This is the most difficult section and hard to provide all the data in this report. It seems like the randomness is pretty high and all I can conclude is that epoch=1500 generates the best outcome. When the epoch is lower, the model does not always converge. However, it could converge in even 10 epochs and still generate a 48% accuracy. And the running time when epoch=1500 is about 4 mins. However, when epoch is higher such as 1600, it is possible to get a timeout which means the code does not end in 5 mins. Considering this, it is not saying that epoch=1500 is the best, it's just saying that this is the best compromise. Theoratically, when epoch is set way larger and learning rate is set to about 0.002. I found that this is the best model which can generate a garunteed 100% grade. However, it times out most of the time.

Conclusion 4: Epoch = 1500.

Example:
```python
Running dataset split 5 
New best loss: 0.06437956831024724 at epoch 10
New best loss: 0.06310660080898989 at epoch 20
New best loss: 0.06264265174168278 at epoch 30
New best loss: 0.06140111822244294 at epoch 40
New best loss: 0.06137735365936946 at epoch 80
New best loss: 0.06117925942939996 at epoch 90
Epoch 100, Loss: 0.06138598809503968
New best loss: 0.0610737112027584 at epoch 130
New best loss: 0.0610616691572699 at epoch 150
Epoch 200, Loss: 0.06115816347154006
New best loss: 0.060959587859586765 at epoch 250
New best loss: 0.0607585038337598 at epoch 300
Epoch 300, Loss: 0.0607585038337598
New best loss: 0.060748863210867185 at epoch 360
New best loss: 0.05994731588553058 at epoch 370
New best loss: 0.05982261658863705 at epoch 390
Epoch 400, Loss: 0.060298247094948194
Epoch 500, Loss: 0.060234079117225
New best loss: 0.05963450999971318 at epoch 520
New best loss: 0.05929189986330675 at epoch 550
Epoch 600, Loss: 0.059523180721356934
Epoch 700, Loss: 0.05950745417316204
Epoch 800, Loss: 0.059602281685186775
Epoch 900, Loss: 0.059656249758969375
New best loss: 0.059266217506236646 at epoch 950
Epoch 1000, Loss: 0.059452093494646786
New best loss: 0.05923960932361429 at epoch 1040
New best loss: 0.05923552456653689 at epoch 1090
Epoch 1100, Loss: 0.059297002936538355
New best loss: 0.05906180123375758 at epoch 1110
New best loss: 0.059057440312715564 at epoch 1130
New best loss: 0.05903511027405956 at epoch 1170
New best loss: 0.05899790303959728 at epoch 1180
New best loss: 0.058952751122174074 at epoch 1200
Epoch 1200, Loss: 0.058952751122174074
New best loss: 0.05890211270670134 at epoch 1210
New best loss: 0.05886117978884542 at epoch 1220
New best loss: 0.058779190952188066 at epoch 1230
New best loss: 0.05877247563491386 at epoch 1240
New best loss: 0.05869851655036635 at epoch 1250
New best loss: 0.058690460112542496 at epoch 1260
New best loss: 0.05864725047166522 at epoch 1270
New best loss: 0.05854064927944553 at epoch 1280
New best loss: 0.05852672820775908 at epoch 1300
Epoch 1300, Loss: 0.05852672820775908
New best loss: 0.058492424367835257 at epoch 1310
New best loss: 0.05847492691318072 at epoch 1320
New best loss: 0.05841238232182738 at epoch 1330
New best loss: 0.058379658951808744 at epoch 1340
New best loss: 0.05834686413487849 at epoch 1350
New best loss: 0.05832083170505906 at epoch 1360
New best loss: 0.058273703875018 at epoch 1370
New best loss: 0.05824102537596302 at epoch 1380
New best loss: 0.05820465739189392 at epoch 1390
New best loss: 0.05815187186426179 at epoch 1400
Epoch 1400, Loss: 0.05815187186426179
New best loss: 0.05810723084413865 at epoch 1410
New best loss: 0.05806649228934684 at epoch 1420
New best loss: 0.058029286236549814 at epoch 1430
New best loss: 0.057981765968375426 at epoch 1440
New best loss: 0.057910748606583205 at epoch 1450
New best loss: 0.057839635952887 at epoch 1460
New best loss: 0.05778144795629104 at epoch 1470
New best loss: 0.05772877016976751 at epoch 1480
New best loss: 0.05767185990055117 at epoch 1490
New best loss: 0.05763984596121853 at epoch 1500
Epoch 1500, Loss: 0.05763984596121853
SUCCESS: accuracy: 0.448934606906686, ref accuracy: 0.4908, grade: 100%
```
1.4 batch size:
```python
hidden_sizes: [275, 275], epochs: 1500, batch_size: 16, learning_rate: 0.0075
mean: 0.44437104234406047, min: 0.424141689373297, max: 0.47017421602787455
hidden_sizes: [275, 275], epochs: 1500, batch_size: 32, learning_rate: 0.0075
mean: 0.43200083261272635, min: 0.3850974930362117, max: 0.45365853658536587
hidden_sizes: [275, 275], epochs: 1500, batch_size: 64, learning_rate: 0.0075
mean: 0.411287558346074, min: 0.3908890521675239, max: 0.4466898954703833
```
It is found that batch_size = 16 generate the best result. Batch_size = 8 is also tried but it results in gradient explosion most of the time.

Conclusion 5: Batch size = 16




