# Load-disaggregation Python Library

The goal of the library is to disaggregate an appliance load from the power meter reading. The library take a load disaggregation models provided in the [NILMTK Documentation](https://github.com/nilmtk/nilmtk/tree/master/docs/manual).

The library is reproduced for those who requires only a load disaggregation function in NILMTK. It eliminates lots of installation and setting up environment step in the original NILMTK. Simply import, pre-process your data and you are good to go.

This library is reproduced for beginners use. I was new to NILMTK and python environment. With a limited document on NILMTK for a newbie, I spent lots of time to understand the toolkit. I hope this library will save time for any other beginner who is taking the same path as I do.

# Requirements
1. Python 3.6 (haven't tested in Python 2)
2. **[training house main meter data]** A whole house main meter data (in Watt)
  I would recommend at least 1 whole day of data
3. **[training appliance meter data]** Appliance meter data (in Watt) of the same house (it should be recorded at the same time of whole house load meter data)
  Can be more than 1 appliance type, but the more number of appliance meter data you have, the better accurate result will be.
4. **[testing house main meter data]** Another house main meter data (in Watt)

# Beginner Guide
#### 1. importing the desired model
  - download desired model file from the repo and place into the same folder with your python file
  - import the desired model in your python file
  
```python
  import .fhmm_model as fhmm
```

#### 2. preparing your training data in the correct format
  - **[training house main meter data]** and **[training appliance meter data]** should be stored in the same dataframe (has to be in the same second, you will need to preprocess your data for this in case your data is recorded at different seconds)
  - the dataframe should include column 'power' which is a series of a whole house load meter data.
  - the dataframe should include a column of one appliance meter data. The column can be named anything you would like. The example 'app1'
  - the dataframe should have a timestamp column. this timestamp is a Unix Epoc timestamp format (10 digit)
  - the training dataframe should be similar to below dataframe
```python
  print(df)
        timestamp   power     app1     app2	app3
  0     1525689485  23.3      0        0	0
  1     1525689490  318.4     295.1    0	0
  2     1525689495  318.7     295.4    0	0
  ...
```


#### 3. training your model with training house data
  - call function train()
  - arguments of train() are
    - dataframe from previous step (step 2) and a list of appliance
  - a list of appliance can be a single appliance to train or more

```python
   list_of_appliance = ['app1', 'app2', 'app3']
   fhmm = FHMM()
   fhmm.train(df, list_of_appliance)
```
  - after training you will get a model which is ready to be tested
  - you can save your model in order to use it next time without training the same data again and again

```python
   fhmm.save("fhmm_trained_model")
```

  - you will get a file containing model you have trained with the extension of pkl (from pickle)
  - you can load this mode to use later from function fhmm.load() to skip the step 2, and 3
  
  
#### 4. preparing your testing data in the correct format  
  - testing data should be stored in a dataframe containing timestamp and power columns
  - the testing dataframe should be similar to below dataframe
  
```python
  print(test_df)
        timestamp   power
  0     1525689485  27.5
  1     1525689490  27.7
  2     1525689495  328.2
  ...
```
 
 
#### 5. disaggregating the testing house data with the model
  - call function disaggregate()
  - disaggregate() requires dataframe from step 4 
  - timestamp column should be the same Unix Epoc timestamp format (10 digit)
  - the output of disaggregate() will be a dataframe containing a result of the disaggregate data. each column contains each appliance trained in your model
  
```python
   prediction = fhmm.disaggregate(df)
```
  - try prediction.plot() for a quick graph plot
  - Done! then you can use the data from prediction to do whatever you would like to
