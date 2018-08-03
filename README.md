# Load-disaggregation Python Library

The goal of the library is to disaggregate an appliance load from the power meter reading. The library take a load disaggregation models provided in the [NILMTK Documentation](https://github.com/nilmtk/nilmtk/tree/master/docs/manual).

The library is reproduced for those who requires only a load disaggregation function in NILMTK. It eliminates lots of installation and setting up environment step in the original NILMTK. Simply import, pre-process your data and you are good to go.

# Requirement
1. Python 3 (haven't tested in Python 2)
2. A whole house load meter data (in Watt)
  I would recommend at least 1 whole day of data
3. Appliance meter data (in Watt) of the same house (it should be recorded at the same time of whole house load meter data)
  Can be more than 1 appliance type, but the more number of appliance meter data you have, the better accurate result will be.
4. Another house load meter data (in Watt)

# Beginners Guide
1. importing the desired model
  - download desired model file from the repo and place into the same folder with your python file
  - import the desired model in your python file
'''
  import fhmm_model as fhmm
'''
2. preparing your data in to the correct format
  - your load meter data and your appliance data should be stored in the same dataframe
  - the dataframe should include column 'power' which is a series of a whole house load meter data.
  - the dataframe should include a column of one appliance meter data. The column can be named anything you would like. The example 'app1'
  - the dataframe should have a timestamp column. this timestamp is a Unix Epoc timestamp (10 digit) format

'''
  df
        power     app1      app2      app3
  0     23.3      0     0     0
  1     318.4      295.1     0     0
  2     318.7      295.4      0     0
  ...
'''

3. 

