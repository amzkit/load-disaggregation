# Load-disaggregation Python Library

The goal of the library is to disaggregate an appliance load from the power meter reading. The library take a load disaggregation models provided in the [NILMTK Documentation](https://github.com/nilmtk/nilmtk/tree/master/docs/manual).

The library is reproduced for those who requires only a load disaggregation function in NILMTK. It eliminates lots of installation and setting up environment step in the original NILMTK. Simply import, pre-process your data and you are good to go.

# Requirement
1. Python 3 (haven't tested in Python 2)
2. Series of load meter data (in Watt)
  I would recommend at least 1 whole day of data
3. Appliance meter data (in Watt) with the same time line as (2)
  Can be more than 1 appliance type, but the more number of appliance meter data you have, the better accurate result will be.
4. Test load meter data (in Watt)

# Beginners Guide
1. download desired model file from the repo and place into the same folder with your python file
