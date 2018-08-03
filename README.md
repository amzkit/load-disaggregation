# Load-disaggregation Python Library

The goal of the library is to disaggregate an appliance load from the power meter reading. The library take a load disaggregation models provided in the [NILMTK Documentation](https://github.com/nilmtk/nilmtk/tree/master/docs/manual).

The library is reproduced for those who requires only a load disaggregation function in NILMTK. It eliminates lots of installation and setting up environment step in the original NILMTK. Simply import, pre-process your data and you are good to go.

# Requirement
1. Python 3 (haven't tested in Python 2)
2. Series of load meter data (in Watt)
3. Appliance meter data (in Watt) with the same time line as (2)
4. Test load meter data

# Beginners Guide
1. download desired model file from the repo and place into the same folder with your python file

