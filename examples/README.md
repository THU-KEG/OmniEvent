# Examples

We provide several event extraction examples in this directory. 

## Important Note

To make sure you run the lastest versions of example scirpts, you need to **install the repository from source** as follows:

```shell
git clone https://github.com/THU-KEG/OmniEvent.git
cd OmniEvent
pip install .
```


## BigModel

The `BigModel` directory contains tuning code for large PLMs. The tuning code is supported by [BMTrain](https://github.com/OpenBMB/BMTrain). 

## ED

The `ED` directory contains examples of event detection. 

## EAE

The `EAE` directory contains examples of event argument extraction. You can conduct EAE independently using golden event triggers or you can use the predictions of `ED` to do event extraction.