# Finding temporal patterns in insulin needs for people with Type 1 Diabetes

This project is the source code for the study published in the [paper](https://doi.org/10.48550/arxiv.2211.07393) below. Please cite as following in your work:

BibTeX:
```
@article{Degen2022,
   author = {Isabella Degen and Zahraa S. Abdallah},
   doi = {10.48550/arxiv.2211.07393},
   month = {11},
   title = {Temporal patterns in insulin needs for Type 1 diabetes},
   url = {https://arxiv.org/abs/2211.07393v1},
   year = {2022},
}
```
Formatted citation:
<div class="csl-entry">Degen, I., &#38; Abdallah, Z. S. (2022). <i>Temporal patterns in insulin needs for Type 1 diabetes</i>. https://doi.org/10.48550/arxiv.2211.07393</div>

## Structure
```
.
└── insulin-need/
    ├── data (AUTO CREATED. DO NOT CHECK IN!)/
    │   └── perid/
    │       ├── p1
    │       ├── p2
    │       └── ...
    ├── src/
    │   ├── scripts
    │   └── <various py files>
    ├── examples/ -> contains example Jupyter notebooks that show how to use the code
    ├── tests
    ├── README.md
    ├── conda.yml -> use to create Python env
    ├── requirements.txt -> refer to if you have version problems
    └── private-yaml-template.yaml -> local config - see instructions
```

## Getting started

### Python Environment

Conda was used as Python environment. You can use the following commands to setup a conda env with all the required dependencies:

1. Create conda env ```conda env create -f conda.yml``` and ```conda activate tmp-22```
2. Update conda env ```conda env update -n tmp-22 --file conda.yml --prune```

*Notes:*
1. *the code was run and tested on a mac x86_64 and partially on a mac arm64 (M1 mac). ```tslearn``` has no native M1 package yet, the conda file has instructions on how to install ```tslearn``` on an M1 mac* 
2. *the [conda env file](conda.yml) is setup to use the latest possible version of all dependencies under Python 3.9. The exact versions of libraries used for the paper are here: for os-64 Mac [requirements.txt](/requirements.txt) & for M1 Mac[requirements-m1.txt](/requirements-m1.txt)*

### IDE
The code was developed, tested and run using the PyCharm Professional IDE. 
This documentation assumes that you run the scripts and tests with that IDE. 
This should work in the free Community Edition of PyCharm.

### Configuration

You need to configure where your downloaded OpenAPS Data Commons dataset zip files are:

1. Rename the [private-yaml-template.yaml](private-yaml-template.yaml)  to ```private.yaml```
2. Change the ```openAPS_data_path``` property in the file to the string of the absolute path to the folder where your copy of the OpenAPS data is

**Note: do not check the data into git!**

### Data
To run most of the code you require the OpenAPS Commons dataset. 
You can request access from the [OpenAPS](https://openaps.org/outcomes/data-commons/) website.

Once you have the data and have provided the path in the ```private.yaml``` configuration file 
you're ready to generate the preprocessed versions of the original zip file.

You have two options. Most code assumes you have the files for 1 created:
1. leave the ```flat_file: FALSE``` in the ```private.yaml``` file and a preprocessed version per id gets created in a folder called ```data/perid```
2. change the ```flat_file: TRUE``` in the ```private.yaml``` file and a preprocessed version containing all id gets created

To create the preprocessed data files run the following scripts depending on what you need:

- [src/scripts/write_blood_glucose_df.py](src/scripts/write_blood_glucose_df.py) -> creates ```bg_df.csv``` files
- [src/scripts/write_device_status_df_dedubed.py](src/scripts/write_device_status_df_dedubed.py) *recommended* -> creates ```device_status_dedubed.csv```
- [src/scripts/write_device_status_df.py](src/scripts/write_device_status_df.py)

These scripts do some preprocessing: they transform the timestamps to uniform UTC timestamps and drop records with no 
timestamp.   
The files reading the device status only work for the n=116 files from the OpenAPS system (not Loop or AndroidAPS, just yet).
It reads the columns configured in the ```device_status_col_type``` property in the 
[configurations.py](/src/configurations.py) file.
Given not all columns are read there will be duplicated entries.[write_device_status_df_dedubed.py](src/scripts/write_device_status_df_dedubed.py)
removes those duplicated entries!

### Tests
Once you have the data generated you can run the [tests](/tests) to check that everything is working as it should be. If they all pass your environment is correctly setup and you have all the data that you need for the methods.

*Note some tests use real data and are ignored anywhere where the data files/path are not available. And some tests are by default ignored because they take a really long time to run. You can run them manually as needed.*


## Examples

There are [example notebooks](examples/) available that show how to use the code to create heatmaps, k-means clustering and the matrix profile.
They include many examples on how to read the data files and resample the data to your needs.





