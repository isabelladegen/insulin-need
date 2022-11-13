# Finding temporal patterns in insulin needs for people with Type 1 Diabetes

This project is the source code that was used for the study published in paper:

TODO...

## Structure
```
.
└── insulin-need/
    ├── data (AUTO CREATED AND NEVER CHECKED IN)/
    │   └── perid/
    │       ├── p1
    │       ├── p2
    │       └── ...
    ├── src/
    │   ├── scripts
    │   └── <various py files>
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
1. *the code was run and tested on a mac x86_64 and partially on a mac arm64 (M1 mac). Specifically tslearn did not run natively on the M1 mac* 
2. *the [conda env file](conda.yml) is setup to use the latest possible version of all dependencies under Python 3.9. The exact versions of libraries used for the paper are here: [requirements.txt](/requirements.txt)*

### IDE
The code was developed, tested and run using the PyCharm Professional IDE. 
This documentation assumes that you run the scripts and tests with that IDE. 
This should also work in the Community Edition of PyCharm.

### Configuration

There's only one configuration variable that you need to set which is the location to the folder that contains your copy
of the OpenAPS Commons data still as zip files:

1. Rename the [private-yaml-template.yaml](private-yaml-template.yaml)  to ```private.yaml```
2. Change the ```openAPS_data_path``` property in the file to contain a string to the folder where your copy of the OpenAPS data is

*Note: both the data and the private.yaml file must not be checked into git!*

### Data
To run any of the code you require the OpenAPS Commons dataset. 
You can request access from the [OpenAPS](https://openaps.org/outcomes/data-commons/) website.

You will need to manually setup the data folder and configurations on your machine as the data folder is ignored. 
Remember you MUST NOT CHECK-IN ANY DATA into a public repository or share with anybody!

Most of the code in this project will read a preprocessed version of the original zip file.
To create those versions run the following scripts:

- [src/scripts/write_blood_glucose_df.py](src/scripts/write_blood_glucose_df.py) edit the script and 
- [src/scripts/write_device_status_df_dedubed.py](src/scripts/write_device_status_df_dedubed.py) *recommended*
- [src/scripts/write_device_status_df.py](src/scripts/write_device_status_df.py)

These scripts do some preprocessing: they cleanup the timestamp and set the time to UTC and they drop records with no 
timestamp. 
The files reading the device status only work for n=116 files atm.
It reads the columns configured in the ```device_status_col_type``` property in the 
[configurations.py](/src/configurations.py) file.
Due to it not reading all the columns the file will have duplicated entries, the [](src/scripts/write_device_status_df_dedubed.py)
removes those duplicated entries.

*Note: for the scripts to work you have to have the original data folder configured. 
The scripts will create a ```data```  and ```data/perid``` folder.
Those folders are ignored as they MUST not be checked-in!*

### Tests
To check that everything is working run all the tests in [tests](/tests) if they all pass your environment is correctly setup
and you have all the required data files.

*Note some tests use real data and are ignored anywhere where the data files/path are not available.*




