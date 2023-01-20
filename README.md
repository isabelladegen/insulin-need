# Finding temporal patterns in insulin needs for people with Type 1 Diabetes

This project preprocesses the OpenAPS Data Commons dataset into .csv files that are time series friendly.
At the moment the code is only preprocessing OpenAPS data. The OpenAPS Data Commons also includes Loop and AndroidAPS data.

It implements the following pattern finding techniques on this dataset:
* Statistics - see example [Confidence Intervals, Violin Plots, Box plots](examples/Statistics.ipynb) & [Heatmap Notebook](examples/Heatmap.ipynb)
* K-means clustering - see example [K-means Notebook](/examples/K-means%20clustering.ipynb)
* Matrix Profile - see example [Matrix Profile Notebook](examples/Matrix%20Profile.ipynb)
* Agglomerative Clustering - see example [Agglomerative Clustering Notebook](examples/Agglomerative%20clustering.ipynb)

If you want to work with the OpenAPS Data Commons you need to apply and get a copy of that data set. See [OpenAPS Data Commons](https://openaps.org/outcomes/data-commons/) on how to do that.
You can then use the preprocessing scripts in this project to safe you a lot of work.

This project is the source code for the study published in [paper](https://doi.org/10.48550/arxiv.2211.07393).
This code is being regularly updated and the version used for the paper was [Tag NeurIPS22_ts4h](https://github.com/isabelladegen/insulin-need/releases/tag/neurips22_ts4h)
If you use any of this code please cite this paper as following in your work:

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

## Project Structure
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
2. If you add new dependencies to the [conda.yml](conda.yml) file you can update the env ```conda env update -n tmp-22 --file conda.yml --prune```

*Notes:*
1. *The code was run and tested on a Mac x86_64 and on a Mac arm64 (M1 mac)* 
2. *The [conda.yml](conda.yml) file uses the latest versions of all dependencies, fixing only Python to 3.9 and pandas to 1.4. 
If the test don't run for you, and you think it's dependency related, you can compare your versions ```conda list``` with the versions originally used [requirements.txt (osx_64)](/requirements.txt) or [requirements-m1.txt (osx_arm64)](/requirements-m1.txt)*

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

1. Write preprocessed irregular raw OpenAPS data files [src/scripts/write_processed_device_status_file.py](src/scripts/write_processed_device_status_file.py) creates per id or for all ids ```irregular_iob_cob_bg.csv``` of IOB, COB and BG data
2. Write preprocessed and hourly & daily down-sampled OpenAPS data files [src/scripts/write_processed_device_status_file.py](src/scripts/write_processed_device_status_file.py) creates per id or for all ids ```hourly_iob_cob_bg.csv``` and ```daily_iob_cob_bg.csv```.
   IOB, COB and BG are aggregated using mean, max, min and std.
3. CGM Data for all systems [src/scripts/write_blood_glucose_df.py](src/scripts/write_blood_glucose_df.py): creates ```bg_df.csv``` per id or for all ids

The output of script 1 and 2 are prerequisites for all methods in this project.
The preprocessing done in script 1. transforms the timestamps to uniform UTC timestamps, drops records with no 
timestamp and removes duplicates. The scripts at the moment only read the OpenAPS system files (not Loop or AndroidAPS, just yet).

Older scripts:
- [src/scripts/write_device_status_df_dedubed.py](src/scripts/write_device_status_df_dedubed.py) crates per id or for all ids ```device_status_dedubed.csv```
- [src/scripts/write_device_status_df.py](src/scripts/write_device_status_df.py)

These scripts read the columns configured in the ```device_status_col_type``` property in the 
[configurations.py](/src/configurations.py) file of the OpenAPS system. Given not all columns are read there are be duplicated entries.[write_device_status_df_dedubed.py](src/scripts/write_device_status_df_dedubed.py)
removes those duplicated entries!


### Tests
Once you have the data generated with the scripts above you should be able to successfully run the [tests](/tests). 
If they pass your environment is correctly setup and you have all the data that you need for the methods.

*Note some tests use real data and are automatically ignored anywhere where the data files/path are not available.
Pay attention as the methods don't work without the proper files!
Some tests are by default ignored because they take a really long time to run. You can run them manually if needed.*


## Examples

There are [example notebooks](examples/) available that show how to use the code.
They include many examples on how to read the data files and how you can use the differently sampled files shaped into
different length time series.





