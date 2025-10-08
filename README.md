# <NAME> Sequence-Based Anomaly Detection Model
<NAME> is an anomaly detection model for any network with Corelight monitoring configured.
It provides automatic, sequential-based detection of anomalous traffic.


<!-- toc -->

- [ Anomaly Detection Model](#-anomaly-detection-model)
- [How  Works](#how--works)
  - [Protocol Anomalies](#protocol-anomalies)
  - [Sequential Anomalies](#sequential-anomalies)
- [Navigating This Repository](#navigating-this-repository)
  - [Folders](#folders)
  - [Modules](#modules)
    - [Zeek (data.datasets)](#zeek-datadatasets)
    - [Zeek Torch (data.datasets)](#zeek-torch-datadatasets)
    - [ZeekCleaner (data.cleaning)](#zeekcleaner-datacleaning)
    - [LogCleaner (data.log\_cleaners)](#logcleaner-datalog_cleaners)
    - [BatchDBSCAN (models.batchdb)](#batchdbscan-modelsbatchdb)
- [Running the Model](#running-the-model)
  - [Changing Model Sensitivity: Too Many or Too Few Notables](#changing-model-sensitivity-too-many-or-too-few-notables)
- [Known Drawbacks/Concerns](#known-drawbacksconcerns)

<!-- tocstop -->


# How <NAME> Works
The <NAME> model is a machine learning based system for detecting both known and Zero-Day attacks carried out
over a network, <NAME> leveraging transformers to identify host-to-host connections that look
out of the ordinary.

The model comprises of two modular parts. Each is designed to detect anomalies pertaining to different aspects of a
connection: protocol, and sequential. 

## Protocol Anomalies
Communication protocols have strict patterns set by RFC. As such, we can know roughly what to expect for certain types
of connections. For example, HTTP connections have well defined structure. We can use this to understand what "typical"
protocols look like on our networks. We define "protocol anomalies" to be connections that use protocols in an 
unitended or bizarre way. For instance, most DNS requests are UDP connections that are only a few hundred bytes long.
If a user is seen making DNS requests that are dozens of megabytes in size, this could definitely be a sign of 
anomalous DNS usage.

We detect these protocol anomalies by first applying tailored feature engineering to different kinds of Corelight logs.
Every log type has its own cleaner that removes, modifies, and creates features in order to make it have cybersecurity
oriented fields. Once this feature engineering is performed, the data are clustered using BatchDBSCAN to create the 
vocabulary for the sequence model.

BatchDBSCAN is an exponentially quicker version of DBSCAN that is explained in depth in the models readme file. 

All code for the feature engineering and clustering can be found in data/log_cleaners.py and models/batchdb.py.

## Sequential Anomalies
When connections happen, they usually do so in a well known order. For example, if we want to access a file server, we
should try to authenticate with Kerberos before making multiple file requests. We define "sequential anomalies" to be
traffic that seems to be occurring out of order or in a weird combination. 

We detect these sequential anomalies using masked token prediction. We train a transformer to learn what sequences of
vocabulary (created by the structural and protcol clusters) are common in the environment our model was trained in.
During inference, if tokens consistently have a low probability, the overall connection is deemend anomalous.

We track how many anomalies are associated with each host. When a host performs enough malicious activity in a short
time frame, we finally are ready to create a notable.

All code for the transformer can be found in models/<PUT THE PATH HERE>.


# Navigating This Repository
## Folders
**assets:** images used in supporting documents like readmes.

**configs:** any config files used for cleaning objects.

**data:** data utils used in the pipeline.

**docs:** any relevant documentation or demonstrations.

**models:** model files and declaration

## Modules
These classes are all well documented. For more information, visit the file itself and read docstring.

### Zeek (data.datasets)
The Zeek class can be used for reading of Zeek/Corelight output files.

**Current Functionality:**
1. Reads Zeek files from messy directories as log or parquet files
2. Train/test splitting data
3. Log file sorting
4. Targeted IP removal and row filtering
6. More descriptive exceptions and warnings
7. Automatic Zeek -> PyTorch Conversion
8. Saving of data objects to python pickle files for fast reading

### Zeek Torch (data.datasets)
This is not a user facing class. Called under the hood by Zeek to prepare data for PyTorch data loading.

### ZeekCleaner (data.cleaning)
Packages LogCleaners nicely and iterably for Zeek to call. Allows a seamless conversion from Zeek to DataLoader.

### LogCleaner (data.log_cleaners)
*Abstract.* Class outline for individual file cleaners. Log types are handled on a case by case basis.
Example is available via GeneralCleaner, which usually works for most data.

### BatchDBSCAN (models.batchdb)
This is a modification of sklearn's DBSCAN designed to run fast on large amounts of data. The limiting factor on
runtime is essentially batch size, which can be scaled down to massively accelerate training. Also supports inference.
It only uses a representative sample of the cluster to perform inference, so it is possible that clustering can vary
between normal DBSCAN and this modification.


# Running the Model
The two core scripts are train.py and inference.py.

```bash
python train.py --data_path --model_path --cleaner_name --cluster_prefix --vocab_name --transformer_name --aceptance_rate --threshold_name
```

With an optional parameter --seed. Calling --help will explain each of these arguments more in depth. Training
can be done at any regular interval, but we recommend allowing ~500k connections to accumulate before training a model.
This ensures the model learns a good distribution and encounters some rarer traffic to prevent the overdetection of
anomalies. 

We also strongly recommend removing any UIDs associated with known malicious behavior! If analysts find a classify
a series of events as malicious, do NOT train the model on them! This model relies on the assumption that all data in
the training set are not malicious.

After training, inference can be performed.

```bash
python inference.py
```

This has the exact same command line arguments as train.py.

## Changing Model Sensitivity: Too Many or Too Few Notables
A crucial command line argument is acceptance rate, which ultimately controls how often the model will create a notable.
During training, we compute anomaly scores (between 0 and 1, with 1 being most anomalous) for all training samples.
To determine what will be considered anomalous during inference, we decide on a threshold in anomaly score that, if
a connection is above, will be deemed an anomaly. To decide on this threshold, we use acceptance rate to find what 
anomaly score only deems "acceptance rate" proportion of the data as anomalous in training.

For example, if acceptance rate is set to 0.001, then the anomaly threshold will be set to whatever makes 0.1% of the
training data classified as anomalous. During inference, this threshold will remain the same. To ensure we do not
create notables for every rare connection, we only create anomalies if we see the same host routinely exhibiting
anomalous behavior during one stretch of time.

# Known Drawbacks/Concerns
We acknowledge several drawbacks and future improvements that will need to be made to our system.

1. Device vocabulary building and the sequence modeling assumes a unique host is a unique IP.
   - IPs can change. And with this method, assuming you are leased a new IP frequently on a network, malicious clients
will be a challenge to detect.
1. Log embeddings will need to be adapted over time and can use lots of improvement.
   - Certain features from the logs, such as SSL's odd_version, will need to be updated in the future as things change.
2. Only protocols that have a UID are supported. Most OT protocols (modbus, DNP3, etc) do not have a uid, and we 
thus cannot run them through our system.
1. At inference time, because of IP address concerns, we deem the lifetime of a device to be one inference batch. 
   - If an actor carries out an attack over an extended period of time (say a few weeks), we will be unable to detect it. 
     This is especially relevant to LOTL, where an actor performs near-normal activity that over the span of weeks can 
     aggregate to cause problems.
1. We had highly limited testing data and have not tested it on big data
   - In turn, we cannot be certain how this will perform on big data in the wild
1. We need to narrow down detections
   - Because we aren't targeting anything specific, we make too many detections
   - The proper format for this model would be to look at one protocol (i.e. DCE/RPC) and use the raw packet captures
     to generate far more features than Zeek could (or view it as unstructured) and have separate models for each 
     protocol. This would massively narrow down the number of notables
1. Ultimately, regardless of the 'acceptance rate' in training, it's tough to control the number of notables in production
