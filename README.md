# Quasar Classifier

A classifier which is able to recognize quasar objects from variable transients.

## Dataset Download ##
  For SDSS Stripe82 quasar-targated dataset, the data repository is here: https://www.kaggle.com/sherrysheng97/sdss-stripe82-quasar-targeted-dataset


  For PLASTiCC dataset, the data repository is here: https://www.kaggle.com/c/PLAsTiCC-2018


## Packages version  

* Tensorflow 2.1.0
* numpy 1.17.2
* pandas 0.25.1
* feets 0.4
* glob 1.2.0
* sklearn 0.23.2

## Quick run on already built classifier

  Jump to the Train your classifier part, adjust the `config.txt file`, and run the `train.py`.

## Run on Kaggle 
  Kaggle notebook is strongly suggested to run all codes: https://www.kaggle.com/sherrysheng97/quasar-classifier-sdss-plasticc

  All training and test data are provided. You just need to modify the configuration settings, and then click 'run all'. 

## Build your own classifier

<!-- ### Match objects' classes with three catalogs(DR14Q, Milliquas, Standard Star Catalog)

In `data/source/`folder, 
`DR14Q/DR14Q_v4_4.fits` is the Sloan Digital Sky Survey Quasar Catalog fourteenth data release; 
`heasarc_milliquas.tdat` is the Million Quasars Catalog with version 6.5 (14 June 2020); 
`stripe82calibStars_v2.6.dat` is the SDSS Stripe 82 Standard Star Catalog with version 1.1 (8 Mar 2007).
run `full_match.py` to match all objects in `Stripe82_DR16_QuasarTargets_unique.csv` and `full_matched_objects.csv` is generated.
```bash
python full_match.py
```
### Convert to json files

In `data/` folder,
you could convert your data format(csv,txt,dat,fit,...) to json file for convenience. The `convert_json.py` is used to convert the original csv file with the combination of all objects' data to json files for each object. All json files will be save in `data/quasar_obs/` folder.
```bash
python convert_json.py
```
Json file schema demo:
```json
{
    "objID1": "8658466040442257764",
    "ra": 356.9962,
    "dec": -0.727944,
    "redshift": 1.302255,
    "type": "QSO",
    "obs": {
        "8658466190782693748": {
            "u": {
                "mjd": 52552.303296,
                "psfMag": 22.19604,
                "psfMagErr": 0.3725143
            },
            "g": {
                "mjd": 52552.3049544,
                "psfMag": 22.02514,
                "psfMagErr": 0.12833139999999998
            },
            "r": {
                "mjd": 52552.3016376,
                "psfMag": 21.98736,
                "psfMagErr": 0.13563699999999998
            },
            "i": {
                "mjd": 52552.30246680001,
                "psfMag": 22.44357,
                "psfMagErr": 0.26586859999999995
            },
            "z": {
                "mjd": 52552.3041252,
                "psfMag": 21.27691,
                "psfMagErr": 0.43855600000000006
            }
        },
        "8658476657620419073": {
            "u": {
                "mjd": 53674.20817728,
                "psfMag": 22.590970000000002,
                "psfMagErr": 0.46709849999999997
            },
            "g": {
                "mjd": 53674.20983585,
                "psfMag": 22.09597,
                "psfMagErr": 0.08932047
            },
            "r": {
                "mjd": 53674.2065187,
                "psfMag": 21.93451,
                "psfMagErr": 0.1321836
            },
            "i": {
                "mjd": 53674.20734799,
                "psfMag": 22.01484,
                "psfMagErr": 0.15644639999999999
            },
            "z": {
                "mjd": 53674.20900656,
                "psfMag": 22.06707,
                "psfMagErr": 0.5027332
            }
        }
    }
}
```

### Preprocess the raw data

After generating the json files, we need to combine and clean the data to genetate a proprecessed file for our training input. In `data/preprocess.py`, you could choose the bands(in program it is the features list) to remove the outliers and unvalid values.
```bash
python preprocess.py
```
Processed file will be generated in `processed/`file. `balanced/` and `unbalanced/` folders conrespond to datasets with balanced/unbalanced classes. 
 -->
### Train your classifier

In `train/` folder, `configs.txt` is used for designing the architecture of the classifier. After setting the configurations, run the `train.py` file to train and test the classifier.
```bash
python train.py
```

### Configurations Setting Explaination


<table>
   <tr>
      <td>Config type</td>
      <td>Parameters</td>
      <td>Explaination</td>
      <td>Example</td>
   </tr>
   <tr>
      <td rowspan="10">input config</td>
      <td>train_path</td>
      <td>the path of the test set file</td>
      <td>../data/processed/unbalanced/final_v1.csv</td>
   </tr>
    <tr>
      <td>save_path</td>
      <td>the folder that saves all results</td>
      <td>results</td>
   </tr>
   <tr>
      <td>seed</td>
      <td>the seed for generating random number</td>
      <td>1</td>
   </tr>
   <tr>
      <td>features</td>
      <td>the bands/features used in training.\n All features: g, r, i, z, u, g_error, r_error, i_error, z_error, u_error
      </td>
      <td>g,r,i</td>
   </tr>
   <tr>
      <td>format</td>
      <td>the input format for the training. Three formats are provided: simple, group, season</td>
      <td>group</td>
   </tr>
   <tr>
      <td>processed</td>
      <td>the preprocess method for the input data. Three methods are provided: 
s: standardization; n: normalization; d: difference between neighboring data points
   </td>
      <td>s</td>
   </tr>
   <tr>
      <td>set_GPR</td>
      <td>whether to choose the Gaussian Process Regression. This method will generate a new regressed light curve for each group of light curve of an object</td>
      <td>True</td>
   </tr>
<tr>
      <td>group_size</td>
      <td>the number of days in all groups</td>
      <td>67</td>
   </tr>
<tr>
      <td>group_num</td>
      <td>the number of groups for each object</td>
      <td>7</td>
   </tr>
<tr>
      <td>cut_fraction</td>
      <td>For prediction, the fraction of data drop from each group. This parameter is used for testing the improvement of accuracy with more complete data</td>
      <td>0.1 or empty</td>
   </tr>
   <tr>
      <td rowspan="4">network config</td>
      <td>rnn_type</td>
      <td>the type of RNN layers. Three types are provided: LSTM, GRU, Simple</td>
      <td>LSTM</td>
   </tr>
   <tr>
      <td>hidden_layers</td>
      <td>a list of hiddren layers' neuron numbers.</td>
      <td>[256,256,256,256,256,256]</td>
   </tr>
   <tr>
      <td>dropout</td>
      <td>the fraction of objects dropped before being fed into the next layer to avoid overfitting</td>
      <td>0.25</td>
   </tr>
   <tr>
      <td>plot_model</td>
      <td>whether to plot the model architecture </td>
      <td>True</td>
   </tr>
   <tr>
      <td rowspan="7">train config</td>
      <td>batch_size</td>
      <td>the number of sequences fed into the layer for each time</td>
      <td>32</td>
   </tr>
   <tr>
      <td>num_epochs</td>
      <td>the times for the input data being processed</td>
      <td>10</td>
   </tr>
   <tr>
      <td>test_fraction</td>
      <td>the fraction of test set among all data</td>
      <td>0.2</td>
   </tr>
   <tr>
      <td>optimizer</td>
      <td>the optimization method for the loss function. Two functions are preferred: Adam, SGD</td>
      <td>Adam</td>
   </tr>
   <tr>
      <td>learning_rate</td>
      <td> a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function</td>
      <td>0.001</td>
   </tr>
   <tr>
      <td>decay</td>
      <td>whether the learning rate will decrease with the increasing nunmber of epochs. If true, decay_value = learning_rate/num_epochs</td>
      <td>True</td>
   </tr>
   <tr>
      <td>metrics</td>
      <td>the metrics during training for testing the performance of the classifier</td>
      <td>accuracy,AUC,f1_score</td>
   </tr>
</table>







