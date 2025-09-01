# Code for GPASS

Configs
===
All configs are located `configs` folder. Before running the code, please make sure you have installed all the required packages. You can install them by running `pip install -r requirements.txt`.
For dataset config, please refer to `configs/dataset`, where you will need `[dataname].yaml` for a specific dataset.
For model config, please refer to `configs/model`.
For run config, please refer to `configs/run`.

Data Preprocessing
===
Before processing the slides, we suggest to arrange the slides in the following structure:
```
data/
    ├── source/
    │   ├── slide_name_1.svs
    │   ├── slide_name_2.svs
    │   └── ...
    └── target/
        ├── slide_name_1.svs
        ├── slide_name_2.svs
        └── ...
```
Data preprocessing was done following the steps in [CLAM](https://github.com/mahmoodlab/CLAM). This will generate the processed slide files in .h5 format which contains the patch coordinates.
An example of the processed slide file structure is as follows:
```
processed_slide/
    ├── source/
    |   |── patches/
    |   |   ├── slide_name_1.h5
    |   |   ├── slide_name_2.h5
    |   |   └── ...
    └── target/
        |── patches/
        |   ├── slide_name_1.h5
        |   ├── slide_name_2.h5
        |   └── ...
```

Training
====
We provide the training code for GPASS in `run.sh`. Training the model is as simple as running `./run.sh`. Running will start, and be default the model will be trained for 100 epochs. At the end of each epoch, the model will be saved to the `checkpoints_dir` specified in the config file along with some generated images for visualization.

Testing
====
For stain normalisation testing, simple change the `run` config to `test` and run `./run.sh`. Testing will be based on the model saved in the `checkpoints_dir` specified in the config file and you may specify the `which_epoch` to load the specific epoch. The tested results will be saved to a log file in the `logs` folder.

For classification testing, please refer to [IDH_Classification](https://github.com/thomascong121/IDH_Classification) where the classification code is located.

Comments
====
Most of the codes are based on [CAGAN](https://github.com/thomascong121/CAGAN_Stain_Norm).