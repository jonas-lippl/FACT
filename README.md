# FACT: Federated Adversarial Cross Training
This repository implements FACT as introduced by [[1]](ADDLINK). FACT provides a highly efficient framework for federated multi-source-single-target domain adaptation. It uses the implicit distributional differences between source clients pairs to adapt to the domain of an independent target client.


### Download the data
To run the experiments the respective data has to be downloaded and added to `./data`.
The Digit-Five images can be obtained from [Google Drive](https://drive.google.com/open?id=1A4RJOFj4BJkmliiEL7g9WzNIDUHLxfmm) [2], download and add the `.mat` files to `./data/digit-five`.

To use the Office-Caltech10 [3] dataset clone [this](https://github.com/ChristophRaab/Office_Caltech_DA_Dataset) repository and copy the four image folders into `./data/office_caltech_10`.

The Office [4] dataset can be obtained from [Google Drive](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA). Download the files and copy the three folders into `./data/office`.

## Setup
We recommend setting up PyTorch with cuda if you have GPUs available. You can use pip to install the requirements
```
pip install -r requirements.txt
```
or use the provided Dockerfile.
```
docker build --tag fact -f Dockerfile .
```
To reproduce the results from FACT you can use the `generate_experiment_script.py` script to set the model parameters. It will generate a bash script that starts all experiments with the given set of parameters.
You can specify, whether you want to run the experiments on the Digit-Five, the Office-Caltech10 or the Office dataset. 
Also, you can split each domain into multiple clients, exclude a number of domains from the training, set parameters like the number of epochs and rounds of federated learning, adjst the batch size,  the learning rate, the learning rate decay rate, the number of times the experiment is repeated and whether you want to include FACT fine-tuning or not.

An example recreating the multi-source-single-target Digit-Five experiment can be found in `./scripts`. The example requires docker and that you define the source path in your `.bashrc`.
```
export SOURCE_PATH_FACT=/path/to/FACT
```
Alternatively you have to edit `generate_experiment_script.py` to include your local path to FACT and remove the docker commands if you wish to run the code in a virtual environment.
Now you can directly run the generated script:
```
bash scripts/experiment.sh
```
This will run `FACT.py` for all possible combinations of the parameters you specified in the `generate_experiment_script.py`. The results of each experiment will be saved in `./saved_models`.
To display the final results you can run
```
bash scripts/experiment_evaluate.sh
```



## References
[1] FACT: Federated Adversarial Cross Training

[2] Peng, Xingchao, et al. "Federated adversarial domain adaptation." arXiv preprint arXiv:1911.02054 (2019).

[3] Gong, Boqing, et al. "Geodesic flow kernel for unsupervised domain adaptation." 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012.

[4] Saenko, Kate, et al. "Adapting visual category models to new domains." Computer Vision–ECCV 2010: 11th European Conference on Computer Vision, Heraklion, Crete, Greece, September 5-11, 2010, Proceedings, Part IV 11. Springer Berlin Heidelberg, 2010.
