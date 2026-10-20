
## Project Structure

```text
architectures/CNNs/    Base network implementations
architectures/SDNs/    Multi-exit networks with internal classifiers
config.py              Command-line arguments
config/                Existing experiment configuration scripts
data.py                Datasets and preprocessing
train_networks.py      *Model training entry point
asr_test.py            Related metric evaluation
latency_test.py        End-to-end experimental metrics
network_architectures.py  Model creation, loading, and saving
model_funcs.py         Training and evaluation functions
results/               Experiment results
```




## Data Preparation


- CIFAR-10: The code uses `./data`; when data is missing, it attempts to download it through torchvision.
- VGGFace: The dataset CSV files are in `./data/vggface/MAAD/`, and the original images are in `./data/vggface/vggface2_top50`.

Specific data formats and preprocessing methods are subject to the dataset implementations in `data.py`.

## Running Experiments

```bash
pip install -r requirements.txt
```
`config/` provides existing experiment scripts, such as the training and evaluation process for CIFAR-10 Gray:

```bash
bash config/att1CIFAR.sh
```

This script runs multiple groups of training and evaluation sequentially, uses GPU 0 by default, and may take a long time.

The entry points can also be called separately. Note that consistent parameter settings are required here:

```bash
python train_networks.py
python asr_test.py
python latency_test.py 
```

Main parameters include:

- `--dataset`: Dataset names, including cifar10G, cifar10GN, cifar10SN, VGGfaceMW, VGGfaceMY, VGGfaceMBH, etc.; datasets only.
- `--model`: Model types, including `vgg`, `resnet`, and `mobilenet`.
- `--add_ic_attack`: Internal classifier position configuration separated by spaces; its length must match the model architecture.
- `--attr_ic`: Attribute configuration for each internal classifier. `-1` indicates an internal classifier before the attack position, `-2` indicates an internal classifier after the attack position, and a value greater than or equal to `0` indicates an attacked position, with the value itself being the attack attribute. Its length must match the valid exit configuration. Refer to the settings in `config/att1CIFAR.sh`.
- `--gpuid`, `--seed`: GPU ID and random seed.
- `--thres`, `--save_path`: Evaluation threshold and result JSON path.

