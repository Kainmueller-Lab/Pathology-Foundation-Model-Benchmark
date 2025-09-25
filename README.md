# Pathology-Foundation-Model-Benchmark
Benchmarking the ability of pathology foundation models to do cell type classification

Assumes a .env file with the following variables:
- `HF_TOKEN`: Huggingface API token to download models with access restrictions

## Installation
install the environment with:
`conda env create -f env.yaml`

## Data 
The data can be downloaded via huggingface:
https://huggingface.co/datasets/Kainmueller-Lab/phenobench

## Running the Benchmark
Once you have downloaded the dataset, and installed the required packages, the code can be run with the following command:

### Slurm
`sbatch launch_train_slurm.sh <pathg_to_config> <model_name>`

### Grid Engine
`qsub launch_train.sh <pathg_to_config> <model_name>`

Example:
`sbatch launch_train_slurm.sh configs/dpt_config.yaml uni2`


## Cell type classes
The cell type classes are merged as follows:

| Cell Type                              | Category                 |
|----------------------------------------|--------------------------|
| background                             | Background               |
| tumor cells                            | Tumor cells              |
| B cells                                | B cells                  |
| CD11b+ monocytes                       | Macrophages/Monocytes    |
| CD11b+CD68+ macrophages                | Macrophages/Monocytes    |
| CD3+ T cells                           | T cells                  |
| CD4+ T cells                           | T cells                  |
| CD8+ T cells                           | T cells                  |
| CD4+ T cells CD45RO+                   | T cells                  |
| CD4+ T cells GATA3+                    | T cells                  |
| Tregs                                  | T cells                  |
| CD163+ macrophages                     | Macrophages/Monocytes    |
| CD68+ macrophages                      | Macrophages/Monocytes    |
| CD68+ macrophages GzmB+                | Macrophages/Monocytes    |
| CD68+CD163+ macrophages                | Macrophages/Monocytes    |
| CD11c+ DCs                             | Dendritic cells          |
| NK cells                               | NK cells                 |
| adipocytes                             | Adipocytes               |
| granulocytes                           | Granulocytes             |
| nerves                                 | Nerves                   |
| plasma cells                           | Plasma cells             |
| smooth muscle                          | Smooth muscle            |
| stroma                                 | Stroma                   |
| vasculature                            | Vasculature/Lymphatics   |
| lymphatics                             | Vasculature/Lymphatics   |
| tumor cells / immune cells             | Other cells              |
| undefined                              | Other cells              |
| immune cells                           | Other cells              |
| immune cells / vasculature             | Other cells              |


## Results
Here are the more detailed result tables:
### PhenoCell dataset:
| Model        | Head        | Score   |
|--------------|-------------|---------|
| HoverNext    | N/A         | 0.238   |
| musk         | SimpleSeg   | 0.200   |
| titan        | SimpleSeg   | 0.232   |
| phikonv2     | SimpleSeg   | 0.243   |
| provgigapath | SimpleSeg   | 0.264   |
| virchow2     | SimpleSeg   | 0.286   |
| uni2         | SimpleSeg   | 0.303   |
| titan        | UNetR       | 0.264   |
| phikonv2     | UNetR       | 0.280   |
| provgigapath | UNetR       | 0.280   |
| uni          | UNetR       | 0.288   |
| virchow2     | UNetR       | 0.295   |
| uni2         | UNetR       | 0.301   |

### Arctique:
| Model         | Head      | Score  |
|---------------|-----------|--------|
| HoverNext     | HoverNext | 0.938  |
| musk          | SimpleSeg | 0.745  |
| phikonv2      | SimpleSeg | 0.746  |
| titan         | SimpleSeg | 0.800  |
| uni           | SimpleSeg | 0.826  |
| uni2          | SimpleSeg | 0.826  |
| provgigapath  | SimpleSeg | 0.830  |
| virchow2      | SimpleSeg | 0.838  |
| titan         | UNetR     | 0.922  |
| phikonv2      | UNetR     | 0.927  |
| provgigapath  | UNetR     | 0.927  |
| musk          | UNetR     | 0.929  |
| uni           | UNetR     | 0.934  |
| virchow2      | UNetR     | 0.943  |
| uni2          | UNetR     | 0.945  |

### PanNuke:
| Model         | Head      | Score   |
|---------------|-----------|---------|
| HoverNext     | HoverNext | 0.784   |
| musk          | SimpleSeg | 0.639   |
| titan         | SimpleSeg | 0.691   |
| phikonv2      | SimpleSeg | 0.697   |
| uni           | SimpleSeg | 0.710   |
| provgigapath  | SimpleSeg | 0.714   |
| virchow2      | SimpleSeg | 0.717   |
| uni2          | SimpleSeg | 0.726   |
| musk          | UNetR     | 0.766   |
| titan         | UNetR     | 0.781   |
| uni2          | UNetR     | 0.785   |
| provgigapath  | UNetR     | 0.791   |
| uni           | UNetR     | 0.792   |
| virchow2      | UNetR     | 0.794   |
| phikonv2      | UNetR     | 0.797   |

### Lizard:
| Dataset       | Model     | Score   |
|---------------|-----------|---------|
| HoverNext     | HoverNext | 0.725   |
| musk          | SimpleSeg | 0.571   |
| titan         | SimpleSeg | 0.601   |
| phikonv2      | SimpleSeg | 0.607   |
| uni           | SimpleSeg | 0.636   |
| provgigapath  | SimpleSeg | 0.655   |
| uni2          | SimpleSeg | 0.684   |
| virchow2      | SimpleSeg | 0.689   |
| titan         | UNetR     | 0.697   |
| phikonv2      | UNetR     | 0.697   |
| provgigapath  | UNetR     | 0.697   |
| uni           | UNetR     | 0.716   |
| musk          | UNetR     | 0.722   |
| virchow2      | UNetR     | 0.735   |
| uni2          | UNetR     | 0.729   |


## Domain shift experiments:
### Lizard:
| Dataset       | Model     | Score   |
|---------------|-----------|---------|
| HoverNext     | HoverNext | 0.639   |
| musk          | SimpleSeg | 0.519   |
| phikonv2      | SimpleSeg | 0.549   |
| titan         | SimpleSeg | 0.553   |
| provgigapath  | SimpleSeg | 0.575   |
| uni           | SimpleSeg | 0.578   |
| uni2          | SimpleSeg | 0.608   |
| virchow2      | SimpleSeg | 0.614   |
| titan         | UNetR     | 0.630   |
| uni           | UNetR     | 0.643   |
| phikonv2      | UNetR     | 0.645   |
| provgigapath  | UNetR     | 0.660   |
| musk          | UNetR     | 0.663   |
| virchow2      | UNetR     | 0.670   |
| uni2          | UNetR     | 0.695   |

### PhenoCell DII vs CLR:
| Dataset      | Model     | Score   |
|--------------|-----------|---------|
| HoverNext    | HoverNext | 0.239   |
| musk         | UNetR     | 0.237   |
| titan        | UNetR     | 0.239   |
| uni          | UNetR     | 0.252   |
| phikonv2     | UNetR     | 0.256   |
| provgigapath | UNetR     | 0.261   |
| virchow2     | UNetR     | 0.266   |
| uni2         | UNetR     | 0.286   |

### PhenoCell MSI vs MSS:
| Dataset      | Model     | Score   |
|--------------|-----------|---------|
| HoverNext    | HoverNext | 0.230   |
| uni          | UNetR     | 0.233   |
| musk         | UNetR     | 0.234   |
| titan        | UNetR     | 0.243   |
| phikonv2     | UNetR     | 0.263   |
| provgigapath | UNetR     | 0.265   |
| uni2         | UNetR     | 0.284   |
| virchow2     | UNetR     | 0.288   |

### PhenoCell stage 3 vs stage 4:
| Dataset      | Model     | Score   |
|--------------|-----------|---------|
| HoverNext    | HoverNext | 0.209   |
| virchow2     | UNetR     | 0.222   |
| titan        | UNetR     | 0.222   |
| provgigapath | UNetR     | 0.223   |
| phikonv2     | UNetR     | 0.230   |
| musk         | UNetR     | 0.232   |
| uni          | UNetR     | 0.233   |
| uni2         | UNetR     | 0.251   |



