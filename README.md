# Pathology-Foundation-Model-Benchmark
Benchmarking the ability of pathology foundation models to do cell type classification

Assumes a .env file with the following variables:
- `HF_TOKEN`: Huggingface API token to download models with access restrictions



The cell type classes are merged as follows:

| Cell Type                              | Category                |
|----------------------------------------|--------------------------|
| background                             | Background               |
| B cells                                | B cells                  |
| CD11b+ monocytes                       | Macrophages/Monocytes    |
| CD11b+CD68+ macrophages                | Macrophages/Monocytes    |
| CD11c+ DCs                             | Dendritic cells          |
| CD163+ macrophages                     | Macrophages/Monocytes    |
| CD3+ T cells                           | T cells                  |
| CD4+ T cells                           | T cells                  |
| CD4+ T cells CD45RO+                   | T cells                  |
| CD4+ T cells GATA3+                    | T cells                  |
| CD68+ macrophages                      | Macrophages/Monocytes    |
| CD68+ macrophages GzmB+                | Macrophages/Monocytes    |
| CD68+CD163+ macrophages                | Macrophages/Monocytes    |
| CD8+ T cells                           | T cells                  |
| NK cells                               | NK cells                 |
| Tregs                                  | T cells                  |
| adipocytes                             | Adipocytes               |
| granulocytes                           | Granulocytes             |
| immune cells                           | Other cells              |
| immune cells / vasculature            | Other cells              |
| lymphatics                             | Vasculature/Lymphatics   |
| nerves                                 | Nerves                   |
| plasma cells                           | Plasma cells             |
| smooth muscle                          | Smooth muscle            |
| stroma                                 | Stroma                   |
| tumor cells                            | Tumor cells              |
| tumor cells / immune cells             | Other cells              |
| undefined                              | Other cells              |
| vasculature                            | Vasculature/Lymphatics   |



Here are the more detailed results table on the PhenoCell dataset:
| Dataset   | Model        | Head      | Score   |
|-----------|--------------|-----------|---------|
| PhenoCell | SimpleSeg    | musk      | 0.200   |
| PhenoCell | SimpleSeg    | titan     | 0.232   |
| PhenoCell | HoverNext    | HoverNext | 0.238   |
| PhenoCell | SimpleSeg    | phikonv2  | 0.243   |
| PhenoCell | UNetR        | titan     | 0.264   |
| PhenoCell | SimpleSeg    | provgigapath | 0.264   |
| PhenoCell | UNetR        | phikonv2  | 0.280   |
| PhenoCell | UNetR        | provgigapath | 0.280   |
| PhenoCell | UNetR        | uni       | 0.288   |
| PhenoCell | SimpleSeg    | virchow2  | 0.286   |
| PhenoCell | UNetR        | virchow2  | 0.295   |
| PhenoCell | UNetR        | uni2      | 0.301   |
| PhenoCell | SimpleSeg    | uni2      | 0.303   |


Arctique:
| Model         | Head      | Score  |
|---------------|-----------|--------|
| HoverNext     | HoverNext | 0.938  |
| provgigapath  | SimpleSeg | 0.830  |
| uni           | SimpleSeg | 0.826  |
| phikonv2      | SimpleSeg | 0.746  |
| titan         | SimpleSeg | 0.800  |
| virchow2      | SimpleSeg | 0.838  |
| musk          | SimpleSeg | 0.745  |
| uni2          | SimpleSeg | 0.826  |
| uni2          | UNetR     | 0.945  |
| virchow2      | UNetR     | 0.943  |
| uni           | UNetR     | 0.934  |
| musk          | UNetR     | 0.929  |
| provgigapath  | UNetR     | 0.927  |
| titan         | UNetR     | 0.922  |
| phikonv2      | UNetR     | 0.927  |

PanNuke:
| Model         | Head      | Score   |
|---------------|-----------|---------|
| HoverNext     | HoverNext | 0.784   |
| provgigapath  | SimpleSeg | 0.714   |
| uni           | SimpleSeg | 0.710   |
| phikonv2      | SimpleSeg | 0.697   |
| titan         | SimpleSeg | 0.691   |
| virchow2      | SimpleSeg | 0.717   |
| musk          | SimpleSeg | 0.639   |
| uni2          | SimpleSeg | 0.726   |
| virchow2      | UNetR     | 0.794   |
| uni           | UNetR     | 0.792   |
| provgigapath  | UNetR     | 0.791   |
| uni2          | UNetR     | 0.785   |
| titan         | UNetR     | 0.781   |
| musk          | UNetR     | 0.766   |
| phikonv2      | UNetR     | 0.797   |

Lizard:
| Dataset       | Model     | Score   |
|---------------|-----------|---------|
| HoverNext     | HoverNext | 0.725   |
| musk          | SimpleSeg | 0.571   |
| phikonv2      | SimpleSeg | 0.607   |
| provgigapath  | SimpleSeg | 0.655   |
| titan         | SimpleSeg | 0.601   |
| uni           | SimpleSeg | 0.636   |
| uni2          | SimpleSeg | 0.684   |
| virchow2      | SimpleSeg | 0.689   |
| musk          | UNetR     | 0.722   |
| phikonv2      | UNetR     | 0.697   |
| provgigapath  | UNetR     | 0.697   |
| titan         | UNetR     | 0.697   |
| uni           | UNetR     | 0.716   |
| uni2          | UNetR     | 0.729   |
| virchow2      | UNetR     | 0.735   |

Here are the detailed domain shift experiments:
Lizard:
| Dataset       | Model     | Score   |
|---------------|-----------|---------|
| HoverNext     | HoverNext | 0.639   |
| virchow2      | SimpleSeg | 0.614   |
| uni2          | SimpleSeg | 0.608   |
| uni           | SimpleSeg | 0.578   |
| provgigapath  | SimpleSeg | 0.575   |
| titan         | SimpleSeg | 0.553   |
| phikonv2      | SimpleSeg | 0.549   |
| musk          | SimpleSeg | 0.519   |
| uni2          | UNetR     | 0.695   |
| virchow2      | UNetR     | 0.670   |
| musk          | UNetR     | 0.663   |
| provgigapath  | UNetR     | 0.660   |
| phikonv2      | UNetR     | 0.645   |
| uni           | UNetR     | 0.643   |
| titan         | UNetR     | 0.630   |

PhenoCell DII vs CLR:
| Dataset       | Model     | Score   |
|---------------|-----------|---------|
| HoverNext     | HoverNext | 0.239   |
| uni2          | UNetR     | 0.286   |
| virchow2      | UNetR     | 0.266   |
| provgigapath  | UNetR     | 0.261   |
| phikonv2      | UNetR     | 0.256   |
| uni           | UNetR     | 0.252   |
| titan         | UNetR     | 0.239   |
| musk          | UNetR     | 0.237   |

PhenoCell MSI vs MSS:
| Dataset       | Model     | Score   |
|---------------|-----------|---------|
| HoverNext     | HoverNext | 0.230   |
| virchow2      | UNetR     | 0.288   |
| uni2          | UNetR     | 0.284   |
| provgigapath  | UNetR     | 0.265   |
| phikonv2      | UNetR     | 0.263   |
| titan         | UNetR     | 0.243   |
| uni           | UNetR     | 0.233   |
| musk          | UNetR     | 0.234   |

PhenoCell stage 3 vs stage 4:
| Dataset       | Model     | Score   |
|---------------|-----------|---------|
| HoverNext     | HoverNext | 0.209   |
| uni2          | UNetR     | 0.251   |
| uni           | UNetR     | 0.233   |
| phikonv2      | UNetR     | 0.230   |
| provgigapath  | UNetR     | 0.223   |
| virchow2      | UNetR     | 0.222   |
| titan         | UNetR     | 0.222   |
| musk          | UNetR     | 0.232   |


