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



Here are the more detailed results table:
| Model         | Head      | Score     |
|---------------|-----------|-----------|
| HoverNext     | HoverNext | 0.938473  |
| provgigapath  | SimpleSeg | 0.829691  |
| uni           | SimpleSeg | 0.826109  |
| phikonv2      | SimpleSeg | 0.745448  |
| titan         | SimpleSeg | 0.800081  |
| virchow2      | SimpleSeg | 0.837534  |
| musk          | SimpleSeg | 0.744879  |
| uni2          | SimpleSeg | 0.825901  |
| uni2          | UNetR     | 0.945103  |
| virchow2      | UNetR     | 0.943260  |
| uni           | UNetR     | 0.933452  |
| musk          | UNetR     | 0.929207  |
| provgigapath  | UNetR     | 0.927416  |
| titan         | UNetR     | 0.922186  |
| phikonv2      | UNetR     | 0.926500  |
