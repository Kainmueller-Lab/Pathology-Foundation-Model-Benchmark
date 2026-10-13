import argparse
import csv
import pandas as pd
import numpy as np
from benchmark.lmdb_dataset import LMDBDataset

if __name__ == "__main__":

    print("\n=== Data Efficiency Split Script ===\n")

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--percentage", type=int, default=20, help="Percentage of training data to keep")
    parser.add_argument("--db_path", type=str, default='/fast/AG_Kainmueller/data/patho_foundation_model_bench_data/schuerch_dataset_lmdb/lmdb')
    parser.add_argument("--old_split_file", type=str, default='/fast/AG_Kainmueller/data/patho_foundation_model_bench_data/schuerch_dataset_lmdb/splits/schuerch_dataset_split.csv', help="File of split to use")
    parser.add_argument("--new_split_path", type=str, default='/fast/AG_Kainmueller/data/patho_foundation_model_bench_data/schuerch_dataset_lmdb/splits/data_efficiency_splits', help="Path to save the new split")

    args = parser.parse_args()

    factor = args.percentage / 100
    print(f"Selected data efficiency factor: {factor:.2f} (using {args.percentage}% of training data)\n")

    split_df = pd.read_csv(args.old_split_file)

    # load dataset
    datasets = {}
    splits = ["train", "valid", "test"]
    for split in splits:
        include_fovs = split_df[split_df["train_test_val_split"] == split]["sample_name"].tolist()
        dataset = LMDBDataset(path=args.db_path, include_sample_names=include_fovs) #include_tile_names=include_fovs)
        datasets[split] = dataset

    n_train_tiles = int(factor * len(datasets['train']))
    print(f"Number of training tiles to keep: {n_train_tiles}")

    tile_list = []
    idc_to_keep = np.random.choice(len(datasets['train']), n_train_tiles)

    for split in splits:
        dataset = datasets[split]
        if split == 'train':
            tile_list.extend((el["tile_name"], split) for i, el in enumerate(dataset) if i in idc_to_keep)
        else:
            tile_list.extend((el["tile_name"], split) for el in dataset)
            


    # Save to CSV
    output_file = f"{args.new_split_path}/split_{args.percentage}_percent_train_tile_level.csv"
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['sample_name', 'train_test_val_split'])
        for tile, split in tile_list:
            writer.writerow([tile, split])

    print(f"New split file saved to: {output_file}")
