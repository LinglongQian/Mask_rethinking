import argparse
import os
from pypots.data.saving import load_dict_from_h5, pickle_load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypots.imputation import LOCF, Mean, Median

'''
python feature_display_electricity.py --dataset Electricity --dataset_fold_path data/point_rate01/electricity_load_diagrams_rate01_step96_point,data/point_rate05/electricity_load_diagrams_rate05_step96_point,data/point_rate09/electricity_load_diagrams_rate09_step96_point,data/block_rate05/electricity_load_diagrams_rate05_step96_block_blocklen8,data/subseq_rate05/electricity_load_diagrams_rate05_step96_subseq_seqlen72 --pdf_saving_path pdf_display_electricity --model_result_parent_fold pretrained_models/results_point_rate01/Electricity,pretrained_models/results_point_rate05/Electricity,pretrained_models/results_point_rate09/Electricity,pretrained_models/results_block_rate05/Electricity,pretrained_models/results_subseq_rate05/Electricity
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_fold_paths",
        type=str,
        help="Comma separated list of dataset fold paths, each should include 3 H5 files train.h5, val.h5 and test.h5",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="the dataset name",
        required=True,
    )
    parser.add_argument(
        "--model_result_parent_folds",
        type=str,
        help="Comma separated list of model result parent folders, corresponding to each dataset",
        required=True,
    )
    parser.add_argument(
        "--pdf_saving_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    dataset_fold_paths = args.dataset_fold_paths.split(',')
    model_result_parent_folds = args.model_result_parent_folds.split(',')
    assert len(dataset_fold_paths) == len(model_result_parent_folds), "Dataset paths and model result paths must have the same length."

    model_lists = [
        'iTransformer', 'SAITS', 'NonstationaryTransformer', 'ETSformer', 'PatchTST', 
        'Crossformer', 'Informer', 'Autoformer', 'Pyraformer', 'Transformer', 
        'BRITS', 'MRNN', 'GRUD', 'TimesNet', 'MICN', 'SCINet', 
        'StemGNN',  'FreTS', 'Koopa', 'DLinear', 'FiLM', 'CSDI', 
        'USGAN', 'GPVAE', 'Mean', 'Median', 'LOCF'
    ]

    # Assume that the feature dimensions of all datasets are the same, take the number of feature dimensions from the first dataset.
    first_dataset_path = dataset_fold_paths[0]
    first_test_set_path = os.path.join(first_dataset_path, "test.h5")
    first_test_set = load_dict_from_h5(first_test_set_path)
    n_features = first_test_set["X"].shape[2]

    # Randomly select 7 features.
    selected_features = np.random.choice(n_features, 7, replace=False)

    n_rows = len(dataset_fold_paths)  # Number of datasets.
    n_cols = len(selected_features)  # Number of randomly selected features.
    fig_size = [4 * n_cols, 4 * n_rows]  # Dynamically adjust the chart size.
    n_round = 0

    for model in model_lists:
        plt.rcParams["font.size"] = 20
        fig, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(fig_size[0], fig_size[1]), dpi=150
        )

        for i, (dataset_fold_path, model_result_parent_fold) in enumerate(zip(dataset_fold_paths, model_result_parent_folds)):
            dataset_name = os.path.basename(dataset_fold_path)
            test_set_path = os.path.join(dataset_fold_path, "test.h5")
            test_set = load_dict_from_h5(test_set_path)
            test_X = test_set["X"]
            test_X_ori = test_set["X_ori"]
            test_X_ori = np.nan_to_num(test_set["X_ori"])

            if model not in ["Mean", "Median", "LOCF"]:
                model_result_path = os.path.join(model_result_parent_fold, f"{model}_{args.dataset}")
                imputed_data_path = os.path.join(
                    model_result_path,
                    f"round_{n_round}/imputation.pkl",
                )
                imputed_data = pickle_load(imputed_data_path)
                if isinstance(imputed_data, dict):
                    test_set_imputation = imputed_data["test_set_imputation"]
                elif isinstance(imputed_data, np.ndarray):
                    test_set_imputation = imputed_data
                else:
                    raise ValueError("The imputed data should be a dictionary or a numpy array.")
                X_imputed = test_set_imputation
            elif model == "Mean":
                X_imputed = Mean().impute(test_set)
            elif model == "Median":
                X_imputed = Median().impute(test_set)
            elif model == "LOCF":
                X_imputed = LOCF().impute(test_set)

            X = test_X_ori
            X_ori = test_X
            sample_idx = 0

            for j, feature_idx in enumerate(selected_features):  # Iterate over the randomly selected features.
                df = pd.DataFrame({"x": np.arange(0, test_X.shape[1]), "val": X_imputed[sample_idx, :, feature_idx]})
                df1 = pd.DataFrame({"x": np.arange(0, test_X.shape[1]), "val": X[sample_idx, :, feature_idx]})
                df2 = pd.DataFrame({"x": np.arange(0, test_X.shape[1]), "val": X_ori[sample_idx, :, feature_idx]})

                axes[i, j].plot(df1.x, df1.val, color="red", marker="x", linestyle="None", label="Missing Values")
                axes[i, j].plot(df2.x, df2.val, color="blue", marker="o", linestyle="None", label="Observed Values")
                axes[i, j].plot(df.x, df.val, color="green", linestyle="solid", label="Imputed Values")
                axes[i, j].legend(loc="upper right", fontsize=8)
                axes[i, j].set_xticks(list(np.arange(0, test_X.shape[1], max(15, test_X.shape[1]//10))))
                # plt.xticks(fontsize=12)
                if i == 0:
                    axes[i, j].set_title(f"Feature #{feature_idx+1}")
                if j == 0:
                    axes[i, j].set_ylabel(f"{dataset_name.replace('electricity_load_diagrams_', '').replace('_step96', '').replace('_blocklen8', '').replace('_seqlen72', '')}")

        # plt.suptitle(f"Imputation Results for {model}", fontsize=30)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to avoid the main title overlapping the subplots.
        plt.savefig(f"{args.pdf_saving_path}/{model}_imputation_comparison.pdf", format='pdf')
        plt.close(fig)
