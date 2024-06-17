import ast
import csv
import os
from os import makedirs
import sys
from pickle import dump, load as pkl_load
import subprocess

import numpy as np


output_folder = "processed"
makedirs(output_folder, exist_ok=True)


def load_and_save(
    category,
    filename,
    dataset,
    dataset_folder,
    dataset_name,
):
    # Create folder in output_folder with dataset_name
    makedirs(os.path.join(output_folder, dataset_name), exist_ok=True)
    temp = np.genfromtxt(
        os.path.join(dataset_folder, category, filename),
        dtype=np.float32,
        delimiter=",",
    )
    print(dataset, category, filename, temp.shape)
    with open(
        os.path.join(output_folder, dataset_name,
                     dataset + "_" + category + ".pkl"),
        "wb",
    ) as file:
        dump(temp, file)


def load_data(dataset_name):
    os.makedirs(
        os.path.join(output_folder, dataset_name),
        exist_ok=True,
    )

    if dataset_name == "SMD":
        dataset_folder = "ServerMachineDataset"
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith(".txt"):
                load_and_save(
                    "train",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    dataset_name,
                )
                load_and_save(
                    "test",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    dataset_name,
                )
                load_and_save(
                    "test_label",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    dataset_name,
                )
    elif dataset_name == "SMAP" or dataset_name == "MSL":
        dataset_folder = "data"
        with open(os.path.join(dataset_folder,
                               "labeled_anomalies.csv",
                               ), "r") as file:
            csv_reader = csv.reader(file, delimiter=",")
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        label_folder = os.path.join(dataset_folder, "test_label")
        makedirs(label_folder, exist_ok=True)
        data_info = [row for row in res if row[1]
                     == dataset_name and row[0] != "P-2"]
        labels = []
        for row in data_info:
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=bool)
            for anomaly in anomalies:
                label[anomaly[0]: anomaly[1] + 1] = True
            labels.extend(label)
        labels = np.asarray(labels)
        print(dataset_name, "test_label", labels.shape)
        with open(
            os.path.join(
                output_folder,
                dataset_name,
                dataset_name + "_" + "test_label" + ".pkl",
            ),
            "wb",
        ) as file:
            dump(labels, file)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(
                    os.path.join(dataset_folder, category, filename + ".npy")
                )
                data.extend(temp)
            data = np.asarray(data)
            print(dataset_name, category, data.shape)
            with open(
                os.path.join(
                    output_folder,
                    dataset_name,
                    dataset_name + "_" + category + ".pkl",
                ),
                "wb",
            ) as file:
                dump(data, file)

        for c in ["train", "test"]:
            concatenate_and_save(c)


def get_data(dataset_name):
    """Loads the pickle files for a given dataset and returns the data.

    Args:
        dataset_name (str): Name of the dataset to load
    """
    print("Loading data for", dataset_name)
    with open(
        os.path.join(output_folder, dataset_name,
                     dataset_name + "_train.pkl"), "rb"
    ) as f:
        train_data = pkl_load(f)

    with open(
        os.path.join(output_folder, dataset_name,
                     dataset_name + "_test.pkl"), "rb"
    ) as f:
        test_data = pkl_load(f)

    with open(
        os.path.join(output_folder, dataset_name,
                     dataset_name + "_test_label.pkl"),
        "rb",
    ) as f:
        test_label = pkl_load(f)

    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test label shape: ", test_label.shape)

    # For anomaly detection unsupervised way, so no train_label
    return (train_data, None), (test_data, test_label)


if __name__ == "__main__":
    datasets = ["SMD", "SMAP", "MSL"]

    args = sys.argv[1:]
    if args[0] == "download":
        commands = [
            "git clone https://github.com/NetManAIOps/OmniAnomaly.git",
            "mv OmniAnomaly/ServerMachineDataset .",
            "rm -rf OmniAnomaly",
            "wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip",
            "unzip data.zip",
            "rm data.zip",
            "cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv",  # noqa
        ]

        for command in commands:
            subprocess.run(command, shell=True)

    # Adds the option to process all datasets at once
    commands = sys.argv[1:]
    if "ALL" in commands:
        commands = datasets

    load = []
    if len(commands) > 0:
        for d in commands:
            if d in datasets:
                load_data(d)
                # tmp = get_data(d)
    else:
        print(
            """
        Usage: python data_preprocess.py <datasets>
        where <datasets> should be one of ['SMD', 'SMAP', 'MSL', 'ALL']
        """
        )
