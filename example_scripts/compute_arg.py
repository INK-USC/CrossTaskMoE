import pandas as pd
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--summary1", default=None, type=str, required=True)
    parser.add_argument("--summary2", default=None, type=str, required=True)

    args = parser.parse_args()

    df1 = pd.read_csv(args.summary1, index_col=0)
    df2 = pd.read_csv(args.summary2, index_col=0)

    df1 = df1[df1["entry"]=="mean"]
    df2 = df2[df2["entry"]=="mean"]

    df1 = df1.rename(columns={"test_performance": "test_performance_1"})
    df2 = df2.rename(columns={"test_performance": "test_performance_2"})
    df1 = df1.drop(columns=["dev_performance", "entry"])
    df2 = df2.drop(columns=["dev_performance", "entry"])
    # df1.set_index("task", inplace=True)
    # df2.set_index("task", inplace=True)
    print(df1.head())
    print(df1.shape)
    print(df2.head())
    print(df2.shape)

    df = df1.merge(df2, left_on="task", right_on="task", how="inner")
    # df = df.drop(columns="dev_performance")
    print(df.shape)
    print(df.head())

    diff = df["test_performance_2"] - df["test_performance_1"]
    df["diff"] = diff
    df["rg"] = diff.div(df["test_performance_1"])
    df.to_csv("results_summary/CMP_mtl_fisher128lstmfreeze.csv")

    print(np.mean(df["rg"]))



if __name__ == "__main__":
    main()