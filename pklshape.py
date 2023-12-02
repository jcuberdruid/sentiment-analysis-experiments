import pandas as pd
import os

def print_shapes_recursively(obj, level=0):
    if isinstance(obj, pd.DataFrame):
        print("  " * level + f"DataFrame shape: {obj.shape}")
        for col in obj.columns:
            print_shapes_recursively(obj[col], level + 1)
    elif isinstance(obj, pd.Series) and isinstance(obj.iloc[0], (pd.DataFrame, pd.Series)):
        print("  " * level + f"Series of {type(obj.iloc[0])} shape: {obj.iloc[0].shape}")
        for item in obj:
            print_shapes_recursively(item, level + 1)

def main():
    file_path = "../results/word_vectors.pkl"
    if os.path.exists(file_path):
        df = pd.read_pickle(file_path)
        print("Opened DataFrame from:", file_path)
        print_shapes_recursively(df)
        print(len(df.iloc[0]))
        print(len(df.iloc[0][0]))
        print(len(df.iloc[0][0][0]))
    else:
        print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()

