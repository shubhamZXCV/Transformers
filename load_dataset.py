import random
import pickle

def load_parallel_corpus(fi_file, en_file):
    data = []
    with open(fi_file, encoding="utf-8") as f_fi, open(en_file, encoding="utf-8") as f_en:
        for fi_line, en_line in zip(f_fi, f_en):
            fi_line = fi_line.strip()
            en_line = en_line.strip()
            if fi_line and en_line:  # ignore empty lines
                data.append({"fi": fi_line, "en": en_line})
    return data


def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:]
    }


if __name__ == "__main__":
    fi_file = "EUbookshop/EUbookshop.fi"
    en_file = "EUbookshop/EUbookshop.en"

    dataset = load_parallel_corpus(fi_file, en_file)
    print("Total sentence pairs:", len(dataset))

    splits = split_dataset(dataset)

    # Save into one pickle file
    with open("dataset.pkl", "wb") as f:
        pickle.dump(splits, f)

    print("Saved dataset.pkl with keys:", list(splits.keys()))
    print("Train size:", len(splits["train"]))
    print("Val size:", len(splits["val"]))
    print("Test size:", len(splits["test"]))
    print("Example:", splits["train"][0])
