import os

def get_save_path(classifier_idx, fold_idx, path):
    if fold_idx is None:
        save_path = os.path.join(path, f"classifier_{classifier_idx}_full_data")
    else:
        save_path = os.path.join(path, f"classifier_{classifier_idx}", f"fold_{fold_idx}")
    return save_path

def get_load_path(classifier_idx, fold_idx, path):
    return get_save_path(classifier_idx, fold_idx, path)

def normalize_dataset(dataset_, min_along_cols=None, max_along_cols=None):
    dataset = dataset_.copy()
    num_of_features = dataset.shape[1]
    print("normalizing dataset with shape", dataset.shape)
    if min_along_cols is None:
        assert max_along_cols is None, "both min and max should be either None or have a definite value"
        min_along_cols = dataset.min(axis=0)
        max_along_cols = dataset.max(axis=0) + 1e-5
    else:
        assert max_along_cols is not None, "both min and max should be either None or have a definite value"
    diff_along_cols = max_along_cols - min_along_cols
    for i in range(num_of_features):
        dataset[:, i] = (dataset[:, i] - min_along_cols[i]) / diff_along_cols[i]
    dataset = dataset * 255.0
    
    return dataset, min_along_cols, max_along_cols