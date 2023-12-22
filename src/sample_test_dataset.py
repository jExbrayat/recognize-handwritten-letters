import numpy as np


def sample_dataset(
    original_dataset_path: str, sample_dataset_path: str, sample_size: int
) -> None:
    original_dataset = np.loadtxt(original_dataset_path, delimiter=",")

    # Create an array of random unique indexes
    random_choice = np.random.choice(
        original_dataset.shape[0], sample_size, replace=False
    )

    # Take a sample of dataset
    sample_dataset = original_dataset[random_choice]

    # Write sampled dataset
    np.savetxt(sample_dataset_path, sample_dataset, delimiter=",")
