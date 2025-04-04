import numpy as np


def sample_dataset(
    original_dataset_path: str, sample_dataset_path: str, sample_size: int
) -> None:
    """Create a new file that is a sample of original dataset.

    Args:
        original_dataset_path (str): Path to orginal dataset file. Must end with .csv
        sample_dataset_path (str): Path for writing new sample dataset file.
            Must end with .csv
        sample_size (int): Desired sample size

    """
    original_dataset = np.loadtxt(original_dataset_path, delimiter=",")

    # Create an array of random unique indexes
    rng = np.random.default_rng(0)

    random_choice = rng.choice(original_dataset.shape[0], sample_size, replace=False)

    # Take a sample of dataset
    sample_dataset = original_dataset[random_choice]

    # Write sampled dataset
    np.savetxt(sample_dataset_path, sample_dataset, delimiter=",")
