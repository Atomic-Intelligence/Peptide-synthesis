from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import gaussian_kde
from copy import deepcopy
from tqdm import tqdm


def process_iteration(seed, synt_peptides, original_peptides, non_zero_col, sample_size):
    synt_peptides_sample = synt_peptides.sample(sample_size, random_state=seed)
    results = {}
    for col in non_zero_col:
        synth_col = synt_peptides_sample[col]
        original_col = original_peptides[col]

        # compute ks
        stat, p_value = ks_2samp(synth_col, original_col)

        # KL divergence
        pdf_real = gaussian_kde(original_col, bw_method='scott')
        pdf_synthetic = gaussian_kde(synth_col, bw_method='scott')

        # Compute KL divergence
        x = np.linspace(min(original_col.min(), synth_col.min()),
                        max(original_col.max(), synth_col.max()), 1000)
        kl_divergence = np.sum(np.where(pdf_synthetic(x) != 0, pdf_real(x) * np.log(pdf_real(x) / pdf_synthetic(x)), 0))

        results[col] = {'ks_p-value': p_value, 'kl_divergence': kl_divergence}

    return seed, results  # Return the seed along with the results


def bootstrapping_data(
        path_to_synt_table: str,
        path_to_original_table: str,
        nonzero_threshold: float = 0.6,
        sample_size: int = 182,
        iteration_number: int = 500,
        n_jobs=-1  # Use all available CPU cores by default
):
    original_peptides = pd.read_csv(path_to_original_table)
    synt_peptides = pd.read_csv(path_to_synt_table)
    if sample_size > synt_peptides.shape[0]:
        return ValueError(f"Sample size {sample_size} is too big")

    non_zero_col = [
                       col for col in original_peptides.columns
                       if (original_peptides[col] == 0.0).sum() < nonzero_threshold * len(original_peptides[col])
                   ][1:]

    M = 10  # limit for comparison
    min_ks = M
    min_statistic = {col: {'kl_divergence': M} for col in non_zero_col}
    best_seed = 1000

    # Generate a list of random seeds for the parallel iterations
    seeds = np.random.randint(0, 10000, size=iteration_number)

    # Parallelize the bootstrapping process and keep track of seeds
    results_list = Parallel(n_jobs=n_jobs)(delayed(process_iteration)(
        seed, synt_peptides, original_peptides, non_zero_col, sample_size
    ) for seed in tqdm(seeds, desc="Bootstrapping Progress"))

    # Post-process the results
    for seed, results in results_list:
        counter_ks = 0
        counter_kl = 0
        for col, result in results.items():
            if result['ks_p-value'] < 0.05:
                counter_ks += 1
            if result['kl_divergence'] < min_statistic[col]['kl_divergence']:
                counter_kl += 1

        ks = counter_ks / len(non_zero_col)
        ks = int(ks * 1000) / 1000  # round to 3 decimal places
        dl = counter_kl / len(non_zero_col)

        # A sample is considered good if the KS result is better or KL divergence improves
        if (ks < min_ks) or (dl > 0.5 and ks == min_ks):
            min_ks = ks
            min_statistic = deepcopy(results)
            best_seed = seed  # Update the best seed

    return best_seed, min_statistic




