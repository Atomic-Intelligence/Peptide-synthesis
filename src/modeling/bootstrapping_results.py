import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import gaussian_kde
from scipy.special import kl_div
from copy import deepcopy
from tqdm import tqdm


def bootstrapping_data(
        path_to_synt_table: str,
        path_to_original_table: str,
        nonzero_threshold: float = 0.6,
        sample_size: int = 182,
        iteration_number: int = 1000,
):
    original_peptides = pd.read_csv(path_to_original_table)
    synt_peptides = pd.read_csv(path_to_synt_table)
    if sample_size > synt_peptides.shape[0]:
        return ValueError(f"Sample size {sample_size} is too big")
    non_zero_col = [
                       col for col in original_peptides.columns
                       if (original_peptides[col] == 0.0).sum() < nonzero_threshold * (len(original_peptides[col]))
                   ][1:]

    M = 10  # limit
    min_ks = M
    min_statistic = {col: {'kl_divergence': M} for col in non_zero_col}
    best_seed = 1000
    for i in tqdm(range(iteration_number), desc="Bootstrapping Progress"):
        seed = np.random.randint(0, 1000)
        synt_peptides_sample = synt_peptides.sample(sample_size, random_state=seed)
        results = {}
        for col in non_zero_col:
            synth_col = synt_peptides_sample[col]
            original_col = original_peptides[col]

            # compute ks
            stat, p_value = ks_2samp(synth_col, original_col)

            # KL divergence
            # Estimate the probability density functions
            pdf_real = gaussian_kde(original_col, bw_method='scott')
            pdf_synthetic = gaussian_kde(synth_col, bw_method='scott')

            # Create a range of values to evaluate the PDF
            x = np.linspace(min(original_col.min(), synth_col.min()),
                            max(original_col.max(), synth_col.max()), 1000)

            # Compute KL divergence
            kl_divergence = np.sum(kl_div(pdf_real(x), pdf_synthetic(x)))
            results[col] = {'ks_p-value': p_value, 'kl_divergence': kl_divergence}

        counter_ks = 0
        counter_kl = 0
        # inspect the results
        for col, result in results.items():
            # alpha = 0.05
            if result['ks_p-value'] < 0.05:
                counter_ks += 1
            # the goal is to minimize kl divergence for as many peptides as possible
            if result['kl_divergence'] < min_statistic[col]['kl_divergence']:
                counter_kl += 1

        ks = counter_ks / len(non_zero_col)
        # round to 3 decimal places
        ks = int(ks * 1000) / 1000
        # dl percentage
        dl = counter_kl / len(non_zero_col)
        # A sample is considered good if the KS result,
        # rounded to three decimal places, is lower than the best previous result,
        # or if it is equal but has a better KL divergence.
        if (ks < min_ks) or (dl > 0.5 and ks == min_ks):
            min_ks = ks
            min_statistic = deepcopy(results)
            best_seed = seed

    return best_seed, min_statistic

