import numpy as np
from typing import List, Dict, Optional, Callable

def process_and_save_reductions(
        study_file: str,
        reductions: List[Optional[str]],
        id_errors_sample: np.ndarray,
        ood_errors_sample: np.ndarray,
        id_errors_cv: Dict[str, List[np.ndarray]],
        ood_errors_cv: Dict[str, List[np.ndarray]],
        id_likelihood_sample: np.ndarray,
        ood_likelihood_sample: np.ndarray,
        id_likelihoods_cv: List[np.ndarray],
        ood_likelihoods_cv: List[np.ndarray],
        id_cal_errors_sample: np.ndarray,
        ood_cal_errors_sample: np.ndarray,
        id_cal_errors_cv: List[np.ndarray],
        ood_cal_errors_cv: List[np.ndarray],
        model_seed: int,
        seed: int
) -> None:
    with open(study_file, "a") as f:
        for reduction in reductions:
            if reduction is not None:
                # Compute the reduction
                reduction_f: Callable = getattr(np, reduction)
                id_errors_sample_red = reduction_f(id_errors_sample, axis=0)
                ood_errors_sample_red = reduction_f(ood_errors_sample, axis=0)

                # Save the results
                id_errors_cv[reduction].append(id_errors_sample_red)
                ood_errors_cv[reduction].append(ood_errors_sample_red)

                # Print the results to file
                f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} ID {reduction} of (MSE, MAE): {id_errors_sample_red}\n")
                f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} OOD {reduction} of (MSE, MAE): {ood_errors_sample_red}\n")

        # Save likelihoods and calibration errors
        id_likelihoods_cv.append(id_likelihood_sample)
        ood_likelihoods_cv.append(ood_likelihood_sample)
        id_cal_errors_cv.append(id_cal_errors_sample)
        ood_cal_errors_cv.append(ood_cal_errors_sample)

        # Print likelihoods and calibration errors
        f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} ID likelihoods: {id_likelihood_sample}\n")
        f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} OOD likelihoods: {ood_likelihood_sample}\n")
        f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} ID calibration errors: {id_cal_errors_sample}\n")
        f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} OOD calibration errors: {ood_cal_errors_sample}\n")

def finalize_and_save_results(
        study_file: str,
        reductions: List[Optional[str]],
        id_errors_cv: Dict[str, List[np.ndarray]],
        ood_errors_cv: Dict[str, List[np.ndarray]],
        id_errors_model: Dict[str, List[np.ndarray]],
        ood_errors_model: Dict[str, List[np.ndarray]],
        id_likelihoods_cv: List[np.ndarray],
        ood_likelihoods_cv: List[np.ndarray],
        id_likelihoods_model: List[np.ndarray],
        ood_likelihoods_model: List[np.ndarray],
        id_cal_errors_cv: List[np.ndarray],
        ood_cal_errors_cv: List[np.ndarray],
        id_cal_errors_model: List[np.ndarray],
        ood_cal_errors_model: List[np.ndarray],
        model_seed: int
) -> None:
    with open(study_file, "a") as f:
        for reduction in reductions:
            if reduction is not None:
                # Compute final reduction and average
                id_errors_cv[reduction] = np.vstack(id_errors_cv[reduction])
                ood_errors_cv[reduction] = np.vstack(ood_errors_cv[reduction])
                id_errors_cv[reduction] = np.mean(id_errors_cv[reduction], axis=0)
                ood_errors_cv[reduction] = np.mean(ood_errors_cv[reduction], axis=0)

                # Save the results
                id_errors_model[reduction].append(id_errors_cv[reduction])
                ood_errors_model[reduction].append(ood_errors_cv[reduction])

                # Print the results to file
                f.write(f"\tModel Seed: {model_seed} ID {reduction} of (MSE, MAE): {id_errors_cv[reduction]}\n")
                f.write(f"\tModel Seed: {model_seed} OOD {reduction} of (MSE, MAE): {ood_errors_cv[reduction]}\n")

        # Compute and average likelihoods and calibration errors
        id_likelihoods_cv = np.mean(id_likelihoods_cv)
        ood_likelihoods_cv = np.mean(ood_likelihoods_cv)
        id_cal_errors_cv = np.vstack(id_cal_errors_cv)
        ood_cal_errors_cv = np.vstack(ood_cal_errors_cv)
        id_cal_errors_cv = np.mean(id_cal_errors_cv, axis=0)
        ood_cal_errors_cv = np.mean(ood_cal_errors_cv, axis=0)

        # Save the results
        id_likelihoods_model.append(id_likelihoods_cv)
        ood_likelihoods_model.append(ood_likelihoods_cv)
        id_cal_errors_model.append(id_cal_errors_cv)
        ood_cal_errors_model.append(ood_cal_errors_cv)

        # Print the results to file
        f.write(f"\tModel Seed: {model_seed} ID likelihoods: {id_likelihoods_cv}\n")
        f.write(f"\tModel Seed: {model_seed} OOD likelihoods: {ood_likelihoods_cv}\n")
        f.write(f"\tModel Seed: {model_seed} ID calibration errors: {id_cal_errors_cv}\n")
        f.write(f"\tModel Seed: {model_seed} OOD calibration errors: {ood_cal_errors_cv}\n")
