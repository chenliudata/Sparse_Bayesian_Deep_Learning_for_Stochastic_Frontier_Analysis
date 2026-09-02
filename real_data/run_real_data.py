from __future__ import annotations

import os
import json
import pickle
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import pyro
import arviz as az
from scipy.optimize import root_scalar
from scipy.stats import norm
from pyro.infer import MCMC, NUTS, Predictive

from config_real import (
    NUM_SAMPLES, WARMUP_STEPS, NUM_CHAINS, TARGET_ACCEPT_PROB, MAX_TREE_DEPTH,
    IG_DF, IG_Q, BASE_RESULTS,
    DATA_PATH, OUTPUT_VAR, INPUT_VARS, N_INPUTS,
)
from mle_real import estimate_mle, mle_summary, mu_cobb_multivar
from mle_nn_real import fit_relu_nn_sfm, fit_crelu_nn_sfm
from model_blm_nin import BLM_SFM_NIn
from model_relu_nin import BNN_SFM_RELU_N_NIn
from model_crelu_hn_nin import BNN_SFM_CRELU_HN_NIn
from model_crelu_hl_nin import BNN_SFM_CRELU_HalfLaplace_NIn
from model_crelu_hs_nin import BNN_SFM_CRELU_HS_NIn
from standardize_real import (
    build_std_stats, standardize_inputs, standardize_output,
    unscale_sigma2, unscale_u, unscale_mu, unscale_blm_betas,
)
from warm_start_real import build_blm_init_real, build_fc_nn_init, build_hs_init

def calibrate_ig_prior(sigma_hat: float, df: int = IG_DF, q: float = IG_Q):
    """Calibrate beta of an InverseGamma(alpha, beta) prior so that the
    prior places its (1 - q)-tail at sigma_hat^2. alpha = df / 2."""
    alpha = df / 2.0
    target = max(float(sigma_hat) ** 2, 1e-10)
    def objective(beta):
        return st.invgamma.cdf(target, a=alpha, scale=beta) - q
    res = root_scalar(objective, bracket=[1e-10, 1e6])
    return alpha, float(res.root)

def save_frontier_outputs(filepath, logy_samples):
    """Save frontier posterior samples on log and original Y scales."""
    logy_np = logy_samples.detach().cpu().numpy()
    y_samples = torch.exp(logy_samples)
    y_np = y_samples.detach().cpu().numpy()
    np.savez(
        filepath,
        logy_samples=logy_np,
        logy_mean=logy_np.mean(0),
        logy_median=np.median(logy_np, axis=0),
        logy_lower95=np.quantile(logy_np, 0.025, axis=0),
        logy_upper95=np.quantile(logy_np, 0.975, axis=0),
        y_samples=y_np,
        y_mean=y_np.mean(0),
        y_median=np.median(y_np, axis=0),
    )


def save_te_outputs(filepath, u_samples):
    """Save TE = exp(-u) posterior samples and per-farm summaries."""
    te_samples = torch.exp(-u_samples)
    te_np = te_samples.detach().cpu().numpy()
    np.savez(
        filepath,
        te_samples=te_np,
        te_mean=te_np.mean(0),
        te_median=np.median(te_np, axis=0),
        te_lower95=np.quantile(te_np, 0.025, axis=0),
        te_upper95=np.quantile(te_np, 0.975, axis=0),
    )
    return te_samples


def _samples_to_idata(mcmc, transform_fn=None):
    """Build an ArviZ InferenceData with proper (chain, draw, ...) dims."""
    grouped = mcmc.get_samples(group_by_chain=True)
    if transform_fn is not None:
        grouped = transform_fn(grouped)
    post = {
        k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
        for k, v in grouped.items()
    }
    return az.from_dict(posterior=post)


def save_pyro_run_real(
    run_dir: str,
    tag: str,
    mcmc,
    model,
    X_std: torch.Tensor,
    stats: Dict[str, float],
    sigma2_u_site: str,
    sigma2_v_site: str,
    u_site: str,
    blm_beta_count: int = 0,
):
    
    raw = mcmc.get_samples()
    samples = {k: v.clone() for k, v in raw.items()}
    samples[sigma2_u_site] = unscale_sigma2(raw[sigma2_u_site], stats["y_std"])
    samples[sigma2_v_site] = unscale_sigma2(raw[sigma2_v_site], stats["y_std"])
    samples[u_site] = unscale_u(raw[u_site], stats["y_std"])
    if "mu" in raw:
        samples["mu"] = unscale_mu(raw["mu"], stats)

    if blm_beta_count > 0:
        beta_dict = {f"beta{j}_blm": raw[f"beta{j}_blm"]
                     for j in range(blm_beta_count)}
        beta_log = unscale_blm_betas(beta_dict, stats)
        for j in range(blm_beta_count):
            samples[f"beta{j}_blm"] = beta_log[f"beta{j}_blm"]

    np.savez(
        os.path.join(run_dir, f"posterior_samples_{tag}.npz"),
        **{k: v.detach().cpu().numpy() for k, v in samples.items()},
    )

    def _transform_grouped(grouped):
        out = dict(grouped)
        out[sigma2_u_site] = unscale_sigma2(grouped[sigma2_u_site], stats["y_std"])
        out[sigma2_v_site] = unscale_sigma2(grouped[sigma2_v_site], stats["y_std"])
        out[u_site] = unscale_u(grouped[u_site], stats["y_std"])
        if "mu" in grouped:
            out["mu"] = unscale_mu(grouped["mu"], stats)
        if blm_beta_count > 0:
            beta_dict_g = {f"beta{j}_blm": grouped[f"beta{j}_blm"]
                           for j in range(blm_beta_count)}
            beta_log_g = unscale_blm_betas(beta_dict_g, stats)
            for j in range(blm_beta_count):
                out[f"beta{j}_blm"] = beta_log_g[f"beta{j}_blm"]
        return out

    idata = _samples_to_idata(mcmc, transform_fn=_transform_grouped)
    az.to_netcdf(idata, os.path.join(run_dir, f"idata_{tag}.nc"))

   
    if "mu" in raw:
        mu_std = raw["mu"]
    else:
        predictive = Predictive(model, posterior_samples=raw, return_sites=["mu"])
        mu_std = predictive(X_std)["mu"]
    mu_logy = unscale_mu(mu_std, stats)

    save_frontier_outputs(os.path.join(run_dir, f"frontier_{tag}.npz"), mu_logy)
    te_samples = save_te_outputs(os.path.join(run_dir, f"te_{tag}.npz"),
                                 samples[u_site])
    return samples, mu_logy, te_samples


def _expand_initial_params_for_chains(initial_params, num_chains: int):
    """Expand warm-start values so each MCMC chain has its own initial point.

    Pyro expects an initial value for every chain when num_chains > 1.
    Scalar sites become shape (num_chains,), vector sites such as u become
    shape (num_chains, n), and matrix sites become shape (num_chains, ...).
    """
    if initial_params is None or num_chains <= 1:
        return initial_params

    expanded = {}
    for name, value in initial_params.items():
        tensor = value.detach().clone() if torch.is_tensor(value) else torch.as_tensor(value)
        # Do not expand again if the first dimension already equals num_chains.
        if tensor.ndim > 0 and tensor.shape[0] == num_chains:
            expanded[name] = tensor.clone()
        else:
            expanded[name] = tensor.unsqueeze(0).expand(num_chains, *tensor.shape).clone()
    return expanded

def run_mcmc_model(model, X, Y, initial_params=None):
    pyro.clear_param_store()
    initial_params = _expand_initial_params_for_chains(initial_params, NUM_CHAINS)
    nuts = NUTS(model, target_accept_prob=TARGET_ACCEPT_PROB,
                max_tree_depth=MAX_TREE_DEPTH)
    mcmc = MCMC(
        nuts,
        num_samples=NUM_SAMPLES,
        warmup_steps=WARMUP_STEPS,
        num_chains=NUM_CHAINS,
        mp_context="spawn", 
        initial_params=initial_params,
        disable_progbar=True,
    )
    mcmc.run(X, Y)
    return mcmc

def posterior_mean_sigma(sigma2_samples) -> float:
    arr = sigma2_samples.detach().cpu().numpy() if hasattr(sigma2_samples, "detach") else np.asarray(sigma2_samples)
    return float(np.mean(np.sqrt(arr)))


def posterior_ci_sigma(sigma2_samples) -> tuple:
    arr = sigma2_samples.detach().cpu().numpy() if hasattr(sigma2_samples, "detach") else np.asarray(sigma2_samples)
    s = np.sqrt(arr)
    return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def posterior_te_summary(te_samples) -> dict:
    """Return (sample-average) posterior-mean TE and its 95% CI plus
    cross-farm mean and median of posterior-mean TE."""
    te = te_samples.detach().cpu().numpy() if hasattr(te_samples, "detach") else np.asarray(te_samples)
    te_per_farm_mean = te.mean(0)           
    avg_te = float(te_per_farm_mean.mean())  
    median_te = float(np.median(te_per_farm_mean))
  
    pooled = te.flatten()
    return {
        "mean_TE": avg_te,
        "median_TE": median_te,
        "TE_lower95": float(np.quantile(pooled, 0.025)),
        "TE_upper95": float(np.quantile(pooled, 0.975)),
    }


def make_summary_row(model_name: str, samples: dict, te_samples,
                      sigma2_u_site: str, sigma2_v_site: str) -> dict:
    sigma_u_mean = posterior_mean_sigma(samples[sigma2_u_site])
    sigma_v_mean = posterior_mean_sigma(samples[sigma2_v_site])
    sigma_u_lo, sigma_u_hi = posterior_ci_sigma(samples[sigma2_u_site])
    sigma_v_lo, sigma_v_hi = posterior_ci_sigma(samples[sigma2_v_site])
    te_summary = posterior_te_summary(te_samples)
    return {
        "Model": model_name,
        "sigma_u_mean": sigma_u_mean,
        "sigma_u_lower95": sigma_u_lo,
        "sigma_u_upper95": sigma_u_hi,
        "sigma_v_mean": sigma_v_mean,
        "sigma_v_lower95": sigma_v_lo,
        "sigma_v_upper95": sigma_v_hi,
        **te_summary,
    }

def main():
    os.makedirs(BASE_RESULTS, exist_ok=True)

    print(f"\n========== Loading {DATA_PATH} ==========")
    print(f"  MCMC chains: {NUM_CHAINS}, samples per chain: {NUM_SAMPLES}, warmup per chain: {WARMUP_STEPS}")
    with open(DATA_PATH, "rb") as f:
        df = pickle.load(f)
    print(f"  rows: {len(df)}, columns: {list(df.columns)[:8]}...")
    print(f"  output: {OUTPUT_VAR}, inputs: {INPUT_VARS}")

    Y_obs = df[OUTPUT_VAR].to_numpy(dtype=float)
    X_obs = df[INPUT_VARS].to_numpy(dtype=float)
    if (Y_obs <= 0).any() or (X_obs <= 0).any():
        raise ValueError("All output and input values must be strictly positive "
                         "for log transformation.")
    logy_obs = np.log(Y_obs)
    logX_obs = np.log(X_obs)
    n_obs = len(logy_obs)
    print(f"  n = {n_obs}")

    df.to_csv(os.path.join(BASE_RESULTS, "data.csv"), index=False)
    np.savez(
        os.path.join(BASE_RESULTS, "data_log.npz"),
        Y=Y_obs, X=X_obs, logY=logy_obs, logX=logX_obs,
        input_names=np.array(INPUT_VARS),
        output_name=np.array(OUTPUT_VAR),
    )

    print("\n========== Multivariate Cobb-Douglas MLE ==========")
    theta, sterr, neg_ll, mle_res = estimate_mle(logy_obs, logX_obs)
    z, p, lo95, hi95 = mle_summary(theta, sterr)

    beta_mle = theta[: N_INPUTS + 1]      
    sigma2u_mle = float(theta[N_INPUTS + 1])
    sigma2v_mle = float(theta[N_INPUTS + 2])
    sigma_u_mle = math.sqrt(sigma2u_mle)
    sigma_v_mle = math.sqrt(sigma2v_mle)

    log_frontier_mle = mu_cobb_multivar(beta_mle, logX_obs)
    eps_mle = logy_obs - log_frontier_mle
    sigma_mle = math.sqrt(sigma2u_mle + sigma2v_mle)
    lam_mle = math.sqrt(sigma2u_mle / sigma2v_mle)
    b_mle = (eps_mle * lam_mle) / sigma_mle
    denom_mle = np.clip(1.0 - norm.cdf(b_mle), 1e-300, None)
    hazard_mle = norm.pdf(b_mle) / denom_mle
    sigma_star_mle = (sigma_u_mle * sigma_v_mle) / sigma_mle
    u_hat_mle = sigma_star_mle * (hazard_mle - b_mle)
    te_hat_mle = np.exp(-u_hat_mle)

    print(f"  beta_hat = {beta_mle}")
    print(f"  sigma_u_hat = {sigma_u_mle:.4f}, sigma_v_hat = {sigma_v_mle:.4f}")
    print(f"  Mean MLE TE = {te_hat_mle.mean():.4f} "
          f"(median {np.median(te_hat_mle):.4f})")

    from scipy.stats import skew as _skew
    resid_skew = float(_skew(logy_obs - log_frontier_mle))
    if resid_skew > 0:
        print(f"  [OSW] residual skew = {resid_skew:+.4f} (POSITIVE)")
        print( "        residual skewness issue detected: the data do not strongly identify the inefficiency component.")
        print( "        sigma_u may remain small across all five models, regardless of prior specification.")
    else:
        print(f"  [OSW] residual skew = {resid_skew:+.4f} (negative, OK)")

    np.savez(
        os.path.join(BASE_RESULTS, "mle_result.npz"),
        theta=theta, sterr=sterr,
        zscores=z, pvalues=p, lower_95=lo95, upper_95=hi95,
        neg_loglik=np.array([neg_ll]),
        beta_mle=beta_mle,
        sigma_u_mle=np.array([sigma_u_mle]),
        sigma_v_mle=np.array([sigma_v_mle]),
        sigma2u_mle=np.array([sigma2u_mle]),
        sigma2v_mle=np.array([sigma2v_mle]),
        log_frontier_mle=log_frontier_mle,
        u_hat_mle=u_hat_mle, te_hat_mle=te_hat_mle,
        input_names=np.array(INPUT_VARS),
    )

    stats = build_std_stats(logX_obs, logy_obs, INPUT_VARS)
    X = standardize_inputs(logX_obs, stats)         
    Y = standardize_output(logy_obs, stats)         

    print("\n========== NN-MLE warm starts ==========")
    mle_relu_result, _ = fit_relu_nn_sfm(
        logX_obs, logy_obs,
        seed=0,
        sigma_u_init=float(sigma_u_mle / stats["y_std"]),
        sigma_v_init=float(sigma_v_mle / stats["y_std"]),
    )
    print(f"  ReLU NN-MLE  (orig log scale): "
          f"sigma_u={mle_relu_result.sigma_u_std * stats['y_std']:.4f}, "
          f"sigma_v={mle_relu_result.sigma_v_std * stats['y_std']:.4f}")

    mle_crelu_result, _ = fit_crelu_nn_sfm(
        logX_obs, logy_obs,
        seed=0,
        sigma_u_init=float(sigma_u_mle / stats["y_std"]),
        sigma_v_init=float(sigma_v_mle / stats["y_std"]),
    )
    print(f"  CReLU NN-MLE (orig log scale): "
          f"sigma_u={mle_crelu_result.sigma_u_std * stats['y_std']:.4f}, "
          f"sigma_v={mle_crelu_result.sigma_v_std * stats['y_std']:.4f}")
    
    sigma_total_std = float(
        np.sqrt(sigma_u_mle ** 2 + sigma_v_mle ** 2) / stats["y_std"]
    )
    sigma_u_floor_std = sigma_total_std

    sigma_u_blm_std = max(float(sigma_u_mle / stats["y_std"]), sigma_u_floor_std)
    sigma_v_blm_std = float(sigma_v_mle / stats["y_std"])

    sigma_u_relu_std = max(float(mle_relu_result.sigma_u_std), sigma_u_floor_std)
    sigma_v_relu_std = float(mle_relu_result.sigma_v_std)

    sigma_u_crelu_std = max(float(mle_crelu_result.sigma_u_std), sigma_u_floor_std)
    sigma_v_crelu_std = float(mle_crelu_result.sigma_v_std)

    print(f"  sigma_u floor (standardized): {sigma_u_floor_std:.4f}")
    print(f"  sigma_u_blm_std   = {sigma_u_blm_std:.4f}")
    print(f"  sigma_u_relu_std  = {sigma_u_relu_std:.4f}")
    print(f"  sigma_u_crelu_std = {sigma_u_crelu_std:.4f}")

    ALPHA_U_BLM, BETA_U_BLM_STD = calibrate_ig_prior(sigma_u_blm_std)
    ALPHA_V_BLM, BETA_V_BLM_STD = calibrate_ig_prior(sigma_v_blm_std)
    ALPHA_U_RELU, BETA_U_RELU_STD = calibrate_ig_prior(sigma_u_relu_std)
    ALPHA_V_RELU, BETA_V_RELU_STD = calibrate_ig_prior(sigma_v_relu_std)
    ALPHA_U_CRELU, BETA_U_CRELU_STD = calibrate_ig_prior(sigma_u_crelu_std)
    ALPHA_V_CRELU, BETA_V_CRELU_STD = calibrate_ig_prior(sigma_v_crelu_std)

    json.dump(
        {
            "input_names": INPUT_VARS,
            "stats": stats,
            "sigma_u_blm_std": sigma_u_blm_std, "sigma_v_blm_std": sigma_v_blm_std,
            "sigma_u_relu_std": sigma_u_relu_std, "sigma_v_relu_std": sigma_v_relu_std,
            "sigma_u_crelu_std": sigma_u_crelu_std, "sigma_v_crelu_std": sigma_v_crelu_std,
        },
        open(os.path.join(BASE_RESULTS, "std_stats.json"), "w"),
        indent=2,
    )

    summaries: List[dict] = []

    print("\n========== Running BLM ==========")
    model_blm = BLM_SFM_NIn(
        n_inputs=N_INPUTS,
        alpha_u=ALPHA_U_BLM, alpha_v=ALPHA_V_BLM,
        beta_u=BETA_U_BLM_STD, beta_v=BETA_V_BLM_STD,
    )
    
    sigma2u_init_blm_log = (sigma_u_blm_std * stats["y_std"]) ** 2
    init_blm = build_blm_init_real(
        beta_log=beta_mle,
        sigma2u_log=sigma2u_init_blm_log,
        sigma2v_log=sigma2v_mle,
        stats=stats,
        n=n_obs,
    )
    mcmc_blm = run_mcmc_model(model_blm, X, Y, initial_params=init_blm)
    samples_blm, mu_blm, te_blm = save_pyro_run_real(
        BASE_RESULTS, "blm", mcmc_blm, model_blm, X, stats,
        sigma2_u_site="sigma2_u_blm",
        sigma2_v_site="sigma2_v_blm",
        u_site="u_blm",
        blm_beta_count=N_INPUTS + 1,
    )
    summaries.append(make_summary_row(
        "BLM (Cobb-Douglas)", samples_blm, te_blm,
        "sigma2_u_blm", "sigma2_v_blm",
    ))

    print("\n========== Running BNN CReLU HalfNormal ==========")
    model_hn = BNN_SFM_CRELU_HN_NIn(
        n_inputs=N_INPUTS,
        alpha_u=ALPHA_U_CRELU, alpha_v=ALPHA_V_CRELU,
        beta_u=BETA_U_CRELU_STD, beta_v=BETA_V_CRELU_STD,
        mle_layers=mle_crelu_result.mle_layers,
    )
    init_hn = build_fc_nn_init(
        model_hn,
        sigma2_u_site="sigma2_u_bnn_crelu_hn",
        sigma2_v_site="sigma2_v_bnn_crelu_hn",
        u_site="u_bnn_crelu_hn",
        sigma2_u_init=sigma_u_crelu_std ** 2,
        sigma2_v_init=sigma_v_crelu_std ** 2,
        n=n_obs,
    )
    mcmc_hn = run_mcmc_model(model_hn, X, Y, initial_params=init_hn)
    samples_hn, mu_hn, te_hn = save_pyro_run_real(
        BASE_RESULTS, "bnn_crelu_hn", mcmc_hn, model_hn, X, stats,
        sigma2_u_site="sigma2_u_bnn_crelu_hn",
        sigma2_v_site="sigma2_v_bnn_crelu_hn",
        u_site="u_bnn_crelu_hn",
    )
    summaries.append(make_summary_row(
        "BNN-CReLU HalfNormal", samples_hn, te_hn,
        "sigma2_u_bnn_crelu_hn", "sigma2_v_bnn_crelu_hn",
    ))

    print("\n========== Running BNN CReLU HalfLaplace ==========")
    model_hl = BNN_SFM_CRELU_HalfLaplace_NIn(
        n_inputs=N_INPUTS,
        alpha_u=ALPHA_U_CRELU, alpha_v=ALPHA_V_CRELU,
        beta_u=BETA_U_CRELU_STD, beta_v=BETA_V_CRELU_STD,
        mle_layers=mle_crelu_result.mle_layers,
    )
    init_hl = build_fc_nn_init(
        model_hl,
        sigma2_u_site="sigma2_u_bnn_crelu_hl",
        sigma2_v_site="sigma2_v_bnn_crelu_hl",
        u_site="u_bnn_crelu_hl",
        sigma2_u_init=sigma_u_crelu_std ** 2,
        sigma2_v_init=sigma_v_crelu_std ** 2,
        n=n_obs,
    )
    mcmc_hl = run_mcmc_model(model_hl, X, Y, initial_params=init_hl)
    samples_hl, mu_hl, te_hl = save_pyro_run_real(
        BASE_RESULTS, "bnn_crelu_hl", mcmc_hl, model_hl, X, stats,
        sigma2_u_site="sigma2_u_bnn_crelu_hl",
        sigma2_v_site="sigma2_v_bnn_crelu_hl",
        u_site="u_bnn_crelu_hl",
    )
    summaries.append(make_summary_row(
        "BNN-CReLU HalfLaplace", samples_hl, te_hl,
        "sigma2_u_bnn_crelu_hl", "sigma2_v_bnn_crelu_hl",
    ))

    print("\n========== Running BNN CReLU Horseshoe ==========")
    model_hs = BNN_SFM_CRELU_HS_NIn(
        n_inputs=N_INPUTS,
        alpha_u=ALPHA_U_CRELU, alpha_v=ALPHA_V_CRELU,
        beta_u=BETA_U_CRELU_STD, beta_v=BETA_V_CRELU_STD,
        mle_layers=mle_crelu_result.mle_layers,
    )
    init_hs = build_hs_init(
        model_hs,
        sigma2_u_site="sigma2_u_hs",
        sigma2_v_site="sigma2_v_hs",
        u_site="u_hs",
        sigma2_u_init=sigma_u_crelu_std ** 2,
        sigma2_v_init=sigma_v_crelu_std ** 2,
        n=n_obs,
    )
    mcmc_hs = run_mcmc_model(model_hs, X, Y, initial_params=init_hs)
    samples_hs, mu_hs, te_hs = save_pyro_run_real(
        BASE_RESULTS, "hs", mcmc_hs, model_hs, X, stats,
        sigma2_u_site="sigma2_u_hs",
        sigma2_v_site="sigma2_v_hs",
        u_site="u_hs",
    )
    summaries.append(make_summary_row(
        "BNN-CReLU Horseshoe", samples_hs, te_hs,
        "sigma2_u_hs", "sigma2_v_hs",
    ))

    print("\n========== Running BNN ReLU ==========")
    model_relu = BNN_SFM_RELU_N_NIn(
        n_inputs=N_INPUTS,
        alpha_u=ALPHA_U_RELU, alpha_v=ALPHA_V_RELU,
        beta_u=BETA_U_RELU_STD, beta_v=BETA_V_RELU_STD,
        mle_layers=mle_relu_result.mle_layers,
    )
    init_relu = build_fc_nn_init(
        model_relu,
        sigma2_u_site="sigma2_u_bnn_relu",
        sigma2_v_site="sigma2_v_bnn_relu",
        u_site="u_bnn_relu",
        sigma2_u_init=sigma_u_relu_std ** 2,
        sigma2_v_init=sigma_v_relu_std ** 2,
        n=n_obs,
    )
    mcmc_relu = run_mcmc_model(model_relu, X, Y, initial_params=init_relu)
    samples_relu, mu_relu, te_relu = save_pyro_run_real(
        BASE_RESULTS, "bnn_relu", mcmc_relu, model_relu, X, stats,
        sigma2_u_site="sigma2_u_bnn_relu",
        sigma2_v_site="sigma2_v_bnn_relu",
        u_site="u_bnn_relu",
    )
    summaries.append(make_summary_row(
        "BNN-ReLU", samples_relu, te_relu,
        "sigma2_u_bnn_relu", "sigma2_v_bnn_relu",
    ))

    mle_summary_row = {
        "Model": "MLE (Cobb-Douglas)",
        "sigma_u_mean": sigma_u_mle,
        "sigma_u_lower95": np.nan,
        "sigma_u_upper95": np.nan,
        "sigma_v_mean": sigma_v_mle,
        "sigma_v_lower95": np.nan,
        "sigma_v_upper95": np.nan,
        "mean_TE": float(te_hat_mle.mean()),
        "median_TE": float(np.median(te_hat_mle)),
        "TE_lower95": np.nan,
        "TE_upper95": np.nan,
    }
    summaries.insert(0, mle_summary_row)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(BASE_RESULTS, "summary.csv"), index=False)
    print("\n========== Summary ==========")
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs in {BASE_RESULTS}/")


if __name__ == "__main__":
    main()
