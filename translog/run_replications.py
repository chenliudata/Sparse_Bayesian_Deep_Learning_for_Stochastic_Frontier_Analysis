import os
import json
import math
import numpy as np
import pandas as pd
import torch
import pyro
import arviz as az
import scipy.stats as st

from scipy.stats import norm
from scipy.optimize import root_scalar
from pyro.infer import MCMC, NUTS, Predictive

from config_translog import (
    SEEDS,
    NUM_SAMPLES,
    WARMUP_STEPS,
    NUM_CHAINS,
    TARGET_ACCEPT_PROB,
    MAX_TREE_DEPTH,
    BETA_TRUE,
    SIGMA_U_TRUE,
    SIGMA_V_TRUE,
    BASE_RESULTS,
    N,
    IG_DF,
    IG_Q,
    MLE_RELU_EPOCHS,
    MLE_CRELU_EPOCHS,
)
from simulation import simulate_translog_2in, mu_translog
from mle import estimate_mle, mle_summary, mu_cobb
from mle_nn import fit_relu_nn_sfm, fit_crelu_nn_sfm
from model_blm import BLM_Cobb_2In
from model_bnn_relu import BNN_SFM_RELU_2In
from model_bnn_crelu import BNN_SFM_CRELU_2In
from model_hs import BNN_SFM_CRELU_HS_Node_2In
from model_laplace import BNN_SFM_CRELU_LAPLACE_2In
from standardize import (
    build_std_stats,
    standardize_inputs_2in,
    standardize_output,
    unscale_sigma2,
    unscale_u,
    unscale_mu,
    unscale_blm_betas_cobb,
)
from warm_start import build_blm_init, build_fc_nn_init, build_hs_init

def calibrate_ig_prior(sigma_hat: float, df: int = IG_DF, q: float = IG_Q):
    alpha = df / 2.0
    target = max(float(sigma_hat) ** 2, 1e-10)
    def objective(beta):
        return st.invgamma.cdf(target, a=alpha, scale=beta) - q
    res = root_scalar(objective, bracket=[1e-10, 1e6])
    return alpha, float(res.root)

def save_frontier_outputs(filepath, logy_samples):
   
    logy_np = logy_samples.detach().cpu().numpy()
    y_samples = torch.exp(logy_samples)
    y_np = y_samples.detach().cpu().numpy()
    np.savez(
        filepath,
        logy_samples=logy_np,
        logy_mean=logy_np.mean(0),
        logy_median=np.median(logy_np, axis=0),
        y_samples=y_np,
        y_mean=y_np.mean(0),
        y_median=np.median(y_np, axis=0),
    )

def save_te_outputs(filepath, u_samples):
    te_samples = torch.exp(-u_samples)
    te_np = te_samples.detach().cpu().numpy()
    np.savez(
        filepath,
        te_samples=te_np,
        te_mean=te_np.mean(0),
        te_median=np.median(te_np, axis=0),
    )
    return te_samples

def _samples_to_idata(mcmc, transform_fn=None):
    """Build ArviZ InferenceData."""
    grouped = mcmc.get_samples(group_by_chain=True)
    if transform_fn is not None:
        grouped = transform_fn(grouped)
    post = {
        k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
        for k, v in grouped.items()
    }
    return az.from_dict(posterior=post)

def save_pyro_run_std(run_dir, tag, mcmc, model, X_std, stats, beta_names=False):
    raw = mcmc.get_samples()
    samples = {k: v.clone() for k, v in raw.items()}
    samples["sigma2_u"] = unscale_sigma2(raw["sigma2_u"], stats["y_std"])
    samples["sigma2_v"] = unscale_sigma2(raw["sigma2_v"], stats["y_std"])
    samples["u"] = unscale_u(raw["u"], stats["y_std"])
    
    if "mu" in raw:
        samples["mu"] = unscale_mu(raw["mu"], stats)

    if beta_names:
        b0, b1, b2 = unscale_blm_betas_cobb(
            raw["beta0"], raw["beta1"], raw["beta2"],
            stats,
        )
        samples["beta0"] = b0
        samples["beta1"] = b1
        samples["beta2"] = b2

    np.savez(
        os.path.join(run_dir, f"posterior_samples_{tag}.npz"),
        **{k: v.detach().cpu().numpy() for k, v in samples.items()},
    )

    def _transform_grouped(grouped):
        out = dict(grouped)
        out["sigma2_u"] = unscale_sigma2(grouped["sigma2_u"], stats["y_std"])
        out["sigma2_v"] = unscale_sigma2(grouped["sigma2_v"], stats["y_std"])
        out["u"] = unscale_u(grouped["u"], stats["y_std"])
        if "mu" in grouped:
            out["mu"] = unscale_mu(grouped["mu"], stats)
        if beta_names:
            b0, b1, b2 = unscale_blm_betas_cobb(
                grouped["beta0"], grouped["beta1"], grouped["beta2"], stats,
            )
            out["beta0"] = b0
            out["beta1"] = b1
            out["beta2"] = b2
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
    te_samples = save_te_outputs(os.path.join(run_dir, f"te_{tag}.npz"), samples["u"])
    return samples, mu_logy, te_samples

def run_mcmc_model(model, X, Y, initial_params=None):
    pyro.clear_param_store()
    nuts_kernel = NUTS(model, target_accept_prob=TARGET_ACCEPT_PROB, max_tree_depth=MAX_TREE_DEPTH)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=NUM_SAMPLES,
        warmup_steps=WARMUP_STEPS,
        num_chains=NUM_CHAINS,
        initial_params=initial_params,
    )
    mcmc.run(X, Y)
    return mcmc

def rmse(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return np.sqrt(np.mean((a - b) ** 2))

def mae_scalar(est, true):
    return float(np.abs(est - true))

def crps(samples, y_true):
    """Empirical CRPS using sorted posterior samples."""
    samples = np.asarray(samples)
    y_true = np.asarray(y_true)

    if samples.ndim == 1:
        s = np.sort(samples)
        n = len(s)
        t1 = np.mean(np.abs(s - y_true))
        t2 = (s @ (2 * np.arange(n) - n + 1)) / (n * n)
        return float(t1 - t2)

    if samples.ndim == 2:
        n, m = samples.shape
        if y_true.ndim != 1 or y_true.shape[0] != m:
            raise ValueError(
                f"For 2D samples of shape {samples.shape}, y_true must be 1D "
                f"of length {m}, got shape {y_true.shape}."
            )
        s_sorted = np.sort(samples, axis=0)            # (n, m)
        coeffs = (2 * np.arange(n) - n + 1) / (n * n)  # (n,)
        t1 = np.mean(np.abs(samples - y_true[None, :]), axis=0)  # (m,)
        t2 = coeffs @ s_sorted                          # (m,)
        return float((t1 - t2).mean())

    raise ValueError(f"Unsupported samples shape: {samples.shape}")

def make_metric_row(
    seed, N, model_name,
    logy_hat, te_hat,
    sigma_u_hat, sigma_v_hat,
    log_frontier_true, te_true,
    mae_te, crps_te, crps_sigma_u, crps_sigma_v,
):
    return {
        "seed": seed,
        "N": N,
        "Model": model_name,
        "RMSE_logY": rmse(logy_hat, log_frontier_true),
        "RMSE_TE": rmse(te_hat, te_true),
        "MAE_TE": mae_te,
        "MAE_sigma_u": mae_scalar(sigma_u_hat, SIGMA_U_TRUE),
        "MAE_sigma_v": mae_scalar(sigma_v_hat, SIGMA_V_TRUE),
        "CRPS_TE": crps_te,
        "CRPS_sigma_u": crps_sigma_u,
        "CRPS_sigma_v": crps_sigma_v,
    }

def posterior_mean_sigma(sigma2_samples):
   
    arr = sigma2_samples.detach().cpu().numpy() if hasattr(sigma2_samples, "detach") else np.asarray(sigma2_samples)
    return float(np.mean(np.sqrt(arr)))

def main():
    os.makedirs(BASE_RESULTS, exist_ok=True)
    all_results = []
    beta_true = np.array(BETA_TRUE, dtype=float)

    for i, seed in enumerate(SEEDS, start=1):
        print(f"\n========== Replication {i}/{len(SEEDS)} | seed={seed} ==========")
        run_dir = os.path.join(BASE_RESULTS, f"seed{seed}")
        os.makedirs(run_dir, exist_ok=True)

        np.random.seed(seed)
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)

        df = simulate_translog_2in(
            N=N, beta=beta_true,
            sigma_u=SIGMA_U_TRUE, sigma_v=SIGMA_V_TRUE,
            seed=seed,
        )
        df.to_csv(os.path.join(run_dir, "data.csv"), index=False)

        x1_obs = df["X1"].to_numpy()
        x2_obs = df["X2"].to_numpy()
        logx1_obs = df["logx1"].to_numpy()
        logx2_obs = df["logx2"].to_numpy()
        y_obs = df["Y"].to_numpy()
        logy_obs = df["logy"].to_numpy()

        log_frontier_true = mu_translog(beta_true, logx1_obs, logx2_obs)
        frontier_true = np.exp(log_frontier_true)
        log_eps_true = logy_obs - log_frontier_true

        sigma_true = np.sqrt(SIGMA_U_TRUE**2 + SIGMA_V_TRUE**2)
        lam_true = SIGMA_U_TRUE / SIGMA_V_TRUE
        b_true = (log_eps_true * lam_true) / sigma_true
        denom_true = np.clip(1.0 - norm.cdf(b_true), 1e-300, None)
        hazard_true = norm.pdf(b_true) / denom_true
        sigma_star_true = (SIGMA_U_TRUE * SIGMA_V_TRUE) / sigma_true
        u_true = sigma_star_true * (hazard_true - b_true)
        te_true = np.exp(-u_true)

        np.savez(
            os.path.join(run_dir, "true_data.npz"),
            x1_obs=x1_obs, x2_obs=x2_obs,
            logx1_obs=logx1_obs, logx2_obs=logx2_obs,
            y_obs=y_obs, logy_obs=logy_obs,
            log_frontier_true=log_frontier_true,
            frontier_true=frontier_true,
            log_eps_true=log_eps_true,
            te_true=te_true,
        )

        with open(os.path.join(run_dir, "true_params.json"), "w") as f:
            json.dump(
                {
                    "N": N, "seed": seed,
                    "beta_true": beta_true.tolist(),
                    "sigma_u_true": SIGMA_U_TRUE,
                    "sigma_v_true": SIGMA_V_TRUE,
                },
                f, indent=2,
            )

        stats = build_std_stats(logx1_obs, logx2_obs, logy_obs)
        X = standardize_inputs_2in(logx1_obs, logx2_obs, stats)
        Y = standardize_output(logy_obs, stats)

        theta, sterr, neg_loglik, mle_results = estimate_mle(logy_obs, logx1_obs, logx2_obs)
        zscores, pvalues, lower_95, upper_95 = mle_summary(theta, sterr)

        beta_mle = theta[:3]
        sigma2u_mle = theta[3]
        sigma2v_mle = theta[4]
        sigma_u_mle = np.sqrt(sigma2u_mle)
        sigma_v_mle = np.sqrt(sigma2v_mle)

        log_frontier_mle = mu_cobb(beta_mle, logx1_obs, logx2_obs)
        eps_mle = logy_obs - log_frontier_mle
        sigma_mle = np.sqrt(sigma2u_mle + sigma2v_mle)
        lam_mle = np.sqrt(sigma2u_mle / sigma2v_mle)
        b_mle = (eps_mle * lam_mle) / sigma_mle
        denom_mle = np.clip(1.0 - norm.cdf(b_mle), 1e-300, None)
        hazard_mle = norm.pdf(b_mle) / denom_mle
        sigma_star_mle = (sigma_u_mle * sigma_v_mle) / sigma_mle
        u_hat_mle = sigma_star_mle * (hazard_mle - b_mle)
        te_hat_mle = np.exp(-u_hat_mle)

        np.savez(
            os.path.join(run_dir, "mle_result.npz"),
            theta=theta, sterr=sterr, converged=np.array([mle_results.success]),
            beta0_mle=np.array([beta_mle[0]]),
            beta1_mle=np.array([beta_mle[1]]),
            beta2_mle=np.array([beta_mle[2]]),
            sigma_u_mle=np.array([sigma_u_mle]), sigma_v_mle=np.array([sigma_v_mle]),
            sigma2u_mle=np.array([sigma2u_mle]), sigma2v_mle=np.array([sigma2v_mle]),
            zscores=zscores, pvalues=pvalues, lower_95=lower_95, upper_95=upper_95,
            neg_loglik=np.array([neg_loglik]), log_frontier_mle=log_frontier_mle, te_hat_mle=te_hat_mle,
        )

        all_results.append(make_metric_row(
            seed, N, "MLE",
            log_frontier_mle, te_hat_mle,
            sigma_u_mle, sigma_v_mle,
            log_frontier_true, te_true,
            np.mean(np.abs(te_hat_mle - te_true)),
            np.nan, np.nan, np.nan,
        ))

        sigma_u_true_std = float(SIGMA_U_TRUE / stats["y_std"])
        sigma_v_true_std = float(SIGMA_V_TRUE / stats["y_std"])

        mle_relu_result, _ = fit_relu_nn_sfm(
            logx1_obs, logx2_obs, logy_obs,
            seed=seed,
            sigma_u_init=sigma_u_true_std,
            sigma_v_init=sigma_v_true_std,
        )
        np.savez(
            os.path.join(run_dir, "mle_nn_relu.npz"),
            sigma_u_std=mle_relu_result.sigma_u_std,
            sigma_v_std=mle_relu_result.sigma_v_std,
            sigma_u_init_std=sigma_u_true_std,
            sigma_v_init_std=sigma_v_true_std,
            max_epochs=MLE_RELU_EPOCHS,
            best_epoch=mle_relu_result.best_epoch,
            epochs_run=mle_relu_result.epochs_run,
            loss_history=np.asarray(mle_relu_result.loss_history),
            activation=np.array([mle_relu_result.activation]),
            **{f"W{j+1}": w.numpy() for j, (w, _) in enumerate(mle_relu_result.mle_layers)},
            **{f"b{j+1}": b.numpy() for j, (_, b) in enumerate(mle_relu_result.mle_layers)},
            **stats,
        )

        mle_crelu_result, _ = fit_crelu_nn_sfm(
            logx1_obs, logx2_obs, logy_obs,
            seed=seed,
            sigma_u_init=sigma_u_true_std,
            sigma_v_init=sigma_v_true_std,
        )
        np.savez(
            os.path.join(run_dir, "mle_nn_crelu.npz"),
            sigma_u_std=mle_crelu_result.sigma_u_std,
            sigma_v_std=mle_crelu_result.sigma_v_std,
            sigma_u_init_std=sigma_u_true_std,
            sigma_v_init_std=sigma_v_true_std,
            max_epochs=MLE_CRELU_EPOCHS,
            best_epoch=mle_crelu_result.best_epoch,
            epochs_run=mle_crelu_result.epochs_run,
            loss_history=np.asarray(mle_crelu_result.loss_history),
            activation=np.array([mle_crelu_result.activation]),
            **{f"W{j+1}": w.numpy() for j, (w, _) in enumerate(mle_crelu_result.mle_layers)},
            **{f"b{j+1}": b.numpy() for j, (_, b) in enumerate(mle_crelu_result.mle_layers)},
            **stats,
        )

        sigma_u_blm_std = float(sigma_u_mle / stats["y_std"])
        sigma_v_blm_std = float(sigma_v_mle / stats["y_std"])
        sigma_u_relu_std = float(mle_relu_result.sigma_u_std)
        sigma_v_relu_std = float(mle_relu_result.sigma_v_std)
        sigma_u_crelu_std = float(mle_crelu_result.sigma_u_std)
        sigma_v_crelu_std = float(mle_crelu_result.sigma_v_std)

        ALPHA_U_BLM, BETA_U_BLM_STD = calibrate_ig_prior(sigma_u_blm_std, df=IG_DF, q=IG_Q)
        ALPHA_V_BLM, BETA_V_BLM_STD = calibrate_ig_prior(sigma_v_blm_std, df=IG_DF, q=IG_Q)
        ALPHA_U_RELU, BETA_U_RELU_STD = calibrate_ig_prior(sigma_u_relu_std, df=IG_DF, q=IG_Q)
        ALPHA_V_RELU, BETA_V_RELU_STD = calibrate_ig_prior(sigma_v_relu_std, df=IG_DF, q=IG_Q)
        ALPHA_U_CRELU, BETA_U_CRELU_STD = calibrate_ig_prior(sigma_u_crelu_std, df=IG_DF, q=IG_Q)
        ALPHA_V_CRELU, BETA_V_CRELU_STD = calibrate_ig_prior(sigma_v_crelu_std, df=IG_DF, q=IG_Q)

        np.savez(
            os.path.join(run_dir, "std_stats.npz"),
            **stats,
            ALPHA_U_BLM=ALPHA_U_BLM, ALPHA_V_BLM=ALPHA_V_BLM,
            BETA_U_BLM_std=BETA_U_BLM_STD, BETA_V_BLM_std=BETA_V_BLM_STD,
            ALPHA_U_RELU=ALPHA_U_RELU, ALPHA_V_RELU=ALPHA_V_RELU,
            BETA_U_RELU_std=BETA_U_RELU_STD, BETA_V_RELU_std=BETA_V_RELU_STD,
            ALPHA_U_CRELU=ALPHA_U_CRELU, ALPHA_V_CRELU=ALPHA_V_CRELU,
            BETA_U_CRELU_std=BETA_U_CRELU_STD, BETA_V_CRELU_std=BETA_V_CRELU_STD,
            IG_DF=IG_DF, IG_Q=IG_Q,
            sigma_u_blm_std=sigma_u_blm_std, sigma_v_blm_std=sigma_v_blm_std,
            sigma_u_relu_std=sigma_u_relu_std, sigma_v_relu_std=sigma_v_relu_std,
            sigma_u_crelu_std=sigma_u_crelu_std, sigma_v_crelu_std=sigma_v_crelu_std,
            sigma_u_nn_init_std=sigma_u_true_std, sigma_v_nn_init_std=sigma_v_true_std,
        )

        n_obs = Y.numel()

        print(
        f"deterministic MLE (original log scale): "
        f"sigma_u_hat={sigma_u_mle:.4f}, sigma_v_hat={sigma_v_mle:.4f}"
        )

        print(
            f"ReLU NN-MLE (original log scale): "
            f"sigma_u_hat={sigma_u_relu_std * stats['y_std']:.4f}, "
            f"sigma_v_hat={sigma_v_relu_std * stats['y_std']:.4f}"
        )

        print(
            f"CReLU NN-MLE (original log scale): "
            f"sigma_u_hat={sigma_u_crelu_std * stats['y_std']:.4f}, "
            f"sigma_v_hat={sigma_v_crelu_std * stats['y_std']:.4f}"
        )

        print("Running BLM")
        model_blm = BLM_Cobb_2In(
            alpha_u=ALPHA_U_BLM, alpha_v=ALPHA_V_BLM,
            beta_u=BETA_U_BLM_STD, beta_v=BETA_V_BLM_STD,
        )
        init_blm = build_blm_init(beta_mle, sigma2u_mle, sigma2v_mle, stats, n_obs)
        mcmc_blm = run_mcmc_model(model_blm, X, Y, initial_params=init_blm)
        samples_blm, mu_blm, te_blm = save_pyro_run_std(
            run_dir, "blm", mcmc_blm, model_blm, X, stats, beta_names=True,
        )
        all_results.append(make_metric_row(
            seed, N, "BLM",
            mu_blm.mean(0).detach().cpu().numpy(),
            te_blm.mean(0).detach().cpu().numpy(),
            posterior_mean_sigma(samples_blm["sigma2_u"]),
            posterior_mean_sigma(samples_blm["sigma2_v"]),
            log_frontier_true, te_true,
            np.mean(np.abs(te_blm.mean(0).detach().cpu().numpy() - te_true)),
            crps(te_blm.detach().cpu().numpy(), te_true),
            crps(np.sqrt(samples_blm["sigma2_u"].detach().cpu().numpy()), SIGMA_U_TRUE),
            crps(np.sqrt(samples_blm["sigma2_v"].detach().cpu().numpy()), SIGMA_V_TRUE),
        ))

        print("Running BNN ReLU")
        model_relu = BNN_SFM_RELU_2In(
            alpha_u=ALPHA_U_RELU, alpha_v=ALPHA_V_RELU,
            beta_u=BETA_U_RELU_STD, beta_v=BETA_V_RELU_STD,
            mle_layers=mle_relu_result.mle_layers,
        )
        init_relu = build_fc_nn_init(
            model_relu, sigma_u_relu_std ** 2, sigma_v_relu_std ** 2, n_obs,
        )
        mcmc_relu = run_mcmc_model(model_relu, X, Y, initial_params=init_relu)
        samples_relu, mu_relu, te_relu = save_pyro_run_std(
            run_dir, "bnn_relu", mcmc_relu, model_relu, X, stats,
        )
        all_results.append(make_metric_row(
            seed, N, "ReLU Normal",
            mu_relu.mean(0).detach().cpu().numpy(),
            te_relu.mean(0).detach().cpu().numpy(),
            posterior_mean_sigma(samples_relu["sigma2_u"]),
            posterior_mean_sigma(samples_relu["sigma2_v"]),
            log_frontier_true, te_true,
            np.mean(np.abs(te_relu.mean(0).detach().cpu().numpy() - te_true)),
            crps(te_relu.detach().cpu().numpy(), te_true),
            crps(np.sqrt(samples_relu["sigma2_u"].detach().cpu().numpy()), SIGMA_U_TRUE),
            crps(np.sqrt(samples_relu["sigma2_v"].detach().cpu().numpy()), SIGMA_V_TRUE),
        ))

        print("Running CReLU HalfNormal")
        model_crelu = BNN_SFM_CRELU_2In(
            alpha_u=ALPHA_U_CRELU, alpha_v=ALPHA_V_CRELU,
            beta_u=BETA_U_CRELU_STD, beta_v=BETA_V_CRELU_STD,
            mle_layers=mle_crelu_result.mle_layers,
        )
        init_crelu = build_fc_nn_init(
            model_crelu, sigma_u_crelu_std ** 2, sigma_v_crelu_std ** 2, n_obs,
        )
        mcmc_crelu = run_mcmc_model(model_crelu, X, Y, initial_params=init_crelu)
        samples_crelu, mu_crelu, te_crelu = save_pyro_run_std(
            run_dir, "bnn_crelu", mcmc_crelu, model_crelu, X, stats,
        )
        all_results.append(make_metric_row(
            seed, N, "CReLU HalfNormal",
            mu_crelu.mean(0).detach().cpu().numpy(),
            te_crelu.mean(0).detach().cpu().numpy(),
            posterior_mean_sigma(samples_crelu["sigma2_u"]),
            posterior_mean_sigma(samples_crelu["sigma2_v"]),
            log_frontier_true, te_true,
            np.mean(np.abs(te_crelu.mean(0).detach().cpu().numpy() - te_true)),
            crps(te_crelu.detach().cpu().numpy(), te_true),
            crps(np.sqrt(samples_crelu["sigma2_u"].detach().cpu().numpy()), SIGMA_U_TRUE),
            crps(np.sqrt(samples_crelu["sigma2_v"].detach().cpu().numpy()), SIGMA_V_TRUE),
        ))

        print("Running CReLU HalfLaplace")
        model_laplace = BNN_SFM_CRELU_LAPLACE_2In(
            alpha_u=ALPHA_U_CRELU, alpha_v=ALPHA_V_CRELU,
            beta_u=BETA_U_CRELU_STD, beta_v=BETA_V_CRELU_STD,
            mle_layers=mle_crelu_result.mle_layers,
        )
        init_laplace = build_fc_nn_init(
            model_laplace, sigma_u_crelu_std ** 2, sigma_v_crelu_std ** 2, n_obs,
        )
        mcmc_laplace = run_mcmc_model(model_laplace, X, Y, initial_params=init_laplace)
        samples_laplace, mu_laplace, te_laplace = save_pyro_run_std(
            run_dir, "laplace", mcmc_laplace, model_laplace, X, stats,
        )
        all_results.append(make_metric_row(
            seed, N, "CReLU HalfLaplace",
            mu_laplace.mean(0).detach().cpu().numpy(),
            te_laplace.mean(0).detach().cpu().numpy(),
            posterior_mean_sigma(samples_laplace["sigma2_u"]),
            posterior_mean_sigma(samples_laplace["sigma2_v"]),
            log_frontier_true, te_true,
            np.mean(np.abs(te_laplace.mean(0).detach().cpu().numpy() - te_true)),
            crps(te_laplace.detach().cpu().numpy(), te_true),
            crps(np.sqrt(samples_laplace["sigma2_u"].detach().cpu().numpy()), SIGMA_U_TRUE),
            crps(np.sqrt(samples_laplace["sigma2_v"].detach().cpu().numpy()), SIGMA_V_TRUE),
        ))

        print("Running CReLU Horseshoe")
        model_hs = BNN_SFM_CRELU_HS_Node_2In(
            alpha_u=ALPHA_U_CRELU, alpha_v=ALPHA_V_CRELU,
            beta_u=BETA_U_CRELU_STD, beta_v=BETA_V_CRELU_STD,
            mle_layers=mle_crelu_result.mle_layers,
        )
        init_hs = build_hs_init(model_hs, sigma_u_crelu_std ** 2, sigma_v_crelu_std ** 2, n_obs)
        mcmc_hs = run_mcmc_model(model_hs, X, Y, initial_params=init_hs)
        samples_hs, mu_hs, te_hs = save_pyro_run_std(
            run_dir, "hs", mcmc_hs, model_hs, X, stats,
        )
        all_results.append(make_metric_row(
            seed, N, "CReLU Horseshoe",
            mu_hs.mean(0).detach().cpu().numpy(),
            te_hs.mean(0).detach().cpu().numpy(),
            posterior_mean_sigma(samples_hs["sigma2_u"]),
            posterior_mean_sigma(samples_hs["sigma2_v"]),
            log_frontier_true, te_true,
            np.mean(np.abs(te_hs.mean(0).detach().cpu().numpy() - te_true)),
            crps(te_hs.detach().cpu().numpy(), te_true),
            crps(np.sqrt(samples_hs["sigma2_u"].detach().cpu().numpy()), SIGMA_U_TRUE),
            crps(np.sqrt(samples_hs["sigma2_v"].detach().cpu().numpy()), SIGMA_V_TRUE),
        ))

        pd.DataFrame(all_results).to_csv(
            os.path.join(BASE_RESULTS, "all_results_running.csv"), index=False,
        )

    pd.DataFrame(all_results).to_csv(os.path.join(BASE_RESULTS, "all_results.csv"), index=False)
    print("\nAll replications completed.")

if __name__ == "__main__":
    main()
