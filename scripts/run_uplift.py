"""Part 4: analysis of a real randomized e-mail experiment (Hillstrom 2008), following uplift/PREREG.md.

Outputs go to results/uplift/: results.md, results.json, segment_effects.png, qini.png

    python scripts/download_data.py hillstrom
    python scripts/run_uplift.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.uplift.analysis import (  # noqa: E402
    ARM_NAMES, CONTROL, MENS, WOMENS, TLearner, best_arm_policy, bootstrap_policy_values, budget_policy,
    diff_in_means, features, holm, interaction_test, load_hillstrom, policy_value, smd, srm_test,
    stratified_split, summarize_boot,
)

SEED = 42
N_BOOT = 10_000
SRM_ALPHA = 0.001
DATA = ROOT / "data" / "hillstrom" / "hillstrom.csv"
OUT = ROOT / "results" / "uplift"
CANDIDATES = {
    "LogisticRegression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=100, random_state=SEED),
}
FRACS = [round(f, 1) for f in np.arange(0, 1.01, 0.1)]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def pct(x: float, d: int = 2) -> str:
    return f"{100 * x:+.{d}f}%p"


# -------------------------------------------------------------------------------------------------
def quality_checks(df, X, feat_names):
    counts = np.array([(df.arm == a).sum() for a in (CONTROL, MENS, WOMENS)])
    chi2, p = srm_test(counts)
    bal = []
    for j, name in enumerate(feat_names):
        row = {"feature": name}
        for a in (MENS, WOMENS):
            row[ARM_NAMES[a]] = smd(X[df.arm == a, j], X[df.arm == CONTROL, j])
        bal.append(row)
    max_smd = max(abs(r[k]) for r in bal for k in ("Mens", "Womens"))
    return {"counts": counts.tolist(), "chi2": chi2, "p": p, "alarm": p < SRM_ALPHA}, bal, max_smd


def average_effects(df):
    res = {}
    for metric in ("visit", "conversion", "spend"):
        y = df[metric].to_numpy(dtype=float)
        res[metric] = {ARM_NAMES[a]: diff_in_means(y[df.arm == a], y[df.arm == CONTROL]) for a in (MENS, WOMENS)}
    adj = holm([res["visit"]["Mens"]["p"], res["visit"]["Womens"]["p"]])
    res["visit"]["Mens"]["p_holm"], res["visit"]["Womens"]["p_holm"] = float(adj[0]), float(adj[1])
    return res


def matching_interactions(df):
    y, arm = df["visit"].to_numpy(dtype=float), df["arm"].to_numpy()
    r = {"Mens": interaction_test(y, arm, df["mens"].to_numpy(), MENS),
         "Womens": interaction_test(y, arm, df["womens"].to_numpy(), WOMENS)}
    adj = holm([r["Mens"]["p"], r["Womens"]["p"]])
    r["Mens"]["p_holm"], r["Womens"]["p_holm"] = float(adj[0]), float(adj[1])
    return r


def select_model(X, arm, y):
    """5-fold CV on train; score = value of the best-e-mail policy computed out of fold."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for name, make in CANDIDATES.items():
        up = {MENS: np.empty(len(y)), WOMENS: np.empty(len(y))}
        for tr, va in cv.split(X, arm):
            pred = TLearner(make).fit(X[tr], arm[tr], y[tr]).predict_uplift(X[va])
            for a in up:
                up[a][va] = pred[a]
        scores[name] = policy_value(y, arm, best_arm_policy(up))
        log(f"   CV {name}: out-of-fold policy value {scores[name]:.4f}")
    return max(scores, key=scores.get), scores


def post_hoc_causes(df):
    """Post-hoc (NOT pre-registered): why targeting did not help.

    (a) Per purchase group, the effect of each e-mail and Womens - Mens. Choosing the e-mail can only
        pay off in a group where the other e-mail is clearly better.
    (b) Mens e-mail effect by recency and history bins: budget targeting needs the effect to vary a lot.
    """
    y, arm = df["visit"].to_numpy(dtype=float), df["arm"].to_numpy()
    mens, womens = df["mens"].to_numpy(), df["womens"].to_numpy()
    groups = {"mens only": (mens == 1) & (womens == 0), "womens only": (mens == 0) & (womens == 1),
              "both": (mens == 1) & (womens == 1)}
    by_group = []
    for name, m in groups.items():
        by_group.append({"group": name, "n": int(m.sum()),
                         "mens": diff_in_means(y[m & (arm == MENS)], y[m & (arm == CONTROL)]),
                         "womens": diff_in_means(y[m & (arm == WOMENS)], y[m & (arm == CONTROL)]),
                         "womens_minus_mens": diff_in_means(y[m & (arm == WOMENS)], y[m & (arm == MENS)])})
    by_bin = []
    for col, edges, labels in (("recency", [0, 3, 6, 9, 12], ["1-3", "4-6", "7-9", "10-12"]),
                               ("history", [0, 100, 200, 500, np.inf], ["<100", "100-200", "200-500", ">=500"])):
        cut = pd.cut(df[col], edges, labels=labels).to_numpy()
        for lab in labels:
            m = cut == lab
            by_bin.append({"feature": col, "bin": lab, "n": int(m.sum()),
                           "mens": diff_in_means(y[m & (arm == MENS)], y[m & (arm == CONTROL)])})
    return {"by_group": by_group, "by_bin": by_bin}


# -------------------------------------------------------------------------------------------------
def plot_segment_effects(inter, path):
    labels, means, los, his, colors = [], [], [], [], []
    for arm_name, col, color in (("Mens", "mens", "#0072B2"), ("Womens", "womens", "#D55E00")):
        for g, gl in ((1, "bought"), (0, "did not buy")):
            e = inter[arm_name][f"effect_g{g}"]
            labels.append(f"{arm_name} e-mail\n{gl} {col}' items")
            means.append(100 * e["diff"])
            los.append(100 * e["ci_low"])
            his.append(100 * e["ci_high"])
            colors.append(color if g == 1 else "#BBBBBB")
    means, los, his = map(np.array, (means, los, his))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors, yerr=[means - los, his - means], capsize=6)
    for xi, m, h in zip(x, means, his):
        ax.text(xi, h + 0.15, f"{m:+.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Visit-rate lift vs no e-mail (pct. points)")
    ax.set_title("E-mail effect on visits by past purchase category (95% CI)")
    ax.set_ylim(min(0, los.min()) - 0.5, his.max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_qini(qini, single_inc, path):
    fr = np.array(FRACS)
    m = np.array([qini[f]["inc"] for f in FRACS]) * 1000
    lo = np.array([qini[f]["ci_low"] for f in FRACS]) * 1000
    hi = np.array([qini[f]["ci_high"] for f in FRACS]) * 1000
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.fill_between(100 * fr, lo, hi, color="#0072B2", alpha=0.18, lw=0, label="Uplift model, 95% CI")
    ax.plot(100 * fr, m, "o-", color="#0072B2", label="Uplift model (top x% get their better e-mail)")
    ax.plot([0, 100], [0, single_inc * 1000], "--", color="#777777", label="Random x% get the single best e-mail")
    ax.set_xlabel("Customers e-mailed (%)")
    ax.set_ylabel("Extra visits per 1,000 customers")
    ax.set_title("Targeting curve on the test half")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_hillstrom(DATA)
    X = features(df)
    feat_names = ["recency", "history", "mens", "womens", "newbie"] + list(
        pd.get_dummies(df[["zip_code", "channel"]]).columns)

    log("0) Quality checks: SRM and covariate balance")
    srm, balance, max_smd = quality_checks(df, X, feat_names)
    log(f"   counts {srm['counts']} chi2={srm['chi2']:.3f} p={srm['p']:.3f}; max |SMD| = {max_smd:.4f}")
    if srm["alarm"]:
        raise SystemExit("SRM alarm: stop and investigate before analysing outcomes (PREREG section 3).")

    log("1) U1: average effects (full data)")
    ate = average_effects(df)
    for a in ("Mens", "Womens"):
        r = ate["visit"][a]
        log(f"   visit {a}-Control {r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p_holm={fmt_p(r['p_holm'])}")

    log("2) U2: matching interactions (full data)")
    inter = matching_interactions(df)
    for a in ("Mens", "Womens"):
        r = inter[a]
        log(f"   {a}: diff of effects {r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p_holm={fmt_p(r['p_holm'])}")

    log("3) Uplift model: split, CV selection on train, fit, predict test")
    arm, y = df["arm"].to_numpy(), df["visit"].to_numpy(dtype=float)
    tr, te = stratified_split(arm, seed=SEED)
    chosen, cv_scores = select_model(X[tr], arm[tr], y[tr])
    log(f"   chosen: {chosen}")
    model = TLearner(CANDIDATES[chosen]).fit(X[tr], arm[tr], y[tr])
    up = model.predict_uplift(X[te])
    single_arm = MENS if y[tr][arm[tr] == MENS].mean() >= y[tr][arm[tr] == WOMENS].mean() else WOMENS
    log(f"   single best e-mail on train: {ARM_NAMES[single_arm]}")

    n_te = len(te)
    mens_te, womens_te = df["mens"].to_numpy()[te], df["womens"].to_numpy()[te]
    rule = np.where((mens_te == 1) & (womens_te == 0), MENS,
                    np.where((womens_te == 1) & (mens_te == 0), WOMENS, single_arm))
    policies = {
        "model": best_arm_policy(up),
        "single": np.full(n_te, single_arm),
        "top50": budget_policy(up, 0.5),
        "control": np.full(n_te, CONTROL),
        "rule": rule,
    }
    for f in FRACS:
        policies[f"frac_{f}"] = budget_policy(up, f)

    log(f"4) Bootstrap policy values on test ({N_BOOT:,} paired resamples)")
    yt, at = y[te], arm[te]
    point = {k: policy_value(yt, at, p) for k, p in policies.items()}
    boot = bootstrap_policy_values(yt, at, policies, n_boot=N_BOOT, seed=SEED)

    u3_diff = boot["model"] - boot["single"]
    u3 = {"model": point["model"], "single": point["single"], "diff": point["model"] - point["single"],
          **summarize_boot(u3_diff)}
    full_inc_b = boot["single"] - boot["control"]
    ratio_b = (boot["top50"] - boot["control"]) / full_inc_b
    full_inc = point["single"] - point["control"]
    u4 = {"top50": point["top50"], "control": point["control"], "full": point["single"],
          "inc_top50": point["top50"] - point["control"], "inc_full": full_inc,
          "ratio": (point["top50"] - point["control"]) / full_inc,
          **summarize_boot(ratio_b, null=0.5), "full_inc_ci": summarize_boot(full_inc_b)}
    rule_cmp = {"rule": point["rule"], "diff_model_minus_rule": point["model"] - point["rule"],
                **summarize_boot(boot["model"] - boot["rule"]),
                "rule_minus_single": point["rule"] - point["single"],
                "rule_minus_single_ci": summarize_boot(boot["rule"] - boot["single"])}
    qini = {f: {"inc": point[f"frac_{f}"] - point["control"],
                **summarize_boot(boot[f"frac_{f}"] - boot["control"])} for f in FRACS}
    share_womens = float((policies["model"] == WOMENS).mean())
    log(f"   U3 model-single {u3['diff']:+.4f} [{u3['ci_low']:+.4f}, {u3['ci_high']:+.4f}]")
    log(f"   U4 ratio {u4['ratio']:.3f} [{u4['ci_low']:.3f}, {u4['ci_high']:.3f}]")

    log("5) Exploratory: model vs single on conversion and spend (test)")
    secondary = {}
    sec_pols = {"model": policies["model"], "single": policies["single"]}
    for metric in ("conversion", "spend"):
        ym = df[metric].to_numpy(dtype=float)[te]
        b = bootstrap_policy_values(ym, at, sec_pols, n_boot=N_BOOT, seed=SEED)
        secondary[metric] = {"model": policy_value(ym, at, sec_pols["model"]),
                             "single": policy_value(ym, at, sec_pols["single"]),
                             **summarize_boot(b["model"] - b["single"])}
        secondary[metric]["diff"] = secondary[metric]["model"] - secondary[metric]["single"]

    log("6) Post-hoc cause analysis (full data, not pre-registered)")
    causes = post_hoc_causes(df)

    log("7) Writing results")
    plot_segment_effects(inter, OUT / "segment_effects.png")
    plot_qini(qini, full_inc, OUT / "qini.png")
    results = {
        "data": {"n": len(df), "n_train": len(tr), "n_test": len(te)},
        "srm": srm, "max_abs_smd": max_smd, "balance": balance,
        "ate": ate, "interactions": inter,
        "model_selection": {"cv_policy_value": cv_scores, "chosen": chosen,
                            "single_arm_on_train": ARM_NAMES[single_arm]},
        "U3": u3, "U4": u4, "rule": rule_cmp, "qini": {str(k): v for k, v in qini.items()},
        "model_policy_share_womens": share_womens, "secondary_policy": secondary, "post_hoc": causes,
        "runtime_sec": round(time.time() - t0, 1),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results)
    log(f"Done in {time.time() - t0:.0f}s")


def write_markdown(r):
    t = lambda rows, headers: tabulate(rows, headers=headers, tablefmt="github", disable_numparse=True)
    L = ["# Results: Hillstrom e-mail experiment (part 4)", "",
         "Auto-generated by `scripts/run_uplift.py`, following `uplift/PREREG.md`. Seed 42, bootstrap 10,000.", ""]
    s = r["srm"]
    L += ["## 0. Quality checks", "",
          f"- Arm sizes (Control, Mens, Womens): {s['counts']}; SRM chi2 = {s['chi2']:.3f}, p = {s['p']:.3f}"
          f" -> {'ALARM' if s['alarm'] else 'no alarm'} (threshold {SRM_ALPHA})",
          f"- Max |SMD| over all pre-treatment features and both arms: {r['max_abs_smd']:.4f} (threshold 0.1)", "",
          t([[b["feature"], f"{b['Mens']:+.4f}", f"{b['Womens']:+.4f}"] for b in r["balance"]],
            ["feature", "SMD Mens vs Control", "SMD Womens vs Control"]), ""]

    L += ["## 1. Average effects (full data, n = 64,000)", ""]
    rows = []
    for metric in ("visit", "conversion", "spend"):
        for a in ("Mens", "Womens"):
            e = r["ate"][metric][a]
            unit = (lambda v: f"{v:+.4f}") if metric != "spend" else (lambda v: f"{v:+.3f}")
            base = f"{e['mean_c']:.4f}" if metric != "spend" else f"{e['mean_c']:.3f}"
            rows.append([metric, f"{a} - Control", base, unit(e["diff"]), f"[{unit(e['ci_low'])}, {unit(e['ci_high'])}]",
                         f"{e['rel']:+.1%}", fmt_p(e["p"]), fmt_p(e["p_holm"]) if "p_holm" in e else "-"])
    L += [t(rows, ["metric", "comparison", "control mean", "diff", "95% CI", "relative", "p", "p (Holm, U1 only)"]), "",
          "visit is the primary metric (U1). conversion and spend are secondary: no multiplicity correction.", ""]

    L += ["## 2. U2: does the effect depend on past purchase category? (visit, full data)", ""]
    rows = []
    for a, col in (("Mens", "mens"), ("Womens", "womens")):
        i = r["interactions"][a]
        e1, e0 = i["effect_g1"], i["effect_g0"]
        rows.append([f"{a} e-mail", f"{e1['diff']:+.4f} [{e1['ci_low']:+.4f}, {e1['ci_high']:+.4f}] (n_t={e1['n_t']})",
                     f"{e0['diff']:+.4f} [{e0['ci_low']:+.4f}, {e0['ci_high']:+.4f}] (n_t={e0['n_t']})",
                     f"{i['diff']:+.4f} [{i['ci_low']:+.4f}, {i['ci_high']:+.4f}]", fmt_p(i["p"]), fmt_p(i["p_holm"])])
    L += [t(rows, ["e-mail", "effect if bought that category", "effect if not", "difference", "p", "p (Holm)"]), ""]

    ms = r["model_selection"]
    L += ["## 3. Uplift model (train half, 5-fold CV)", "",
          t([[k, f"{v:.4f}"] for k, v in ms["cv_policy_value"].items()], ["candidate", "out-of-fold policy value (visit)"]), "",
          f"Chosen: **{ms['chosen']}**. Single best e-mail on train: **{ms['single_arm_on_train']}**. "
          f"Model policy sends the Womens e-mail to {r['model_policy_share_womens']:.1%} of test customers.", ""]

    u3, u4, ru = r["U3"], r["U4"], r["rule"]
    L += ["## 4. U3 and U4 (test half, n = %d)" % r["data"]["n_test"], "",
          t([["U3: model policy vs single e-mail to all", f"{u3['model']:.4f} vs {u3['single']:.4f}",
              f"{u3['diff']:+.4f}", f"[{u3['ci_low']:+.4f}, {u3['ci_high']:+.4f}]", fmt_p(u3["p_boot"])],
             ["U4: share of full-send increment from top 50%", f"{u4['inc_top50']:.4f} / {u4['inc_full']:.4f}",
              f"{u4['ratio']:.3f}", f"[{u4['ci_low']:.3f}, {u4['ci_high']:.3f}]", fmt_p(u4["p_boot"]) + " (vs 0.5)"]],
            ["hypothesis", "values (visit rate)", "estimate", "95% bootstrap CI", "p"]), "",
          f"- Visit rate with no e-mail (test): {u4['control']:.4f}; full-send increment "
          f"{u4['inc_full']:+.4f} [{u4['full_inc_ci']['ci_low']:+.4f}, {u4['full_inc_ci']['ci_high']:+.4f}]", ""]

    L += ["## 5. Exploratory (not used for decisions)", "",
          t([["model - rule", f"{ru['diff_model_minus_rule']:+.4f}", f"[{ru['ci_low']:+.4f}, {ru['ci_high']:+.4f}]"],
             ["rule - single", f"{ru['rule_minus_single']:+.4f}",
              f"[{ru['rule_minus_single_ci']['ci_low']:+.4f}, {ru['rule_minus_single_ci']['ci_high']:+.4f}]"]]
            + [[f"model - single on {m}", f"{v['diff']:+.4f}", f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"]
               for m, v in r["secondary_policy"].items()],
            ["comparison (test, visit unless noted)", "diff", "95% bootstrap CI"]), "",
          "Rule: Mens e-mail if the customer bought only mens' items, Womens if only womens', otherwise the single best e-mail.", "",
          "### Targeting curve (extra visits vs no e-mail, per customer in the test half)", "",
          t([[f"{int(100 * float(f))}%", f"{q['inc']:+.4f}", f"[{q['ci_low']:+.4f}, {q['ci_high']:+.4f}]",
              f"{float(f) * u4['inc_full']:+.4f}"]
             for f, q in r["qini"].items()],
            ["e-mailed", "increment (model)", "95% CI", "random x% get the single best e-mail"]), "",
          "At 100% the model sends each customer their better e-mail, so it differs from the single-e-mail full send by U3.", ""]
    ci = lambda e: f"{e['diff']:+.4f} [{e['ci_low']:+.4f}, {e['ci_high']:+.4f}]"
    L += ["## 6. Post-hoc: why targeting did not help (full data, NOT pre-registered)", "",
          t([[g["group"], g["n"], ci(g["mens"]), ci(g["womens"]), ci(g["womens_minus_mens"])]
             for g in r["post_hoc"]["by_group"]],
            ["past purchases", "customers", "Mens e-mail effect", "Womens e-mail effect", "Womens - Mens"]), "",
          t([[b["feature"], b["bin"], b["n"], f"{b['mens']['mean_c']:.4f}", ci(b["mens"])]
             for b in r["post_hoc"]["by_bin"]],
            ["feature", "bin", "customers", "control visit rate", "Mens e-mail effect"]), ""]
    (OUT / "results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
