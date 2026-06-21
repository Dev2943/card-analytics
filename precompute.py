"""
precompute.py — runs at Docker BUILD time.
Fetches the UCI data, runs the full pipeline (ETL → segmentation → model → KPIs),
and saves the results to results.json so the live app loads instantly with NO
runtime fetch or model training.
"""
import json
import numpy as np
import data_pipeline as dp
import segmentation as seg
import default_model as dm
import portfolio_kpis as pk


def main():
    print("Loading + cleaning + engineering...")
    df = dp.load_raw()
    df = dp.clean(df)
    df = dp.engineer_features(df)

    print("Segmenting...")
    X_scaled = seg.scale_features(df)
    k_values, inertias, silhouettes = seg.choose_k(X_scaled)
    best_k = k_values[int(np.argmax(silhouettes))]
    df = seg.fit_segments(df, X_scaled, best_k)
    profile = seg.profile_segments(df)
    names = seg.suggest_names(profile)
    df["segment_name"] = df["segment"].map(names)

    print("Training default models...")
    X_train, X_test, y_train, y_test = dm.split_data(df)
    logreg, gb, scaler = dm.train_models(X_train, y_train)
    res_lr = dm.evaluate("LogReg", y_test, logreg.predict_proba(scaler.transform(X_test))[:, 1])
    res_gb = dm.evaluate("GradBoost", y_test, gb.predict_proba(X_test)[:, 1])

    X_all = scaler.transform(df[dm.MODEL_FEATURES].values)
    df["pred_prob_default"] = logreg.predict_proba(X_all)[:, 1]

    print("Computing KPIs...")
    df = pk.build_churn_proxy(df)
    df = pk.compute_expected_loss(df)
    df = pk.compute_revenue(df)
    kpis = pk.segment_kpis(df)

    # Profile with names as index
    profile2 = profile.copy()
    profile2.index = [f"{i} — {names.get(i, '')}" for i in profile2.index]

    results = {
        "total_customers": int(len(df)),
        "default_rate": float(df["default"].mean()),
        "total_expected_loss": float(df["expected_loss"].sum()),
        "total_revenue": float(df["monthly_revenue"].sum()),
        "auc_lr": float(res_lr["roc_auc"]),
        "auc_gb": float(res_gb["roc_auc"]),
        "k_values": [int(k) for k in k_values],
        "inertias": [float(x) for x in inertias],
        "silhouettes": [float(x) for x in silhouettes],
        "profile": profile2.reset_index().to_dict(orient="records"),
        "kpis": kpis.reset_index().to_dict(orient="records"),
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results.json — {len(df)} customers, default rate {df['default'].mean():.1%}")


if __name__ == "__main__":
    main()
