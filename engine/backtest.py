"""Backtest walk-forward: en cada bloque se reentrena solo con el pasado."""
import numpy as np
import pandas as pd

from .elo_system import esperado_a_1x2
from .features import FEATS
from .models import ETIQ, ajustar_dc, entrenar_lr, matriz, predecir_lr, probs_1x2, tasas_dc


def metricas(P, y):
    P = np.clip(P, 1e-9, 1)
    P = P / P.sum(axis=1, keepdims=True)
    oh = np.eye(3)[y]
    return {"Log Loss": -np.mean(np.log(P[np.arange(len(y)), y])),
            "Brier": np.mean(np.sum((P - oh) ** 2, axis=1)),
            "Accuracy": np.mean(P.argmax(axis=1) == y)}


def tabla_calibracion(P, y):
    oh, p = np.eye(3)[y].ravel(), P.ravel()
    cat = pd.cut(p, np.linspace(0, 1, 11), include_lowest=True)
    t = (pd.DataFrame({"p": p, "o": oh, "c": cat}).groupby("c", observed=True)
         .agg(Predicho=("p", "mean"), Observado=("o", "mean"), N=("o", "size")).reset_index(drop=True))
    t["Ideal"] = t["Predicho"]
    return t


def correr_backtest(df, feats, half_life, w_dc, n_test, bloque_dias=21, progreso=None, ventaja_elo=60.0):
    y_all = df["FTR"].map(ETIQ).to_numpy()
    ini = len(df) - n_test
    bloque_id = ((df["Date"] - df["Date"].iloc[ini]).dt.days // bloque_dias).to_numpy()
    ok_train = (feats["n_prev"] >= 5).to_numpy()
    ids = np.unique(bloque_id[ini:])
    P_dc, P_lr, P_base, idxs = [], [], [], []

    for k, b in enumerate(ids):
        te = np.where(bloque_id == b)[0]
        te = te[te >= ini]
        corte = te[0]
        dc = ajustar_dc(df.iloc[:corte], df["Date"].iloc[corte], half_life)
        tr = np.where(ok_train[:corte])[0]
        lr = entrenar_lr(feats.iloc[tr][FEATS], y_all[tr])
        for i in te:
            lam, mu = tasas_dc(dc, df["HomeTeam"].iloc[i], df["AwayTeam"].iloc[i])
            P_dc.append(probs_1x2(matriz(lam, mu, dc["rho"])))
        P_lr.append(predecir_lr(lr, feats.iloc[te][FEATS]))
        P_base.extend([np.bincount(y_all[:corte], minlength=3) / corte] * len(te))
        idxs.extend(te)
        if progreso:
            progreso((k + 1) / len(ids))

    idxs = np.array(idxs)
    P_dc, P_lr = np.array(P_dc), np.vstack(P_lr)
    diff = feats["elo_diff"].to_numpy()[idxs] + ventaja_elo
    return {"y": y_all[idxs], "base": np.array(P_base), "elo": esperado_a_1x2(1 / (1 + 10 ** (-diff / 400))),
            "dc": P_dc, "lr": P_lr, "ens": w_dc * P_dc + (1 - w_dc) * P_lr,
            "mercado": df[["mH", "mD", "mA"]].iloc[idxs].to_numpy()}
