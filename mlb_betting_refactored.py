"""
FASE 3: PIPELINE INTEGRADO REFACTORIZADO
=========================================

Orquestación end-to-end que integra:
  1. features_advanced.py → Ingeniería avanzada (Pythagorean, Log5, ERA/WHIP)
  2. models_xgboost.py → Entrenamiento XGBoost + Confidence Scoring
  3. Cálculo EV+ y Kelly Criterion con umbral de confianza

Reemplaza completamente mlb_betting.py anterior.

Uso:
    python mlb_betting_refactored.py [YYYY-MM-DD]

Referencias Científicas:
  - Zhao et al. (2024): XGBoost supera RF en deportes
  - Bae et al. (2026): Confidence scoring >70% logra precisión >70%
  - Lo et al. (2025): Log5 + Pythagorean = mejor predicción H2H
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

# Importar módulos de Fase 1 y Fase 2
from features_advanced import (
    pythagorean_expectation,
    log5_probability,
    home_field_advantage_weighted,
    calculate_pitcher_metrics,
    calculate_bullpen_metrics,
    build_advanced_features,
)

from models_xgboost import (
    ConfidenceScoringXGBClassifier,
    CalibratedXGBRegressor,
    MLBModelTrainer,
    CONFIDENCE_THRESHOLD_MONEYLINE,
    CONFIDENCE_THRESHOLD_PROPS,
)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

SEASON = 2026
DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "output"
CACHE_TTL = 6 * 3600  # 6 horas

WINDOWS = [3, 5, 10]

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"

HITTING = ["runs", "hits", "doubles", "triples", "homeRuns", "totalBases",
           "strikeOuts", "baseOnBalls", "walk", "atBats"]

PITCHING = ["runs", "hits", "strikeOuts", "baseOnBalls", "homeRuns",
            "inningsPitched", "era", "whip"]

# Targets: Solo los que vamos a predecir
TARGETS = [
    "hom_runs", "away_runs", "total_runs",
    "hom_hits", "away_hits",
    "hom_total_bases", "away_total_bases",
    "hom_k", "away_k",
    "hom_p_k", "away_p_k",
    "total_k",
    "hom_win",  # ← IMPORTANTE: Moneyline con Confidence Scoring
]

API_BASE = "https://statsapi.mlb.com/api/v1"

# Park Factors (2025/2026 Baseline)
PARK_FACTORS = {
    "Colorado Rockies": 1.35,
    "Boston Red Sox": 1.08,
    "Cincinnati Reds": 1.12,
    "Philadelphia Phillies": 1.06,
    "Kansas City Royals": 1.05,
    "Chicago Cubs": 1.04,
    "Baltimore Orioles": 1.02,
    "Texas Rangers": 1.02,
    "Los Angeles Dodgers": 1.01,
    "Atlanta Braves": 1.01,
    "Minnesota Twins": 1.00,
    "Chicago White Sox": 1.00,
    "St. Louis Cardinals": 0.99,
    "Milwaukee Brewers": 0.99,
    "New York Yankees": 0.98,
    "Houston Astros": 0.98,
    "Toronto Blue Jays": 0.98,
    "Arizona Diamondbacks": 0.97,
    "Washington Nationals": 0.97,
    "Los Angeles Angels": 0.96,
    "San Francisco Giants": 0.95,
    "Detroit Tigers": 0.95,
    "Pittsburgh Pirates": 0.95,
    "New York Mets": 0.94,
    "Cleveland Guardians": 0.94,
    "Tampa Bay Rays": 0.93,
    "Miami Marlins": 0.93,
    "San Diego Padres": 0.92,
    "Seattle Mariners": 0.91,
    "Athletics": 0.96,
}


# ============================================================================
# CLIENTE MLB STATS API
# ============================================================================

def _get(url: str, params: Dict[str, Any] | None = None) -> Dict:
    """Solicitud HTTP con reintentos."""
    for intento in range(4):
        try:
            r = requests.get(url, params=params, timeout=45)
            if r.status_code == 429:
                time.sleep(10 * (intento + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if intento == 3:
                return {}
            time.sleep(10 * (intento + 1))
    return {}


def _leer_cache(nombre: str) -> pd.DataFrame | None:
    """Lee cache si existe y es válido."""
    ruta = os.path.join(DATA_DIR, nombre)
    if os.path.exists(ruta) and (time.time() - os.path.getmtime(ruta)) < CACHE_TTL:
        try:
            return pd.read_csv(ruta)
        except Exception:
            return None
    return None


def _guardar_cache(nombre: str, df: pd.DataFrame) -> None:
    """Guarda DataFrame en cache."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, nombre), index=False)


def _parsear_calendario(payload: Dict) -> pd.DataFrame:
    """Parsea respuesta de calendario MLB."""
    juegos = []
    for dia in payload.get("dates", []):
        for g in dia.get("games", []):
            local = g.get("teams", {}).get("home", {})
            visit = g.get("teams", {}).get("away", {})

            hom_p = local.get("probablePitcher", {}).get("fullName", "TBD")
            away_p = visit.get("probablePitcher", {}).get("fullName", "TBD")
            hom_p_id = local.get("probablePitcher", {}).get("id", None)
            away_p_id = visit.get("probablePitcher", {}).get("id", None)

            juegos.append({
                "game_pk": g.get("gamePk"),
                "game_date": (g.get("gameDate") or "")[:10],
                "status": g.get("status", {}).get("abstractGameState"),
                "hom_team_id": local.get("team", {}).get("id"),
                "away_team_id": visit.get("team", {}).get("id"),
                "hom_team_name": local.get("team", {}).get("name"),
                "away_team_name": visit.get("team", {}).get("name"),
                "hom_pitcher": hom_p,
                "away_pitcher": away_p,
                "hom_pitcher_id": hom_p_id,
                "away_pitcher_id": away_p_id,
                "hom_score": local.get("score"),
                "away_score": visit.get("score"),
            })
    return pd.DataFrame(juegos)


def calendario_temporada() -> pd.DataFrame:
    """Obtiene calendario completo de la temporada."""
    cache = f"calendario_{SEASON}.csv"
    df = _leer_cache(cache)
    if df is not None:
        return df
    payload = _get(
        f"{API_BASE}/schedule",
        {"sportId": 1, "season": SEASON, "gameType": "R", "hydrate": "probablePitcher"},
    )
    df = _parsear_calendario(payload)
    if not df.empty:
        _guardar_cache(cache, df)
    return df


def calendario_dia(fecha: str) -> pd.DataFrame:
    """Obtiene calendario para un día específico."""
    cache = f"calendario_{fecha}.csv"
    df = _leer_cache(cache)
    if df is not None:
        return df
    payload = _get(
        f"{API_BASE}/schedule",
        {"sportId": 1, "date": fecha, "hydrate": "probablePitcher"},
    )
    df = _parsear_calendario(payload)
    if not df.empty:
        _guardar_cache(cache, df)
    return df


def game_log_equipo(team_id: int, grupo: str) -> pd.DataFrame:
    """Obtiene game log de un equipo (hitting/pitching)."""
    cache = f"equipo_{team_id}_{SEASON}_{grupo}.csv"
    df = _leer_cache(cache)
    if df is not None:
        return df
    payload = _get(
        f"{API_BASE}/teams/{team_id}/stats",
        {"season": SEASON, "group": grupo, "stats": "gameLog"},
    )
    filas = []
    for bloque in payload.get("stats", []):
        for split in bloque.get("splits", []):
            fila = {
                "date": (split.get("date") or "")[:10],
                "team_id": split.get("team", {}).get("id"),
                "opponent_id": split.get("opponent", {}).get("id"),
            }
            fila.update(split.get("stat", {}))
            filas.append(fila)
    df = pd.DataFrame(filas)
    if not df.empty:
        _guardar_cache(cache, df)
    return df


def obtener_cuotas_odds_api(fecha: str) -> Dict[str, Dict[str, float]]:
    """Obtiene cuotas en vivo de The Odds API."""
    if not ODDS_API_KEY:
        return {}

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,totals",
        "oddsFormat": "american",
        "date": fecha,
    }

    try:
        r = requests.get(ODDS_API_URL, params=params, timeout=15)
        if r.status_code != 200:
            return {}

        data = r.json()
        cuotas = {}

        if not isinstance(data, list):
            return cuotas

        for game in data:
            home = game.get("home_team")
            away = game.get("away_team")

            if not home or not away:
                continue

            key = f"{away} @ {home}"
            bookmakers = game.get("bookmakers", [])

            if not bookmakers:
                continue

            bm = bookmakers[0]
            markets_list = bm.get("markets", [])
            markets = {m["key"]: m for m in markets_list}

            # H2H
            h2h = markets.get("h2h", {})
            h2h_outcomes = {
                o["name"]: o.get("price", 0.0)
                for o in h2h.get("outcomes", [])
            }
            h_odds = h2h_outcomes.get(home, 0.0)
            a_odds = h2h_outcomes.get(away, 0.0)

            # Totals
            totals = markets.get("totals", {})
            total_line = 0.0
            for outcome in totals.get("outcomes", []):
                if "point" in outcome:
                    total_line = float(outcome["point"])
                    break

            cuotas[key] = {
                "hom_ml_odds": float(h_odds),
                "away_ml_odds": float(a_odds),
                "total_line": float(total_line),
            }

        return cuotas

    except Exception:
        return {}


# ============================================================================
# UTILIDADES NUMÉRICAS
# ============================================================================

def a_float(x, por_defecto=0.0):
    """Convierte a float de forma segura."""
    if x is None:
        return por_defecto
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return por_defecto
    try:
        return float(x)
    except (TypeError, ValueError):
        return por_defecto


def parsear_innings(x):
    """Parsea innings (ej: 6.1 = 6 + 1/3)."""
    if isinstance(x, str) and "." in x:
        entero, _, resto = x.partition(".")
        try:
            return float(entero) + {"0": 0.0, "1": 1 / 3, "2": 2 / 3}.get(resto[:1], 0.0)
        except ValueError:
            pass
    return a_float(x)


def normalizar_log(raw: pd.DataFrame) -> pd.DataFrame:
    """Normaliza game log."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "team_id", "opponent_id"])
    df = df.sort_values("date").reset_index(drop=True)
    df["team_id"] = df["team_id"].astype(int)
    df["opponent_id"] = df["opponent_id"].astype(int)

    for c in HITTING + PITCHING:
        if c in df.columns:
            df[c] = df[c].apply(a_float)

    if "inningsPitched" in df.columns:
        df["inningsPitched"] = df["inningsPitched"].apply(parsear_innings)
        ip = df["inningsPitched"].replace(0, np.nan)
        df["k_per_9"] = df["strikeOuts"] / (ip / 9)
        df["whip_game"] = (df["hits"] + df["baseOnBalls"]) / ip
        df[["k_per_9", "whip_game"]] = df[["k_per_9", "whip_game"]].fillna(0.0)

    if "atBats" in df.columns:
        df["avg_game"] = df["hits"] / df["atBats"].replace(0, np.nan)
        df["avg_game"] = df["avg_game"].fillna(0.0)

    return df


def características_equipo(df: pd.DataFrame, prefijo: str = "") -> pd.DataFrame:
    """Calcula características tradicionales (medias móviles)."""
    if df.empty:
        return pd.DataFrame()

    cols = [c for c in HITTING + PITCHING + ["k_per_9", "whip_game", "avg_game"]
            if c in df.columns]

    feat: Dict[str, Any] = {
        "date": df["date"].values,
        "team_id": df["team_id"].values,
        "opponent_id": df["opponent_id"].values,
    }
    for c in cols:
        for w in WINDOWS:
            feat[f"{prefijo}{c}_avg{w}"] = df[c].shift(1).rolling(w, min_periods=1).mean().values
        feat[f"{prefijo}{c}_season"] = df[c].expanding().mean().shift(1).values
        ult5 = df[c].shift(1).rolling(5, min_periods=1).mean()
        prev10 = df[c].shift(6).rolling(10, min_periods=1).mean()
        feat[f"{prefijo}{c}_trend"] = (ult5 - prev10).values
    return pd.DataFrame(feat)


def ultimas_características(frame: pd.DataFrame, fecha: pd.Timestamp) -> Dict:
    """Extrae últimas características disponibles antes de una fecha."""
    if frame is None or frame.empty:
        return {}
    previos = frame[frame["date"] < fecha]
    if previos.empty:
        return {}
    return previos.sort_values("date").iloc[-1].to_dict()


def dias_descanso(df: pd.DataFrame, fecha: pd.Timestamp) -> float:
    """Calcula días de descanso desde el último juego."""
    if df is None or df.empty:
        return 0.0
    previos = df[df["date"] < fecha]
    if previos.empty:
        return 0.0
    return max(0.0, (fecha - previos["date"].max()).days)


def cargar_equipo(team_id: int) -> Dict:
    """Carga hitting + pitching + features para un equipo."""
    hitting_raw = game_log_equipo(team_id, "hitting")
    pitching_raw = game_log_equipo(team_id, "pitching")

    hitting = normalizar_log(hitting_raw)
    pitching = normalizar_log(pitching_raw)

    hf = características_equipo(hitting, prefijo="h_")
    pf = características_equipo(pitching, prefijo="p_")

    if hf.empty:
        combinado = pf
    elif pf.empty:
        combinado = hf
    else:
        combinado = pd.merge(hf, pf, on=["date", "team_id", "opponent_id"], how="outer")

    return {"hitting": hitting, "pitching": pitching, "features": combinado}


def cargar_equipos(calendario: pd.DataFrame) -> Dict:
    """Carga datos de todos los equipos en calendario."""
    ids = pd.unique(calendario[["hom_team_id", "away_team_id"]].values.ravel())
    equipos = {}
    for tid in ids:
        if pd.isna(tid):
            continue
        try:
            equipos[int(tid)] = cargar_equipo(int(tid))
        except Exception:
            pass
    return equipos


# ============================================================================
# CONSTRUCCIÓN DATASET DE ENTRENAMIENTO
# ============================================================================

def construir_datos_entrenamiento() -> pd.DataFrame:
    """Construye dataset histórico para entrenamiento."""
    print("  📥 Cargando calendario histórico...")
    programa = calendario_temporada()
    finales = programa[programa["status"] == "Final"].copy()
    finales = finales.dropna(subset=["hom_score", "away_score"])

    if finales.empty:
        print("  ⚠️ Sin juegos finalizados. Abortando entrenamiento.")
        return pd.DataFrame()

    print(f"  ✅ {len(finales)} juegos finales encontrados")

    print("  📥 Cargando datos de equipos (hitting/pitching)...")
    equipos = cargar_equipos(finales)

    print("  🔧 Construyendo características...")
    filas = []
    for _, g in finales.iterrows():
        gdate = pd.Timestamp(g["game_date"])
        hid, aid = int(g["hom_team_id"]), int(g["away_team_id"])
        pf = PARK_FACTORS.get(g["hom_team_name"], 1.00)

        # Características tradicionales
        feat = {
            "game_date": gdate,
            "park_factor": pf,
            "hom_team_id": hid,
            "away_team_id": aid,
            "hom_pitcher_id": g.get("hom_pitcher_id"),
            "away_pitcher_id": g.get("away_pitcher_id"),
        }

        for lado, tid in [("hom", hid), ("away", aid)]:
            eq = equipos.get(tid)
            if not eq:
                continue
            for k, v in ultimas_características(eq["features"], gdate).items():
                if k in ("date", "team_id", "opponent_id"):
                    continue
                feat[f"{lado}_{k}"] = v
            feat[f"{lado}_descanso"] = dias_descanso(eq["hitting"], gdate)

        # Características AVANZADAS (Fase 1)
        try:
            advanced_feats = build_advanced_features(
                g,
                equipos.get(hid, {}),
                equipos.get(aid, {}),
                {},  # recent_performance vacío para histórico
                {},  # pitcher_db vacío
                {},  # bullpen_db vacío
            )
            feat.update(advanced_feats)
        except Exception as e:
            print(f"    ⚠️ Error calculando características avanzadas: {e}")

        # Targets
        feat["hom_runs"] = a_float(g["hom_score"])
        feat["away_runs"] = a_float(g["away_score"])
        feat["total_runs"] = feat["hom_runs"] + feat["away_runs"]
        feat["hom_win"] = 1 if feat["hom_runs"] > feat["away_runs"] else 0

        filas.append(feat)

    df = pd.DataFrame(filas)
    if df.empty:
        return df

    # Limpiar features faltantes
    cols_feat = [c for c in df.columns 
                 if c not in TARGETS and c != "game_date" 
                 and c not in ["hom_team_id", "away_team_id", "hom_pitcher_id", "away_pitcher_id"]]
    df = df.dropna(subset=cols_feat, how="all")

    print(f"  ✅ Dataset construido: {len(df)} juegos, {len(cols_feat)} features")

    return df


# ============================================================================
# ENTRENAMIENTO DE MODELOS (FASE 2)
# ============================================================================

def entrenar_modelos_fase2(df: pd.DataFrame) -> MLBModelTrainer:
    """Entrena suite completa de XGBoost + Confidence Scoring."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if len(df) < 50:
        print("  ⚠️ Insuficientes datos para entrenar (< 50 juegos).")
        return None

    df = df.sort_values("game_date").reset_index(drop=True)
    
    # Identificar columns de features (excluir targets y metadata)
    cols_features = [c for c in df.columns 
                     if c not in TARGETS 
                     and c not in ["game_date", "hom_team_id", "away_team_id", 
                                   "hom_pitcher_id", "away_pitcher_id"]]
    
    print(f"  ✅ {len(cols_features)} features identificadas para entrenamiento")

    # Entrenar modelos
    trainer = MLBModelTrainer(model_dir=MODEL_DIR)
    X = df[cols_features].fillna(0.0)
    
    # Usar solo targets disponibles
    targets_disponibles = [t for t in TARGETS if t in df.columns]
    
    metrics = trainer.train_all_targets(
        df,
        targets=targets_disponibles,
        feature_cols=cols_features,
        verbose=True
    )

    # Guardar modelos
    trainer.save_all_models()

    return trainer


# ============================================================================
# PREDICCIÓN DIARIA CON CONFIDENCE SCORING
# ============================================================================

def american_to_decimal(american_odds: float) -> float:
    """Convierte odds americanos a decimales."""
    if american_odds > 0:
        return (american_odds / 100.0) + 1.0
    elif american_odds < 0:
        return (100.0 / abs(american_odds)) + 1.0
    return 1.0


def predecir_dia(
    fecha: str | None = None,
    trainer: MLBModelTrainer | None = None
) -> pd.DataFrame:
    """Predice para un día específico con Confidence Scoring."""
    fecha = fecha or datetime.date.today().strftime("%Y-%m-%d")

    print(f"\n{'='*70}")
    print(f"🔮 PREDICCIONES PARA {fecha}")
    print(f"{'='*70}")

    juegos = calendario_dia(fecha)

    if juegos.empty:
        print("  ⚠️ Sin juegos programados.")
        return pd.DataFrame()

    proximos = juegos[juegos["status"].isin(
        ["Preview", "Scheduled", "Pre-Game", "Warmup", "In-Progress", "Live"]
    )]

    if proximos.empty:
        print("  ⚠️ Sin juegos en estado válido.")
        return pd.DataFrame()

    print(f"  ✅ {len(proximos)} juegos encontrados")

    equipos = cargar_equipos(proximos)
    cuotas_api = obtener_cuotas_odds_api(fecha)

    predicciones = []
    gdate = pd.Timestamp(fecha)

    for idx, g in proximos.iterrows():
        hid, aid = g["hom_team_id"], g["away_team_id"]
        if pd.isna(hid) or pd.isna(aid):
            continue

        hid, aid = int(hid), int(aid)
        pf = PARK_FACTORS.get(g["hom_team_name"], 1.00)

        # Características tradicionales
        feat = {
            "park_factor": pf,
            "hom_team_id": hid,
            "away_team_id": aid,
            "hom_pitcher_id": g.get("hom_pitcher_id"),
            "away_pitcher_id": g.get("away_pitcher_id"),
        }

        for lado, tid in [("hom", hid), ("away", aid)]:
            eq = equipos.get(tid)
            if not eq:
                continue
            for k, v in ultimas_características(eq["features"], gdate).items():
                if k in ("date", "team_id", "opponent_id"):
                    continue
                feat[f"{lado}_{k}"] = v
            feat[f"{lado}_descanso"] = dias_descanso(eq["hitting"], gdate)

        # Características AVANZADAS
        try:
            advanced_feats = build_advanced_features(
                g,
                equipos.get(hid, {}),
                equipos.get(aid, {}),
                {},
                {},
                {},
            )
            feat.update(advanced_feats)
        except Exception as e:
            print(f"    ⚠️ Error en features avanzadas: {e}")

        row_pred = {
            "fecha": fecha,
            "matchup": f"{g['away_team_name']} @ {g['hom_team_name']}",
            "hom_team": g["hom_team_name"],
            "away_team": g["away_team_name"],
            "hom_pitcher": g["hom_pitcher"],
            "away_pitcher": g["away_pitcher"],
            "park_factor": pf,
        }

        # ========== INFERENCIA CON MODELOS XGBOOST ==========
        
        if trainer and trainer.models:
            # Moneyline con Confidence Scoring
            if "hom_win" in trainer.models:
                clf = trainer.models["hom_win"]
                feat_df = pd.DataFrame([feat])
                result = clf.predict_with_confidence(feat_df)
                
                row_pred["ml_local_%"] = result["prob_home_win"] * 100
                row_pred["ml_visitante_%"] = result["prob_away_win"] * 100
                row_pred["confidence_moneyline"] = result["confidence"] * 100
                row_pred["high_confidence"] = result["high_confidence"]
                row_pred["moneyline_rec"] = result["recommendation"]

            # Regresiones para propiedades
            regression_targets = [t for t in TARGETS if t != "hom_win" and t in trainer.models]
            for target in regression_targets:
                reg = trainer.models[target]
                result = reg.predict_with_interval(feat_df, confidence=0.90)
                row_pred[f"{target}_pred"] = result["prediction"]
                row_pred[f"{target}_lower"] = result["lower_bound"]
                row_pred[f"{target}_upper"] = result["upper_bound"]
        else:
            # Sin modelos disponibles
            row_pred["ml_local_%"] = 50.0
            row_pred["ml_visitante_%"] = 50.0
            row_pred["confidence_moneyline"] = 0.0
            row_pred["high_confidence"] = False
            row_pred["moneyline_rec"] = "❌ Sin modelos disponibles"

        # ========== CÁLCULO EV+ Y KELLY CRITERION ==========
        
        match_key = f"{g['away_team_name']} @ {g['hom_team_name']}"
        odds_info = cuotas_api.get(match_key, {})

        cuota_h_am = odds_info.get("hom_ml_odds", 0.0)
        cuota_a_am = odds_info.get("away_ml_odds", 0.0)

        cuota_h_dec = american_to_decimal(cuota_h_am) if cuota_h_am != 0.0 else 0.0
        cuota_a_dec = american_to_decimal(cuota_a_am) if cuota_a_am != 0.0 else 0.0

        row_pred["cuota_local"] = cuota_h_dec if cuota_h_dec > 1.0 else "—"
        row_pred["cuota_visitante"] = cuota_a_dec if cuota_a_dec > 1.0 else "—"

        # EV+ solo si alta confianza (ZHAO 2024, BAE 2026)
        p_local = row_pred.get("ml_local_%", 50.0) / 100.0
        p_away = row_pred.get("ml_visitante_%", 50.0) / 100.0
        is_high_conf = row_pred.get("high_confidence", False)

        ev_h = (p_local * cuota_h_dec) - 1.0 if cuota_h_dec > 1.0 else -1.0
        ev_a = (p_away * cuota_a_dec) - 1.0 if cuota_a_dec > 1.0 else -1.0

        # IMPORTANTE: Solo genera picks si HIGH CONFIDENCE
        if is_high_conf and ev_h > 0.03 and ev_h > ev_a and cuota_h_dec > 1.0:
            b = cuota_h_dec - 1.0
            kelly = max(0.0, ((b * p_local) - (1 - p_local)) / b) * 0.25
            row_pred["pick_ev"] = f"✅ LOCAL (+EV {ev_h:.1%})"
            row_pred["stake_kelly"] = f"{kelly:.1%}"
        elif is_high_conf and ev_a > 0.03 and ev_a > ev_h and cuota_a_dec > 1.0:
            b = cuota_a_dec - 1.0
            kelly = max(0.0, ((b * p_away) - (1 - p_away)) / b) * 0.25
            row_pred["pick_ev"] = f"✅ VISITANTE (+EV {ev_a:.1%})"
            row_pred["stake_kelly"] = f"{kelly:.1%}"
        else:
            row_pred["pick_ev"] = "⚠️ Sin valor claro o baja confianza"
            row_pred["stake_kelly"] = "0%"

        predicciones.append(row_pred)

    df_pred = pd.DataFrame(predicciones)
    
    # Reporte de picks EV+
    picks_ev = df_pred[df_pred["pick_ev"].str.contains("✅", na=False)]
    print(f"\n  📊 PICKS CON EV+ (Confianza Alta): {len(picks_ev)}/{len(df_pred)}")
    for _, pick in picks_ev.iterrows():
        print(f"     • {pick['matchup']}: {pick['pick_ev']} | Stake: {pick['stake_kelly']}")

    return df_pred


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

def main():
    """Orquestación completa del pipeline."""
    fecha_target = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().strftime("%Y-%m-%d")

    print(f"\n{'='*70}")
    print(f"🚀 MLB BETTING REFACTORED - PIPELINE CIENTÍFICO")
    print(f"{'='*70}")
    print(f"Fecha objetivo: {fecha_target}")
    print(f"Temporada: {SEASON}")
    print(f"{'='*70}\n")

    # ========== FASE 1: CONSTRUCCIÓN DE DATOS ==========
    print("📊 FASE 1: CONSTRUCCIÓN DE DATASET DE ENTRENAMIENTO")
    print("-" * 70)
    df_train = construir_datos_entrenamiento()

    if df_train.empty:
        print("❌ Dataset vacío. Abortando.")
        sys.exit(1)

    # ========== FASE 2: ENTRENAMIENTO XGBOOST ==========
    print("\n🤖 FASE 2: ENTRENAMIENTO XGBOOST + TIME-SERIES SPLIT")
    print("-" * 70)
    trainer = entrenar_modelos_fase2(df_train)

    if trainer is None:
        print("❌ Error en entrenamiento. Abortando.")
        sys.exit(1)

    # ========== FASE 3: PREDICCIÓN DIARIA ==========
    print("\n📈 FASE 3: PREDICCIÓN DIARIA + CONFIDENCE SCORING")
    print("-" * 70)
    df_res = predecir_dia(fecha_target, trainer)

    if df_res.empty:
        print("⚠️ Sin predicciones para generar.")
        return

    # ========== SALIDA ==========
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_out = os.path.join(OUTPUT_DIR, f"predicciones_refactored_{fecha_target}.csv")
    df_res.to_csv(file_out, index=False)

    print(f"\n{'='*70}")
    print(f"✅ PROCESO COMPLETADO CON ÉXITO")
    print(f"{'='*70}")
    print(f"Archivo guardado: {file_out}")
    print(f"Total juegos analizados: {len(df_res)}")
    print(f"Picks EV+ generados: {(df_res['pick_ev'].str.contains('✅', na=False)).sum()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
