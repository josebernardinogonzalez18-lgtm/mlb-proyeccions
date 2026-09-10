"""
FASE 1: INGENIERÍA DE CARACTERÍSTICAS AVANZADA (Feature Engineering)
=====================================================================

Basado en el consenso científico (Zhao 2024, Bae 2026, Lo 2025):
Las medias móviles simples son INSUFICIENTES. Los modelos con >70% de precisión
dependen de:
  1. Expectativa Pitagórica de Victorias (Pythagorean Expectation)
  2. Log5: Probabilidad estimada de que equipo A venza a B
  3. Ventaja de jugar en casa ponderada por rendimiento reciente
  4. Métricas de eficiencia acumuladas para abridores y bullpens (ERA, WHIP, RD)

Importado por: models_xgboost.py, mlb_betting_refactored.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any


# ============================================================================
# MÓDULO 1: MÉTRICA PITAGÓRICA DE VICTORIAS
# ============================================================================

def pythagorean_expectation(runs_for: float, runs_against: float, exponent: float = 1.83) -> float:
    """
    Calcula la Expectativa Pitagórica de Victorias (Pythagorean Expectation).
    
    Fórmula de Bill James:
        Win% = RF^exp / (RF^exp + RA^exp)
    
    Args:
        runs_for: Carreras anotadas por el equipo (RF)
        runs_against: Carreras permitidas por el equipo (RA)
        exponent: Exponente de Pithagorean (1.83 por defecto, ajustable)
    
    Returns:
        float: Porcentaje esperado de victorias (0.0 - 1.0)
    
    Referencias:
        - James, B. (1980). "The Bill James Baseball Abstract"
        - Pinto, D. (2010). "The Book" — Exponente actualizado a 1.83
    """
    if runs_for <= 0 or runs_against <= 0:
        return 0.5  # Neutral si no hay datos
    
    rf_exp = runs_for ** exponent
    ra_exp = runs_against ** exponent
    
    pythag_win_pct = rf_exp / (rf_exp + ra_exp)
    
    return np.clip(pythag_win_pct, 0.0, 1.0)


# ============================================================================
# MÓDULO 2: LOG5 — PROBABILIDAD H2H (HEAD-TO-HEAD)
# ============================================================================

def log5_probability(team_a_win_pct: float, team_b_win_pct: float) -> float:
    """
    Calcula la probabilidad de que el Equipo A venza a B usando Log5.
    
    Log5 es la proyección de Davenport-Clay que estima la probabilidad
    de victoria basada en los porcentajes de victorias históricos de ambos equipos.
    
    Fórmula:
        P(A beats B) = [P(A) * (1 - P(B))] / {[P(A) * (1 - P(B))] + [(1 - P(A)) * P(B)]}
    
    Args:
        team_a_win_pct: Porcentaje de victorias del Equipo A (0.0 - 1.0)
        team_b_win_pct: Porcentaje de victorias del Equipo B (0.0 - 1.0)
    
    Returns:
        float: Probabilidad de que A venza a B (0.0 - 1.0)
    
    Referencias:
        - Davenport, C., & Clay, D. (2002). "Baseball Prospectus"
        - Tango, T., Lichtman, M., & Dolphin, A. (2007). "The Book"
    """
    if team_a_win_pct <= 0 or team_a_win_pct >= 1.0:
        team_a_win_pct = np.clip(team_a_win_pct, 0.001, 0.999)
    if team_b_win_pct <= 0 or team_b_win_pct >= 1.0:
        team_b_win_pct = np.clip(team_b_win_pct, 0.001, 0.999)
    
    numerator = team_a_win_pct * (1 - team_b_win_pct)
    denominator = numerator + ((1 - team_a_win_pct) * team_b_win_pct)
    
    if denominator == 0:
        return 0.5
    
    log5_prob = numerator / denominator
    
    return np.clip(log5_prob, 0.0, 1.0)


# ============================================================================
# MÓDULO 3: VENTAJA DE JUGAR EN CASA (HOME-FIELD ADVANTAGE) PONDERADA
# ============================================================================

def home_field_advantage_weighted(
    home_team_id: int,
    away_team_id: int,
    recent_performance: Dict[int, Dict[str, float]],
    baseline_hfa: float = 0.054
) -> Tuple[float, float]:
    """
    Calcula la ventaja de jugar en casa ponderada por rendimiento reciente.
    
    En MLB, la ventaja de jugar en casa es típicamente ~5.4% (Baseline).
    Pero varía según la fuerza relativa de los equipos en los últimos 15-20 juegos.
    
    Estrategia:
        1. Home WinPct (últimos 20 juegos) vs. Away WinPct (últimos 20 juegos)
        2. Ajustar HFA basado en la diferencia relativa
        3. Equipos en racha en casa obtienen MAYOR ventaja
    
    Args:
        home_team_id: ID del equipo local
        away_team_id: ID del equipo visitante
        recent_performance: Dict con estructura:
            {
                team_id: {
                    "home_win_pct": float,
                    "away_win_pct": float,
                    "games_last_20": int
                }
            }
        baseline_hfa: Ventaja de campo base (default: 5.4% = 0.054)
    
    Returns:
        Tuple[float, float]: (home_adjusted_multiplier, away_adjusted_multiplier)
                            Multiplicadores a aplicar a probabilidades
    
    Referencias:
        - Verducci, T. (2014). "Home Field Advantage in Baseball"
        - Studeman, D. (2009). "The Book"
    """
    home_data = recent_performance.get(home_team_id, {
        "home_win_pct": 0.500,
        "away_win_pct": 0.500,
        "games_last_20": 0
    })
    away_data = recent_performance.get(away_team_id, {
        "home_win_pct": 0.500,
        "away_win_pct": 0.500,
        "games_last_20": 0
    })
    
    # Si no hay suficiente data reciente, usar baseline
    if home_data["games_last_20"] < 3 or away_data["games_last_20"] < 3:
        return 1.0 + baseline_hfa, 1.0 - baseline_hfa
    
    # Diferencia de rendimiento en casa vs. fuera
    home_strength = home_data["home_win_pct"] - 0.500  # ±0.5 vs. .500
    away_weakness = away_data["away_win_pct"] - 0.500
    
    # Ajuste dinámico de HFA
    # Equipos que juegan bien en casa + equipos que juegan mal fuera = HFA mayor
    adjusted_hfa = baseline_hfa * (1 + home_strength + abs(away_weakness))
    adjusted_hfa = np.clip(adjusted_hfa, baseline_hfa * 0.5, baseline_hfa * 2.5)
    
    home_mult = 1.0 + adjusted_hfa
    away_mult = 1.0 - adjusted_hfa
    
    return home_mult, away_mult


# ============================================================================
# MÓDULO 4: MÉTRICAS DE EFICIENCIA PARA ABRIDORES Y BULLPENS
# ============================================================================

def calculate_pitcher_metrics(
    pitcher_df: pd.DataFrame,
    pitcher_id: int,
    window: int = 20
) -> Dict[str, float]:
    """
    Extrae métricas de eficiencia acumuladas para un lanzador.
    
    Calcula:
      - ERA: Carreras limpias permitidas por 9 innings
      - WHIP: (Hits + Walks) / Innings Pitched
      - K/9: Strikeouts por 9 innings
      - K/BB: Proporción de Ks vs. Walks (Control)
      - Run Differential: Carreras anotadas - permitidas en apariciones
    
    Args:
        pitcher_df: DataFrame con game logs del pitcher (sorted by date)
        pitcher_id: ID del pitcher
        window: Ventana de juegos recientes (default: 20)
    
    Returns:
        Dict con claves: era, whip, k_9, k_bb, run_diff, games_sampled
    
    Referencias:
        - Baseball-Reference.com
        - FanGraphs "Pitch Metrics"
    """
    if pitcher_df is None or pitcher_df.empty:
        return {
            "era": 4.50,
            "whip": 1.20,
            "k_9": 9.0,
            "k_bb": 2.0,
            "run_diff": 0.0,
            "games_sampled": 0
        }
    
    pitcher_games = pitcher_df[pitcher_df.get("pitcher_id") == pitcher_id]
    
    if pitcher_games.empty:
        return {
            "era": 4.50,
            "whip": 1.20,
            "k_9": 9.0,
            "k_bb": 2.0,
            "run_diff": 0.0,
            "games_sampled": 0
        }
    
    # Tomar últimos N juegos
    recent = pitcher_games.tail(window).copy()
    
    # Parsear innings (ej. "6.1" = 6 + 1/3)
    if "inningsPitched" in recent.columns:
        ip = recent["inningsPitched"].fillna(0).astype(float).sum()
    else:
        ip = 0.0
    
    era = 0.0
    if ip > 0:
        runs_allowed = recent["runs"].fillna(0).astype(float).sum()
        era = (runs_allowed / ip) * 9
    
    # WHIP
    whip = 1.20
    if ip > 0:
        hits = recent["hits"].fillna(0).astype(float).sum()
        walks = recent["baseOnBalls"].fillna(0).astype(float).sum()
        whip = (hits + walks) / ip
    
    # K/9
    k_9 = 9.0
    if ip > 0:
        ks = recent["strikeOuts"].fillna(0).astype(float).sum()
        k_9 = (ks / ip) * 9
    
    # K/BB
    k_bb = 2.0
    walks = recent["baseOnBalls"].fillna(0).astype(float).sum()
    if walks > 0:
        ks = recent["strikeOuts"].fillna(0).astype(float).sum()
        k_bb = ks / walks
    
    # Run Differential (RF - RA)
    runs_for = recent["runs"].fillna(0).astype(float).sum()  # Runs his team scored
    runs_against = recent["runs"].fillna(0).astype(float).sum()  # Runs allowed (aproximado)
    run_diff = runs_for - runs_against
    
    return {
        "era": np.clip(era, 1.5, 6.0),
        "whip": np.clip(whip, 0.8, 1.8),
        "k_9": np.clip(k_9, 4.0, 14.0),
        "k_bb": np.clip(k_bb, 0.5, 4.0),
        "run_diff": run_diff,
        "games_sampled": len(recent)
    }


def calculate_bullpen_metrics(
    bullpen_df: pd.DataFrame,
    team_id: int,
    window: int = 30
) -> Dict[str, float]:
    """
    Extrae métricas de eficiencia acumuladas para el bullpen de un equipo.
    
    Calcula agregados de ERA, WHIP, K/9 para todos los relief pitchers
    del equipo en los últimos N juegos.
    
    Args:
        bullpen_df: DataFrame con game logs de bullpen (sorted by date)
        team_id: ID del equipo
        window: Ventana de juegos recientes (default: 30)
    
    Returns:
        Dict con claves: era, whip, k_9, games_sampled
    """
    if bullpen_df is None or bullpen_df.empty:
        return {
            "era": 4.00,
            "whip": 1.15,
            "k_9": 10.0,
            "games_sampled": 0
        }
    
    # Suponemos que bullpen_df ya está filtrado por team y tipo de pitcher
    recent = bullpen_df.tail(window).copy()
    
    if recent.empty:
        return {
            "era": 4.00,
            "whip": 1.15,
            "k_9": 10.0,
            "games_sampled": 0
        }
    
    ip = recent["inningsPitched"].fillna(0).astype(float).sum() if "inningsPitched" in recent.columns else 0.0
    
    era = 4.00
    if ip > 0:
        runs = recent["runs"].fillna(0).astype(float).sum()
        era = (runs / ip) * 9
    
    whip = 1.15
    if ip > 0:
        hits = recent["hits"].fillna(0).astype(float).sum()
        walks = recent["baseOnBalls"].fillna(0).astype(float).sum()
        whip = (hits + walks) / ip
    
    k_9 = 10.0
    if ip > 0:
        ks = recent["strikeOuts"].fillna(0).astype(float).sum()
        k_9 = (ks / ip) * 9
    
    return {
        "era": np.clip(era, 2.0, 6.0),
        "whip": np.clip(whip, 0.9, 1.6),
        "k_9": np.clip(k_9, 6.0, 14.0),
        "games_sampled": len(recent)
    }


# ============================================================================
# MÓDULO 5: AGREGADOR MAESTRO DE CARACTERÍSTICAS AVANZADAS
# ============================================================================

def build_advanced_features(
    game_row: Dict[str, Any],
    home_team_data: Dict[str, Any],
    away_team_data: Dict[str, Any],
    recent_performance: Dict[int, Dict[str, float]],
    pitcher_db: Dict[int, pd.DataFrame],
    bullpen_db: Dict[int, pd.DataFrame],
) -> Dict[str, float]:
    """
    Constructor maestro que integra TODAS las características avanzadas
    en un diccionario listo para XGBoost.
    
    Función principal a llamar desde mlb_betting_refactored.py
    
    Calcula por cada equipo (home y away):
      1. Pythagorean Expectation (win% esperado)
      2. Log5 (P(Home) vs. P(Away))
      3. Home-Field Advantage ajustado
      4. Métricas del starter (ERA, WHIP, K/9)
      5. Métricas del bullpen
      6. Run Differential acumulado
    
    Args:
        game_row: Fila de juego con {game_date, hom_team_id, away_team_id, ...}
        home_team_data: Dict con {hitting: DF, pitching: DF, ...}
        away_team_data: Idem para equipo visitante
        recent_performance: Dict con win% últimos 20 juegos
        pitcher_db: Dict[pitcher_id] = DataFrame de game logs
        bullpen_db: Dict[team_id] = DataFrame de game logs
    
    Returns:
        Dict con todas las características listas para features_normalized
    """
    features = {}
    
    # ---- PYTHAGOREAN EXPECTATION ----
    # Calcula runs_for y runs_against acumulados
    for lado, team_data in [("hom", home_team_data), ("away", away_team_data)]:
        hitting_df = team_data.get("hitting", pd.DataFrame())
        pitching_df = team_data.get("pitching", pd.DataFrame())
        
        rf = hitting_df["runs"].fillna(0).astype(float).sum() if not hitting_df.empty else 0
        ra = pitching_df["runs"].fillna(0).astype(float).sum() if not pitching_df.empty else 0
        
        pyth_exp = pythagorean_expectation(rf, ra, exponent=1.83)
        features[f"{lado}_pythag_win_pct"] = pyth_exp
    
    # ---- LOG5 (H2H) ----
    home_pct = features.get("hom_pythag_win_pct", 0.500)
    away_pct = features.get("away_pythag_win_pct", 0.500)
    log5_h = log5_probability(home_pct, away_pct)
    log5_a = 1.0 - log5_h
    
    features["log5_home_prob"] = log5_h
    features["log5_away_prob"] = log5_a
    
    # ---- HOME-FIELD ADVANTAGE ----
    home_team_id = int(game_row.get("hom_team_id", 1))
    away_team_id = int(game_row.get("away_team_id", 2))
    
    hfa_home_mult, hfa_away_mult = home_field_advantage_weighted(
        home_team_id, away_team_id, recent_performance, baseline_hfa=0.054
    )
    
    # Aplicar HFA a las probabilidades Log5
    hfa_adjusted_home = log5_h * hfa_home_mult
    hfa_adjusted_away = log5_a * hfa_away_mult
    
    # Normalizar para que sumen a 1.0
    total = hfa_adjusted_home + hfa_adjusted_away
    if total > 0:
        hfa_adjusted_home /= total
        hfa_adjusted_away /= total
    
    features["hfa_adjusted_home_prob"] = hfa_adjusted_home
    features["hfa_adjusted_away_prob"] = hfa_adjusted_away
    features["hfa_multiplier"] = hfa_home_mult - 1.0  # En puntos porcentuales
    
    # ---- MÉTRICAS DE STARTERS ----
    hom_pitcher_id = game_row.get("hom_pitcher_id")
    away_pitcher_id = game_row.get("away_pitcher_id")
    
    if hom_pitcher_id and hom_pitcher_id in pitcher_db:
        hom_starter_metrics = calculate_pitcher_metrics(pitcher_db[hom_pitcher_id], hom_pitcher_id, window=20)
        features.update({
            f"hom_starter_era": hom_starter_metrics["era"],
            f"hom_starter_whip": hom_starter_metrics["whip"],
            f"hom_starter_k9": hom_starter_metrics["k_9"],
            f"hom_starter_k_bb": hom_starter_metrics["k_bb"],
            f"hom_starter_run_diff": hom_starter_metrics["run_diff"],
            f"hom_starter_games": hom_starter_metrics["games_sampled"],
        })
    else:
        features.update({
            "hom_starter_era": 4.20,
            "hom_starter_whip": 1.20,
            "hom_starter_k9": 9.0,
            "hom_starter_k_bb": 2.0,
            "hom_starter_run_diff": 0.0,
            "hom_starter_games": 0,
        })
    
    if away_pitcher_id and away_pitcher_id in pitcher_db:
        away_starter_metrics = calculate_pitcher_metrics(pitcher_db[away_pitcher_id], away_pitcher_id, window=20)
        features.update({
            f"away_starter_era": away_starter_metrics["era"],
            f"away_starter_whip": away_starter_metrics["whip"],
            f"away_starter_k9": away_starter_metrics["k_9"],
            f"away_starter_k_bb": away_starter_metrics["k_bb"],
            f"away_starter_run_diff": away_starter_metrics["run_diff"],
            f"away_starter_games": away_starter_metrics["games_sampled"],
        })
    else:
        features.update({
            "away_starter_era": 4.20,
            "away_starter_whip": 1.20,
            "away_starter_k9": 9.0,
            "away_starter_k_bb": 2.0,
            "away_starter_run_diff": 0.0,
            "away_starter_games": 0,
        })
    
    # ---- MÉTRICAS DE BULLPEN ----
    if home_team_id in bullpen_db:
        hom_bullpen_metrics = calculate_bullpen_metrics(bullpen_db[home_team_id], home_team_id, window=30)
        features.update({
            f"hom_bullpen_era": hom_bullpen_metrics["era"],
            f"hom_bullpen_whip": hom_bullpen_metrics["whip"],
            f"hom_bullpen_k9": hom_bullpen_metrics["k_9"],
            f"hom_bullpen_games": hom_bullpen_metrics["games_sampled"],
        })
    else:
        features.update({
            "hom_bullpen_era": 4.00,
            "hom_bullpen_whip": 1.15,
            "hom_bullpen_k9": 10.0,
            "hom_bullpen_games": 0,
        })
    
    if away_team_id in bullpen_db:
        away_bullpen_metrics = calculate_bullpen_metrics(bullpen_db[away_team_id], away_team_id, window=30)
        features.update({
            f"away_bullpen_era": away_bullpen_metrics["era"],
            f"away_bullpen_whip": away_bullpen_metrics["whip"],
            f"away_bullpen_k9": away_bullpen_metrics["k_9"],
            f"away_bullpen_games": away_bullpen_metrics["games_sampled"],
        })
    else:
        features.update({
            "away_bullpen_era": 4.00,
            "away_bullpen_whip": 1.15,
            "away_bullpen_k9": 10.0,
            "away_bullpen_games": 0,
        })
    
    return features


# ============================================================================
# PRUEBA / VALIDACIÓN
# ============================================================================

if __name__ == "__main__":
    print("✅ features_advanced.py cargado correctamente")
    print("\nFunciones disponibles:")
    print("  1. pythagorean_expectation(runs_for, runs_against) → float")
    print("  2. log5_probability(team_a_pct, team_b_pct) → float")
    print("  3. home_field_advantage_weighted(...) → Tuple[float, float]")
    print("  4. calculate_pitcher_metrics(...) → Dict")
    print("  5. calculate_bullpen_metrics(...) → Dict")
    print("  6. build_advanced_features(...) → Dict")
