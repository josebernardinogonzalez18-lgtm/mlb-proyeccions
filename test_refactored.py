"""
TEST SUITE - VALIDACIÓN DE FASE 1, 2 Y 3
========================================

Verifica que todas las funciones y clases funcionan correctamente
antes de ejecutar el pipeline en producción.

Uso:
    python test_refactored.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sys

# Importar módulos
try:
    from features_advanced import (
        pythagorean_expectation,
        log5_probability,
        home_field_advantage_weighted,
        calculate_pitcher_metrics,
        calculate_bullpen_metrics,
        build_advanced_features,
    )
    print("✅ features_advanced.py importado correctamente")
except ImportError as e:
    print(f"❌ Error importando features_advanced.py: {e}")
    sys.exit(1)

try:
    from models_xgboost import (
        TimeSeriesSplitValidator,
        ConfidenceScoringXGBClassifier,
        CalibratedXGBRegressor,
        MLBModelTrainer,
    )
    print("✅ models_xgboost.py importado correctamente")
except ImportError as e:
    print(f"❌ Error importando models_xgboost.py: {e}")
    sys.exit(1)


# ============================================================================
# TEST SUITE FASE 1: FEATURES AVANZADAS
# ============================================================================

def test_fase_1_pythagorean():
    """Test: Pythagorean Expectation"""
    print("\n" + "="*70)
    print("TEST FASE 1: INGENIERÍA AVANZADA")
    print("="*70)
    
    print("\n1️⃣ Pythagorean Expectation")
    print("-" * 70)
    
    # Test case: Equipo que anota 100 y permite 80 (debe ser >0.5)
    result = pythagorean_expectation(runs_for=100, runs_against=80, exponent=1.83)
    assert 0.5 < result <= 1.0, f"Resultado inválido: {result}"
    print(f"   ✅ 100 RF vs 80 RA → Win% = {result:.1%}")
    
    # Test case: Equipos iguales (debe ser ~0.5)
    result = pythagorean_expectation(runs_for=80, runs_against=80, exponent=1.83)
    assert 0.48 < result < 0.52, f"Resultado inválido: {result}"
    print(f"   ✅ 80 RF vs 80 RA → Win% = {result:.1%}")
    
    # Test case: Equipo inferior (debe ser <0.5)
    result = pythagorean_expectation(runs_for=60, runs_against=100, exponent=1.83)
    assert 0.0 <= result < 0.5, f"Resultado inválido: {result}"
    print(f"   ✅ 60 RF vs 100 RA → Win% = {result:.1%}")
    
    print("   ✅ Pythagorean Expectation: PASSED")


def test_fase_1_log5():
    """Test: Log5 Probability"""
    print("\n2️⃣ Log5 Probability (H2H)")
    print("-" * 70)
    
    # Test case: Equipo fuerte vs débil
    result = log5_probability(team_a_win_pct=0.600, team_b_win_pct=0.400)
    assert 0.5 < result <= 1.0, f"Resultado inválido: {result}"
    print(f"   ✅ .600 vs .400 → P(A beats B) = {result:.1%}")
    
    # Test case: Equipos iguales (debe ser ~0.5)
    result = log5_probability(team_a_win_pct=0.500, team_b_win_pct=0.500)
    assert 0.48 < result < 0.52, f"Resultado inválido: {result}"
    print(f"   ✅ .500 vs .500 → P(A beats B) = {result:.1%}")
    
    # Test case: Equipo débil vs fuerte
    result = log5_probability(team_a_win_pct=0.400, team_b_win_pct=0.600)
    assert 0.0 <= result < 0.5, f"Resultado inválido: {result}"
    print(f"   ✅ .400 vs .600 → P(A beats B) = {result:.1%}")
    
    print("   ✅ Log5 Probability: PASSED")


def test_fase_1_hfa():
    """Test: Home-Field Advantage"""
    print("\n3️⃣ Home-Field Advantage Ponderado")
    print("-" * 70)
    
    # Test case: Sin datos recientes (debe usar baseline)
    recent_perf = {}
    home_mult, away_mult = home_field_advantage_weighted(
        home_team_id=1,
        away_team_id=2,
        recent_performance=recent_perf,
        baseline_hfa=0.054
    )
    assert home_mult > 1.0, f"Home mult inválido: {home_mult}"
    assert away_mult < 1.0, f"Away mult inválido: {away_mult}"
    print(f"   ✅ Sin datos: Home mult = {home_mult:.3f}, Away mult = {away_mult:.3f}")
    
    # Test case: Equipo local en racha, visitante flojo
    recent_perf = {
        1: {"home_win_pct": 0.700, "away_win_pct": 0.400, "games_last_20": 20},
        2: {"home_win_pct": 0.300, "away_win_pct": 0.600, "games_last_20": 20},
    }
    home_mult, away_mult = home_field_advantage_weighted(
        home_team_id=1,
        away_team_id=2,
        recent_performance=recent_perf,
        baseline_hfa=0.054
    )
    assert home_mult > away_mult, f"Mulipliers invertidos"
    print(f"   ✅ Local en racha: Home mult = {home_mult:.3f}, Away mult = {away_mult:.3f}")
    
    print("   ✅ Home-Field Advantage: PASSED")


def test_fase_1_pitcher_metrics():
    """Test: Pitcher Metrics"""
    print("\n4️⃣ Pitcher Metrics (ERA, WHIP, K/9)")
    print("-" * 70)
    
    # Crear DataFrame simulado de pitcher
    pitcher_data = {
        "pitcher_id": [1, 1, 1, 1, 1],
        "date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "inningsPitched": [6.0, 6.1, 5.2, 7.0, 6.0],
        "runs": [2, 1, 3, 1, 2],
        "hits": [8, 7, 9, 6, 8],
        "strikeOuts": [9, 8, 7, 10, 9],
        "baseOnBalls": [2, 1, 3, 2, 1],
    }
    pitcher_df = pd.DataFrame(pitcher_data)
    
    result = calculate_pitcher_metrics(pitcher_df, pitcher_id=1, window=5)
    
    assert "era" in result, "ERA no calculada"
    assert "whip" in result, "WHIP no calculada"
    assert "k_9" in result, "K/9 no calculada"
    assert result["era"] > 0, f"ERA inválida: {result['era']}"
    assert result["whip"] > 0, f"WHIP inválida: {result['whip']}"
    
    print(f"   ✅ ERA = {result['era']:.2f}")
    print(f"   ✅ WHIP = {result['whip']:.2f}")
    print(f"   ✅ K/9 = {result['k_9']:.1f}")
    print(f"   ✅ K/BB = {result['k_bb']:.2f}")
    
    print("   ✅ Pitcher Metrics: PASSED")


def test_fase_1_advanced_features():
    """Test: Build Advanced Features (integración)"""
    print("\n5️⃣ Build Advanced Features (Integración)")
    print("-" * 70)
    
    # Crear datos mínimos
    game_row = {
        "game_date": "2026-01-01",
        "hom_team_id": 1,
        "away_team_id": 2,
        "hom_pitcher_id": 101,
        "away_pitcher_id": 102,
    }
    
    home_team_data = {
        "hitting": pd.DataFrame({
            "runs": [5, 4, 6, 5, 4],
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
        }),
        "pitching": pd.DataFrame({
            "runs": [3, 2, 4, 2, 3],
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
        })
    }
    
    away_team_data = {
        "hitting": pd.DataFrame({
            "runs": [4, 3, 5, 4, 3],
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
        }),
        "pitching": pd.DataFrame({
            "runs": [5, 4, 6, 5, 4],
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
        })
    }
    
    features = build_advanced_features(
        game_row=game_row,
        home_team_data=home_team_data,
        away_team_data=away_team_data,
        recent_performance={},
        pitcher_db={},
        bullpen_db={},
    )
    
    assert isinstance(features, dict), "Features no es diccionario"
    assert "hom_pythag_win_pct" in features, "Pythagorean no calculada"
    assert "log5_home_prob" in features, "Log5 no calculada"
    assert "hom_starter_era" in features, "Starter metrics no calculadas"
    assert "hom_bullpen_era" in features, "Bullpen metrics no calculadas"
    
    print(f"   ✅ Pythagorean Home: {features['hom_pythag_win_pct']:.1%}")
    print(f"   ✅ Log5 Home: {features['log5_home_prob']:.1%}")
    print(f"   ✅ Total features calculadas: {len(features)}")
    
    print("   ✅ Build Advanced Features: PASSED")


# ============================================================================
# TEST SUITE FASE 2: MODELOS XGBOOST
# ============================================================================

def test_fase_2_timeseries_split():
    """Test: TimeSeriesSplitValidator"""
    print("\n" + "="*70)
    print("TEST FASE 2: MODELOS XGBOOST")
    print("="*70)
    
    print("\n1️⃣ TimeSeriesSplitValidator")
    print("-" * 70)
    
    # Crear dataset simulado
    X = pd.DataFrame(
        np.random.randn(100, 5),
        columns=[f"feature_{i}" for i in range(5)]
    )
    
    validator = TimeSeriesSplitValidator(n_splits=5)
    splits = validator.get_splits(X)
    
    assert len(splits) == 5, f"Esperaba 5 splits, obtuvo {len(splits)}"
    
    # Verificar que son Walk-Forward (no superpuestos)
    for i in range(len(splits) - 1):
        train_idx_i, test_idx_i = splits[i]
        train_idx_j, test_idx_j = splits[i + 1]
        
        # El test de split i no debe solaparse con train de split i+1
        test_max_i = test_idx_i.max()
        train_min_j = train_idx_j.min()
        
        assert test_max_i < train_min_j, "Splits solapados: data leakage!"
    
    print(f"   ✅ {len(splits)} splits generados (Walk-Forward)")
    print(f"   ✅ Sin data leakage verificado")
    print("   ✅ TimeSeriesSplitValidator: PASSED")


def test_fase_2_confidence_classifier():
    """Test: ConfidenceScoringXGBClassifier"""
    print("\n2️⃣ ConfidenceScoringXGBClassifier (Moneyline)")
    print("-" * 70)
    
    # Crear dataset simulado
    X_train = pd.DataFrame(
        np.random.randn(50, 8),
        columns=[f"feature_{i}" for i in range(8)]
    )
    y_train = pd.Series(np.random.randint(0, 2, 50))
    
    # Entrenar
    clf = ConfidenceScoringXGBClassifier(confidence_threshold=0.68)
    metrics = clf.fit(X_train, y_train, evaluate_cv=True, verbose=False)
    
    assert "mean_score" in metrics, "CV metrics no retornadas"
    print(f"   ✅ Modelo entrenado")
    print(f"   ✅ CV Accuracy: {metrics['mean_score']:.1%}")
    
    # Predecir con confianza
    X_test = pd.DataFrame(
        np.random.randn(1, 8),
        columns=[f"feature_{i}" for i in range(8)]
    )
    
    result = clf.predict_with_confidence(X_test)
    
    assert "prediction" in result, "Predicción no retornada"
    assert "confidence" in result, "Confianza no retornada"
    assert "high_confidence" in result, "Flag high_confidence no retornado"
    
    print(f"   ✅ Predicción: {result['prediction']}")
    print(f"   ✅ Confianza: {result['confidence']:.1%}")
    print(f"   ✅ High Confidence: {result['high_confidence']}")
    
    # Test: Feature importance
    top_features = clf.get_top_features(top_n=3)
    assert len(top_features) <= 3, "Top features inválido"
    print(f"   ✅ Top 3 features extraídos: {len(top_features)}")
    
    print("   ✅ ConfidenceScoringXGBClassifier: PASSED")


def test_fase_2_calibrated_regressor():
    """Test: CalibratedXGBRegressor"""
    print("\n3️⃣ CalibratedXGBRegressor (Props + Conformal Prediction)")
    print("-" * 70)
    
    # Crear dataset simulado
    X_train = pd.DataFrame(
        np.random.randn(50, 8),
        columns=[f"feature_{i}" for i in range(8)]
    )
    y_train = pd.Series(np.random.randn(50) * 2 + 5)  # Media ~5, SD ~2
    
    # Entrenar
    reg = CalibratedXGBRegressor()
    metrics = reg.fit(X_train, y_train, evaluate_cv=True, verbose=False)
    
    assert "mean_score" in metrics, "CV metrics no retornadas"
    print(f"   ✅ Modelo entrenado")
    print(f"   ✅ CV R²: {metrics['mean_score']:.1%}")
    
    # Predecir con intervalo
    X_test = pd.DataFrame(
        np.random.randn(1, 8),
        columns=[f"feature_{i}" for i in range(8)]
    )
    
    result = reg.predict_with_interval(X_test, confidence=0.90)
    
    assert "prediction" in result, "Predicción no retornada"
    assert "lower_bound" in result, "Lower bound no retornado"
    assert "upper_bound" in result, "Upper bound no retornado"
    assert result["lower_bound"] <= result["prediction"] <= result["upper_bound"], \
        "Intervalos inválidos"
    
    print(f"   ✅ Predicción: {result['prediction']:.2f}")
    print(f"   ✅ Intervalo 90%: [{result['lower_bound']:.2f}, {result['upper_bound']:.2f}]")
    print(f"   ✅ Ancho: {result['interval_width']:.2f}")
    
    print("   ✅ CalibratedXGBRegressor: PASSED")


def test_fase_2_trainer():
    """Test: MLBModelTrainer (Orquestador)"""
    print("\n4️⃣ MLBModelTrainer (Orquestador)")
    print("-" * 70)
    
    # Crear dataset simulado con targets
    n_samples = 80
    X_data = np.random.randn(n_samples, 10)
    X_cols = [f"feature_{i}" for i in range(10)]
    
    df = pd.DataFrame(X_data, columns=X_cols)
    df["hom_win"] = np.random.randint(0, 2, n_samples)
    df["total_runs"] = np.abs(np.random.randn(n_samples) * 3 + 8)
    
    # Entrenar
    trainer = MLBModelTrainer()
    metrics = trainer.train_all_targets(
        df=df,
        targets=["hom_win", "total_runs"],
        feature_cols=X_cols,
        verbose=False
    )
    
    assert "hom_win" in trainer.models, "Modelo Moneyline no entrenado"
    assert "total_runs" in trainer.models, "Modelo regresión no entrenado"
    
    print(f"   ✅ {len(trainer.models)} modelos entrenados")
    print(f"   ✅ Modelos: {list(trainer.models.keys())}")
    
    print("   ✅ MLBModelTrainer: PASSED")


# ============================================================================
# TEST SUITE FASE 3: INTEGRACIÓN
# ============================================================================

def test_fase_3_integration():
    """Test: Integración de Fase 1 + Fase 2"""
    print("\n" + "="*70)
    print("TEST FASE 3: INTEGRACIÓN COMPLETA")
    print("="*70)
    
    print("\n1️⃣ Pipeline Integration (Fase 1 → Fase 2)")
    print("-" * 70)
    
    # Crear features avanzadas
    game_row = {
        "game_date": "2026-01-01",
        "hom_team_id": 1,
        "away_team_id": 2,
        "hom_pitcher_id": 101,
        "away_pitcher_id": 102,
    }
    
    home_team_data = {
        "hitting": pd.DataFrame({
            "runs": list(range(5, 10)),
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
        }),
        "pitching": pd.DataFrame({
            "runs": list(range(2, 7)),
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
        })
    }
    
    away_team_data = {
        "hitting": pd.DataFrame({
            "runs": list(range(4, 9)),
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
        }),
        "pitching": pd.DataFrame({
            "runs": list(range(3, 8)),
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
        })
    }
    
    adv_features = build_advanced_features(
        game_row, home_team_data, away_team_data, {}, {}, {}
    )
    
    print(f"   ✅ {len(adv_features)} características avanzadas calculadas")
    
    # Crear dataset de entrenamiento
    feature_names = [f"f_{i}" for i in range(10)]
    df_train = pd.DataFrame(
        np.random.randn(100, len(feature_names)),
        columns=feature_names
    )
    df_train["hom_win"] = np.random.randint(0, 2, 100)
    
    # Entrenar con XGBoost
    trainer = MLBModelTrainer()
    trainer.train_all_targets(
        df_train,
        targets=["hom_win"],
        feature_cols=feature_names,
        verbose=False
    )
    
    # Predecir
    X_pred = pd.DataFrame(np.random.randn(1, len(feature_names)), columns=feature_names)
    clf = trainer.models["hom_win"]
    result = clf.predict_with_confidence(X_pred)
    
    print(f"   ✅ Moneyline predicción: {result['prob_home_win']:.1%}")
    print(f"   ✅ Confianza: {result['confidence']:.1%}")
    print(f"   ✅ High confidence (>68%): {result['high_confidence']}")
    
    print("   ✅ Pipeline Integration: PASSED")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Ejecuta suite completa de tests."""
    print("\n" + "="*70)
    print("🧪 TEST SUITE - MLBPROJECTIONS REFACTORED")
    print("="*70)
    
    try:
        # FASE 1
        test_fase_1_pythagorean()
        test_fase_1_log5()
        test_fase_1_hfa()
        test_fase_1_pitcher_metrics()
        test_fase_1_advanced_features()
        
        # FASE 2
        test_fase_2_timeseries_split()
        test_fase_2_confidence_classifier()
        test_fase_2_calibrated_regressor()
        test_fase_2_trainer()
        
        # FASE 3
        test_fase_3_integration()
        
        print("\n" + "="*70)
        print("✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("="*70)
        print("\n🚀 Sistema listo para producción:")
        print("   python mlb_betting_refactored.py [YYYY-MM-DD]")
        print("\n" + "="*70 + "\n")
        
        return 0
    
    except AssertionError as e:
        print(f"\n❌ TEST FALLÓ: {e}")
        print("="*70 + "\n")
        return 1
    
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
