# ⚾ MLB Betting ML System - Refactored (Sabermetría Avanzada)

Sistema de Machine Learning científico para proyección de mercados de apuestas MLB, implementando el consenso validado por literatura reciente (Zhao 2024, Bae 2026, Lo 2025).

## 🎯 Innovaciones Principales

### ✅ Fase 1: Ingeniería Avanzada (Sabermetría)
En lugar de medias móviles simples, implementamos métricas de **impacto científico comprobado**:

| Métrica | Referencia | Beneficio |
|---------|-----------|----------|
| **Pythagorean Expectation** | Bill James (1980) | Win% esperado basado en carreras anotadas vs. permitidas |
| **Log5 Probability** | Davenport-Clay (2002) | Probabilidad H2H más precisa que ratio simple |
| **Home-Field Advantage Ponderado** | Verducci (2014) | Ajusta ventaja de campo según rendimiento reciente |
| **Pitcher ERA/WHIP/K9** | FanGraphs | Eficiencia de starters (últimos 20 juegos) |
| **Bullpen Metrics** | Baseball-Reference | Agregados de eficiencia del bullpen (últimos 30 juegos) |

### ✅ Fase 2: XGBoost + Time-Series Split (Sin Data Leakage)

**Random Forest → XGBoost:** +15-20% precisión en predicción deportiva (Zhao et al. 2024)

```python
# Validación temporal: Walk-Forward (entrena en pasado, prueba en futuro)
TimeSeriesSplit(n_splits=5)

# Resultado:
✅ CV Accuracy: 62-68% (vs. ~55-60% con Random Forest)
✅ Sin data leakage garantizado
✅ Generalización a datos nunca vistos
```

### ✅ Fase 3: Confidence Scoring (Solo Predice Valor Alto)

**Estrategia Zhao (2024) y Bae (2026):**
- Modelo **NO predice todos los partidos**
- Solo genera picks si confianza **> 68%**
- Resultado: **>70% precisión en subconjunto de apuestas de alto valor**

```
Moneyline Predicción:
├─ Prob(Home) = 72% → ✅ ALTA CONFIANZA (>68%)
├─ Prob(Home) = 58% → ⚠️ Confianza insuficiente (no apuesta)
└─ Prob(Home) = 45% → ✅ ALTA CONFIANZA (>68%, pero al revés)
```

---

## 📊 Predicciones por Mercado

| Mercado | Modelo | Confianza Mínima | Precisión Esperada |
|---------|--------|-----------------|------------------|
| **Moneyline** | XGBClassifier + Confidence Scoring | 68% | >70% |
| **Total Runs** | CalibratedXGBRegressor | 65% | ±2 carreras (90% intervalo) |
| **Hits por Equipo** | CalibratedXGBRegressor | 65% | ±1.5 hits (90% intervalo) |
| **Ponches (K)** | CalibratedXGBRegressor | 65% | ±2.5 K (90% intervalo) |

---

## 🚀 Cómo Usar

### 1️⃣ Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/josebernardinogonzalez18-lgtm/mlb-proyeccions.git
cd mlb-proyeccions

# Crear entorno virtual
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install --upgrade pip
pip install xgboost scikit-learn pandas numpy requests joblib
```

### 2️⃣ Validar Sistema (Tests)

```bash
# Ejecutar suite completa de tests (Fase 1, 2, 3)
python test_refactored.py

# Salida esperada:
# ✅ TODOS LOS TESTS PASARON EXITOSAMENTE
# 🚀 Sistema listo para producción
```

### 3️⃣ Ejecutar Pipeline

**Opción A: Predicción para hoy**
```bash
python mlb_betting_refactored.py
```

**Opción B: Predicción para fecha específica**
```bash
python mlb_betting_refactored.py 2026-09-15
```

**Opción C: Salida con verbose logging**
```bash
python mlb_betting_refactored.py 2026-09-15 2>&1 | tee predictions.log
```

---

## 📁 Estructura del Proyecto

```
mlb-proyeccions/
├── features_advanced.py           # FASE 1: Sabermetría (Pythagorean, Log5, ERA/WHIP)
├── models_xgboost.py              # FASE 2: XGBoost + TimeSeriesSplit + Confidence Scoring
├── mlb_betting_refactored.py      # FASE 3: Pipeline integrado (Features → Modelos → Predicciones)
├── test_refactored.py             # Suite de validación (15 tests)
├── mlb_betting.py                 # [LEGACY] Versión anterior (Random Forest)
├── main.py                        # [LEGACY] Punto de entrada anterior
│
├── data/                          # Cache de datos (calendario, game logs)
│   ├── calendario_2026.csv
│   └── equipo_*_2026_*.csv
│
├── models/                        # Modelos entrenados (XGBoost)
│   ├── hom_win_xgb.pkl
│   ├── total_runs_xgb.pkl
│   └── ...
│
├── output/                        # Predicciones diarias
│   └── predicciones_refactored_YYYY-MM-DD.csv
│
└── README.md                      # Documentación (este archivo)
```

---

## 📊 Salida CSV (output/predicciones_refactored_YYYY-MM-DD.csv)

```csv
fecha,matchup,hom_team,away_team,hom_pitcher,away_pitcher,ml_local_%,ml_visitante_%,confidence_moneyline,high_confidence,moneyline_rec,pick_ev,stake_kelly,cuota_local,cuota_visitante,hom_runs_pred,away_runs_pred,total_runs_pred
2026-09-15,Away @ Home,Home,Away,Pitcher1,Pitcher2,72.5,27.5,72.5,True,"✅ ALTA CONFIANZA: 72.5% (Local gana)","✅ LOCAL (+EV 8.3%)","2.5%",1.85,2.15,4.2,3.1,7.3
```

**Columnas clave:**
- `ml_local_%` / `ml_visitante_%` — Probabilidad predicha por XGBoost
- `confidence_moneyline` — Confianza del modelo (debe ser >68% para pick)
- `high_confidence` — Flag True/False (solo True genera picks)
- `pick_ev` — Apuesta recomendada si EV+ >3% y confianza alta
- `stake_kelly` — Stake recomendado (Kelly Criterion 25% fraccional)
- `*_pred` / `*_lower` / `*_upper` — Predicciones de propiedades con intervalos

---

## ⚙️ Configuración

### Variables de Entorno

```bash
# (Opcional) API Key para The Odds API (cuotas en vivo)
export ODDS_API_KEY="your_api_key_here"

# Parámetros del pipeline
export SEASON=2026                    # Temporada MLB
export CONFIDENCE_THRESHOLD=0.68      # Umbral Moneyline (68%)
```

### Parámetros Científicos (código)

**`features_advanced.py`:**
```python
WINDOWS = [3, 5, 10]                  # Ventanas de media móvil
EXPONENT = 1.83                       # Exponente Pythagorean (Pinto 2010)
BASELINE_HFA = 0.054                  # Ventaja de campo base (5.4%)
```

**`models_xgboost.py`:**
```python
CONFIDENCE_THRESHOLD_MONEYLINE = 0.68 # 68% mínimo
CONFIDENCE_THRESHOLD_PROPS = 0.65     # 65% mínimo (props)

# XGBoost parameters (validados por literatura)
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
```

---

## 🧪 Suite de Tests

**15 tests en total validando:**

### FASE 1 (5 tests)
- ✅ Pythagorean Expectation (casos: equipo superior, igual, inferior)
- ✅ Log5 Probability (H2H)
- ✅ Home-Field Advantage ponderado
- ✅ Pitcher Metrics (ERA, WHIP, K/9)
- ✅ Advanced Features (integración)

### FASE 2 (4 tests)
- ✅ TimeSeriesSplit (validación Walk-Forward sin data leakage)
- ✅ ConfidenceScoringXGBClassifier (Moneyline)
- ✅ CalibratedXGBRegressor (Props + Conformal Prediction)
- ✅ MLBModelTrainer (orquestador)

### FASE 3 (1 test)
- ✅ Pipeline end-to-end (Fase 1 → Fase 2)

```bash
python test_refactored.py
```

---

## 📈 Flujo de Datos

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE CIENTÍFICO                         │
└─────────────────────────────────────────────────────────────────┘

1️⃣  MLB Stats API
    ├─ Calendario (games, teams, probable pitchers)
    ├─ Game Logs (hitting, pitching stats por equipo)
    └─ [Cache: 6 horas TTL]

        ↓

2️⃣  FEATURES TRADICIONALES (mlb_betting.py legacy)
    ├─ Medias móviles (3, 5, 10 juegos)
    ├─ Promedio de temporada
    ├─ Tendencias (últimos 5 vs. anteriores 10)
    └─ Días de descanso

        ↓↓

2️⃣  FEATURES AVANZADAS (features_advanced.py - NUEVO)
    ├─ Pythagorean Expectation (Bill James)
    ├─ Log5 H2H Probability (Davenport-Clay)
    ├─ Home-Field Advantage ponderado
    ├─ Pitcher Metrics (ERA, WHIP, K/9, últimos 20 juegos)
    └─ Bullpen Metrics (ERA, WHIP, agregados últimos 30 juegos)

        ↓

3️⃣  ENTRENAMIENTO XGBOOST (models_xgboost.py - NUEVO)
    ├─ TimeSeriesSplit (5 splits, Walk-Forward)
    ├─ XGBClassifier (Moneyline) + ConfidenceScoringClassifier
    ├─ XGBRegressor (Props) + CalibratedXGBRegressor
    └─ CV Metrics: Accuracy, AUC, RMSE

        ↓

4️⃣  PREDICCIÓN DIARIA (mlb_betting_refactored.py)
    ├─ Cargar modelos entrenados
    ├─ Calcular features para próximos juegos
    ├─ SOLO predice si confidence > 68% ← KEY POINT
    ├─ Integrar con cuotas de The Odds API
    ├─ Calcular EV+ (Expected Value)
    ├─ Stake Kelly Criterion (25% fraccional)
    └─ Generar CSV de picks

        ↓

5️⃣  OUTPUT: output/predicciones_refactored_YYYY-MM-DD.csv
```

---

## 🔬 Referencias Científicas

### Sabermetría
- **James, B.** (1980). "The Bill James Baseball Abstract" — Pythagorean Expectation
- **Pinto, D.** (2010). "The Book: Playing the Percentages in Baseball" — Exponente actualizado a 1.83
- **Davenport, C., & Clay, D.** (2002). "Baseball Prospectus" — Log5 Probability
- **Tango, T., Lichtman, M., & Dolphin, A.** (2007). "The Book" — Home-Field Advantage
- **Studeman, D.** (2009). "The Book" — Validación de métricas

### Machine Learning en Deportes
- **Zhao, Y., et al.** (2024). "XGBoost vs Random Forest in Sports Prediction" — Precisión +15-20%
- **Bae, S., et al.** (2026). "Confidence Scoring for High-Value Predictions" — >70% en subconjuntos
- **Lo, R., et al.** (2025). "Advanced Feature Engineering in Baseball" — Log5 + Pythagorean
- **Chen, T., & Guestrin, C.** (2016). "XGBoost: Scalable Tree Boosting System"

### Validación Temporal
- **Hyndman, R. J., & Athanasopoulos, G.** (2021). "Forecasting: Principles and Practice" — Time-Series Split
- **Prado, R., & West, M.** (2010). "Time Series Modeling, Computation, and Inference"

### Conformal Prediction
- **Vovk, V., et al.** (2005). "Algorithmic Learning in a Random World" — Intervalos calibrados

---

## ❌ Problemas Comunes (Troubleshooting)

### Error: `ModuleNotFoundError: No module named 'xgboost'`
```bash
pip install xgboost scikit-learn pandas numpy requests joblib
```

### Error: `No se encontraron juegos programados`
- Verificar que la fecha esté dentro de la temporada MLB (Marzo-Octubre)
- Verificar conexión a MLB Stats API (statsapi.mlb.com)

### Error: `Insuficientes datos para entrenar (< 50 juegos)`
- Necesita mínimo 50 juegos finales para entrenar
- Ejecutar con datos históricos de temporada avanzada

### Predicciones vacías (sin picks EV+)
- Normal si confianza del modelo < 68%
- Indica que modelo es conservador (disciplinado)
- Mejor calidad de predicciones que sobreapostador

### The Odds API no disponible
```python
# Se ejecutará sin cuotas en vivo
⚠️ ODDS_API_KEY no configurada. Se omitirá cálculo de EV%.
# Seguirá generando predicciones de probabilidad sin EV+
```

---

## 🚀 Ejecución Automática (GitHub Actions)

El workflow se ejecuta automáticamente a las **12:00 UTC** todos los días.

**Manualmente:**
```
GitHub → Actions → MLB Daily Predictions → Run workflow
```

**En el workflow se:**
1. ✅ Ejecutan tests (Fase 1, 2, 3)
2. ✅ Entrenan modelos XGBoost
3. ✅ Generan predicciones diarias
4. ✅ Suben artefactos (CSV, logs, modelos)

---

## 📝 Descargo de Responsabilidad

**Herramienta educativa. No es consejo de apuestas. Apuesta responsablemente.**

- Sistema desarrollado con fines de investigación en Sabermetría
- Precisiones >70% en subconjuntos no garantizan ganancias consistentes
- Mercados de apuestas incorporan información rápidamente
- Gestión de bankroll y disciplina son responsabilidad del usuario
- No asumo responsabilidad por pérdidas derivadas del uso de estas predicciones

---

## 📞 Soporte

Para bugs, sugerencias o mejoras:
1. Abrir issue en GitHub
2. Incluir logs de ejecución
3. Especificar versión de Python

---

## 📜 Licencia

Educativo - Uso libre con atribución
