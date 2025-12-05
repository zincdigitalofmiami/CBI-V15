# Bucket Definitions

**Status:** Production  
**Last Updated:** December 3, 2025

## 7 Procurement Buckets for Chris

Each bucket represents a specialized context for ZL price forecasting.

---

### 1. Biofuel Policy

**Dashboard Pages:** Dashboard, Strategy

**Sentiment Layer:** Layer 2

**Key Drivers:**
- RINs (D4, D6 prices)
- EPA mandates and RFS volumes
- Biodiesel production (PADD 2)
- Crush margins

**Model:** `model_biofuel`

**Update Frequency:** 15 minutes

---

### 2. Trade/Tariffs

**Dashboard Pages:** Trade Intelligence

**Sentiment Layer:** Layer 3

**Key Drivers:**
- Trump policy scores
- China export sales
- Brazil/Argentina trade relations
- Tariff announcements

**Model:** `model_trade`

**Update Frequency:** 15 minutes (Truth Social feed)

---

### 3. Weather/Supply

**Dashboard Pages:** Sentiment

**Sentiment Layer:** Layer 4

**Key Drivers:**
- Argentina drought Z-score
- Brazil rain anomaly
- USDA WASDE yield surprises
- La Niña / El Niño indicators

**Model:** `model_weather`

**Update Frequency:** Daily

---

### 4. Palm Substitution

**Dashboard Pages:** Strategy

**Sentiment Layer:** Layer 5

**Key Drivers:**
- Indonesia export levy changes
- Malaysia stockpile levels (MPOB)
- Palm oil price spread vs. ZL
- Substitution elasticity

**Model:** `model_palm`

**Update Frequency:** Daily

---

### 5. Energy Complex

**Dashboard Pages:** Strategy

**Sentiment Layer:** Layer 6

**Key Drivers:**
- WTI crude oil price
- HOBO spread (heating oil - crude)
- RB crack spread (gasoline)
- Energy backwardation

**Model:** `model_energy`

**Update Frequency:** Daily

---

### 6. Macro Risk

**Dashboard Pages:** Dashboard

**Sentiment Layer:** Layer 7

**Key Drivers:**
- VIX volatility
- DXY (Dollar Index)
- Fed Funds Rate
- 10-year Treasury yield

**Model:** `model_macro`

**Update Frequency:** 15 minutes

---

### 7. Positioning

**Dashboard Pages:** Sentiment

**Sentiment Layer:** Layer 9

**Key Drivers:**
- CFTC managed money net long
- Producer/merchant short positions
- Speculative extremes
- Commercial hedging activity

**Model:** `model_positioning`

**Update Frequency:** Weekly (Tuesday after CFTC release)

---

## Bucket Scoring

Each bucket receives a daily score (-1.5 to +1.5) based on sentiment layer formulas.

Scores drive:
- Dashboard traffic lights (BUY/WAIT/HOLD)
- SHAP overlay contributions
- Monte Carlo scenario weights

