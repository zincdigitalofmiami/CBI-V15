---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
All data must come from authenticated APIs, official sources, or validated historical records.
---

# ZL = Soybean Oil Futures - Critical Clarification

## ✅ CORRECTED: ZL is Soybean Oil Futures (CBOT)

**NOT corn** - ZL is the Chicago Board of Trade (CBOT) ticker for **Soybean Oil Futures**.

## Soybean Oil (ZL) Unique Characteristics

### 1. Direct Substitution Relationships

**Palm Oil (FCPO)** is the primary substitute:
- When palm oil prices rise → soybean oil demand increases
- Spread between palm and soybean oil is critical feature
- Both are edible oils with similar end uses

### 2. Crush Spread Dynamics

Soybean Oil is part of the soybean complex:
- **ZS** = Soybeans (raw input)
- **ZL** = Soybean Oil (output)
- **ZM** = Soybean Meal (co-product)

The **crush spread** (ZS - ZL - ZM) determines processing profitability and affects ZL supply.

### 3. Geographic Production Drivers

**Primary Production Regions** (most critical for ZL):
- **Brazil**: #1 soybean producer globally → affects ZL supply
- **Argentina**: #3 soybean producer → affects ZL supply
- **US Midwest**: Less critical for ZL vs corn/wheat

**Weather Features Priority**:
1. Brazil precipitation/GDD (highest priority)
2. Argentina precipitation/GDD (high priority)
3. US Midwest (lower priority for ZL)

### 4. Demand Drivers

**Biofuel Demand**:
- Renewable diesel mandates drive ZL demand
- Policy changes significantly impact prices
- Different from corn ethanol dynamics

**Edible Oil Demand**:
- Food processing industry
- Consumer demand patterns
- Export markets (especially Asia)

### 5. Currency Impact

**USD/BRL (Brazilian Real)** is critical:
- Brazil is #1 producer
- Currency movements affect export competitiveness
- More important for ZL than for US-centric commodities

## Feature Set for ZL (Soybean Oil)

### Critical Features (Must Include):

```python
zl_critical_features = {
    'substitution': [
        'palm_price',  # FCPO - primary substitute
        'palm_soybean_oil_spread',  # Direct substitution spread
    ],
    
    'crush_spread': [
        'soybean_crush_spread',  # ZS - ZL - ZM
        'soybean_price',  # ZS input cost
        'soybean_meal_price',  # ZM co-product value
    ],
    
    'weather_brazil': [
        'brazil_precip_7d_zscore',
        'brazil_precip_30d_zscore',
        'brazil_GDD_base10C',
    ],
    
    'weather_argentina': [
        'argentina_precip_7d_zscore',
        'argentina_precip_30d_zscore',
        'argentina_GDD_base10C',
    ],
    
    'macro': [
        'USD_BRL',  # Critical for Brazil exports
        'USD_BRL_returns',
        'renewable_diesel_mandate',  # Biofuel policy
    ],
    
    'positioning': [
        'CFTC_managed_money_long',  # Significant for ZL
        'CFTC_managed_money_short',
        'CFTC_commercial_long',
        'CFTC_commercial_short',
    ]
}
```

## Context Commodities for ZL Model

When building single-asset model with context inputs:

```python
context_commodities = {
    'primary': 'palm_oil',  # FCPO - direct substitute
    'secondary': [
        'soybeans',  # ZS - input relationship
        'soybean_meal',  # ZM - co-product
        'crude_oil',  # Energy correlation
    ]
}
```

## Why Single-Asset First Makes Sense for ZL

1. **Complex relationships**: ZL has unique dynamics (crush spread, palm substitution)
2. **Geographic focus**: Brazil/Argentina weather more critical than Midwest
3. **Policy sensitivity**: Biofuel mandates affect ZL differently than other commodities
4. **Label quality**: Multi-commodity models risk injecting noise from different drivers

## Implementation Notes

### Model Architecture
```python
class ZLSoybeanOilModel(nn.Module):
    """
    Single-asset model for ZL (Soybean Oil Futures)
    """
    def __init__(self):
        # ZL-specific encoder
        self.zl_encoder = ...
        
        # Context encoders (palm, crush spread, etc.)
        self.palm_context = ...
        self.crush_context = ...
        
        # Multi-horizon heads (3, 6, 9, 12 months)
        self.horizon_heads = ...
```

### Feature Priority
1. **Palm oil spread** (highest importance)
2. **Crush spread** (supply dynamics)
3. **Brazil weather** (production driver)
4. **USD/BRL** (export competitiveness)
5. **CFTC positioning** (managed money flows)

## Summary

- ✅ **ZL = Soybean Oil Futures** (CBOT)
- ✅ **Primary substitute**: Palm oil (FCPO)
- ✅ **Key relationships**: Crush spread (ZS-ZL-ZM)
- ✅ **Weather priority**: Brazil > Argentina > US Midwest
- ✅ **Currency impact**: USD/BRL critical
- ✅ **Demand drivers**: Biofuel mandates + edible oil

---

*This clarification ensures the model architecture and features align with soybean oil's unique market dynamics*


