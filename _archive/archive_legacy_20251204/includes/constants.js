// Dataform constants (JavaScript)
// Used across SQL definitions

const LOOKBACK_WINDOWS = {
  short: 5,
  medium: 21,
  long: 63,
  veryLong: 252
};

const HORIZONS = ['1w', '1m', '3m', '6m', '12m'];

const BIG_EIGHT = [
  'crush_margin',
  'china_imports',
  'dollar_index',
  'fed_policy',
  'tariff_intensity',
  'biofuel_demand',
  'crude_oil',
  'vix_regime'
];

const SYMBOLS = {
  core: ['ZL', 'ZS', 'ZM', 'ZC'],
  energy: ['CL', 'HO', 'RB', 'NG'],
  fx: ['DXY', 'USDBRL', 'USDARS', 'USDCNY'],
  competitors: ['FCPO', 'RS', 'RSX']
};

const REGIMES = {
  trump_anticipation_2024: 5000,
  trade_war_2017_2019: 1500,
  crisis_2008_2009: 800,
  inflation_2021_2022: 1200,
  pre_crisis_2000_2007: 50
};

const DATA_QUALITY_THRESHOLDS = {
  max_null_pct: 0.01,  // 1% nulls max
  min_row_count: 100,
  max_staleness_days: 7,
  sanity_bounds: {
    crush_margin: [-100, 1000],
    price_change_pct: [-50, 50],
    volume: [0, 1e9]
  }
};

module.exports = {
  LOOKBACK_WINDOWS,
  HORIZONS,
  BIG_EIGHT,
  SYMBOLS,
  REGIMES,
  DATA_QUALITY_THRESHOLDS
};

