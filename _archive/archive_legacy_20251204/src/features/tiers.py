from typing import List, Dict


class FeatureTiers:
    """
    Defines the feature sets for the Tiered Modeling approach.
    """

    @staticmethod
    def get_tier_0() -> List[str]:
        """
        Tier 0: Baseline (Price Techs + Board Crush)
        """
        return [
            # Price Techs
            "zl_ret_1d",
            "zl_ret_5d",
            "zl_vol_20d",
            "zl_dist_sma_20",
            # Board Crush Spread (The Physics)
            "board_crush_spread",
            "board_crush_z_score_20d",
        ]

    @staticmethod
    def get_tier_1() -> List[str]:
        """
        Tier 1: SHAP-Core + Physical Basis
        """
        base = FeatureTiers.get_tier_0()
        core_19 = [
            "corr_zl_brl_30d",
            "corr_zl_dxy_60d",
            "fred_dfedtarl",
            "fred_dfedtaru",
            "fred_dgs1",
            "fred_dgs10",
            "fred_nfci",
            "fred_t10y3m",
            "regime_weight",
        ]
        basis = [
            # USDA AMS Basis Proxy (Cash - Futures)
            "ams_basis_il",
            "ams_basis_gulf",
        ]
        return base + core_19 + basis

    @staticmethod
    def get_tier_2() -> List[str]:
        """
        Tier 2: Extended (Full Macro, Weather, Sentiment)
        """
        # Placeholder for full matrix columns
        return FeatureTiers.get_tier_1() + [
            "weather_iowa_precip",
            "sentiment_biofuels_net",
        ]

    @staticmethod
    def get_tier(tier_level: int) -> List[str]:
        if tier_level == 0:
            return FeatureTiers.get_tier_0()
        elif tier_level == 1:
            return FeatureTiers.get_tier_1()
        elif tier_level == 2:
            return FeatureTiers.get_tier_2()
        else:
            raise ValueError(f"Unknown Tier Level: {tier_level}")
