"""
ScrapeCreator Bucket Collectors Package.
Each module provides a fetch_bucket_items() function that returns real data from ScrapeCreators API.
"""

from . import (
    collect_biofuel_policy,
    collect_china_demand,
    collect_tariffs_trade_policy,
    collect_trump_truth_social,
)

__all__ = [
    "collect_biofuel_policy",
    "collect_china_demand",
    "collect_tariffs_trade_policy",
    "collect_trump_truth_social",
]
