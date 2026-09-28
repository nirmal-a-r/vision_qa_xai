"""
spi.py
======
Name kept because the project document (Sections 7 and 11) refers to
``src/risk/spi.py`` as the escape-scoring + SPI module. The heuristic that used to
live here (it never beat CRC; see ``baselines.legacy_nearest_rank_threshold``) was
removed on 2026-09-23. This module now only re-exports the two real pieces:

* escape scoring, both 0/1 events, one scalar per image  -> ``src/risk/escape.py``
* the exact SPI transporter, windows, bounds, Algorithm 4 -> ``src/risk/spi_exact.py``

and the SPERC certificate built on them (``src/risk/sperc.py``).
"""

from src.risk.escape import (KINDS, NEG_INF, break_ties, crc_escape_threshold,  # noqa: F401
                             empirical_escape, escape_score, escape_scores,
                             localized_escape_score, part_escape_score)
from src.risk.spi_exact import (coverage_lower_worst_case, coverage_upper_worst_case,  # noqa: F401
                                rank_pmf, rank_windows, select_beta_alg4, spi_covers,
                                spi_score_threshold, tier_h_cap, tier_h_floor,
                                tier_n_interval, transport)
from src.risk.sperc import Certificate, certify_crc, certify_sperc, commissioning_table  # noqa: F401
