import numpy as np
import pandas as pd

from ber.match.features import FEATURES, pair_features


def _norm(rows):
    cols = ["entity_id", "name_core", "name_alt", "legal_class", "addr_core", "addr_missing",
            "house_no", "postcode_like", "nums", "landmark", "name_key"]
    return pd.DataFrame(rows, columns=cols)


S1 = _norm([
    ["S1-a", "anand builders", ["anand builders"], ["PVT_LTD"], "12 main road pune", False, "12", ["411001"], ["12", "411001"], "temple", "anand builders"],
])
M = _norm([
    ["S2-x", "anmd bildrs", ["anmd bildrs"], ["LTD"], "", True, "", [], [], "", "anand builders"],
    ["S2-y", "zeta foods", ["zeta foods"], ["LLC"], "500-502 main road pune", False, "500-502", ["411002"], ["500", "502"], "", "zeta foods"],
    ["S3-z", "anand builders", ["anand builders"], [], "12 main road pune", False, "13", ["411001"], ["13", "411001"], "temple", "anand builders"],
])
CAND = pd.DataFrame({
    "s1_id": ["S1-a"] * 3, "m_id": ["S2-x", "S2-y", "S3-z"], "source": ["S2", "S2", "S3"],
    "b1_score": [0.2, 0.1, 0.9], "b1_rank": [5, 9, 1], "b2_rank": [0, 3, 1], "b3_rank": [0, 4, 1],
    "b4_hn": [False, False, True], "b4_name": [False, False, True],
    "b6n_rank": [1, 0, 2], "b6n_score": [0.92, np.nan, 0.95], "b7_rank": [0, 0, 1],
    "name_cos": [0.92, 0.81, 0.95], "addr_cos": [0.9, 0.85, 0.99],
})


def test_feature_matrix_shape_and_semantics():
    F = pair_features(CAND, S1, M)
    assert list(F.columns) == FEATURES and len(F) == 3
    x, y, z = F.iloc[0], F.iloc[1], F.iloc[2]
    # missing address is flagged as missing, not scored as a mismatch
    assert x["addr_missing"] == 1 and x["addr_tsr"] == 0 and x["hn_missing"] == 1
    assert x["pc_agree"] == 0 and x["pc_conflict"] == 0
    # transliterated key makes the non-Latin pair match on name
    assert x["key_tsr"] == 100 and x["name_tsr"] < x["key_tsr"] and x["name_cos"] > 0.9
    assert x["addr_cos"] == 0  # missing address -> no address similarity, flagged instead
    # PVT_LTD vs LTD are compatible families; LLC is a conflict
    assert x["legal_agree"] == 1 and y["legal_conflict"] == 1
    # postcode conflict, house number fuzzy (12 vs 13 edit distance 1), landmark overlap
    assert y["pc_conflict"] == 1
    assert z["hn_fuzzy"] == 1 and z["pc_agree"] == 1 and z["landmark_jacc"] == 1.0
    assert z["b4_hit"] == 1 and z["b6n_hit"] == 1 and z["b7_hit"] == 1 and z["n_blockers"] == 6 and z["is_s3"] == 1


def test_parallel_matches_serial():
    from ber.match.features import pair_features_parallel

    big = pd.concat([CAND] * 5, ignore_index=True)
    a = pair_features(big, S1, M)
    b = pair_features_parallel(big, S1, M, chunk=4)
    pd.testing.assert_frame_equal(a, b)
