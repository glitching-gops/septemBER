from collections import Counter

from ber.normalize.namekeys import Segmenter, TranslitMap, block_text, name_key, skeleton


def test_skeleton_aligns_vowelless_transliteration():
    assert skeleton("indian") == skeleton("imdiyn") == "indn"
    assert skeleton("builders") == skeleton("bildrs") == "bldrs"
    assert skeleton("ab") == "ab" and skeleton("2024") == "2024"


def test_translit_map_learns_frequent_word_with_precision():
    pairs = [("anand builders", "anmd bildrs")] * 6 + [("om builders", "om bildrs")] * 3 + [("x", "rare")] * 2
    m = TranslitMap.learn(pairs, min_support=5, min_precision=0.8)
    assert m.mapping["bildrs"] == "builders"
    assert "anmd" in m.mapping  # support 6, always with "anand" or "builders"; best is ambiguous but >= 0.8
    assert "rare" not in m.mapping  # support 2 < 5
    assert "om" not in m.mapping  # identity is never mapped


def test_translit_applied_only_to_non_latin_names():
    m = TranslitMap({"bildrs": "builders"})
    assert name_key("anmd bildrs", "devanagari", False, m, None) == "anmd builders"
    assert name_key("bildrs", "latin", False, m, None) == "bildrs"


def test_segmenter_splits_glued_domain_names_only_into_known_words():
    c = Counter({"services": 100, "private": 100, "mercantile": 50, "silver": 80, "investments": 60})
    s = Segmenter(c, min_count=20)
    assert s.segment("servicesprivatemercantile") == ["services", "private", "mercantile"]
    assert s.segment("silverinvestments") == ["silver", "investments"]
    assert s.segment("qqzzxxyyww") == ["qqzzxxyyww"]
    assert name_key("silverinvestments", "latin", True, None, s) == "silver investments"
    assert name_key("silverinvestments", "latin", False, None, s) == "silverinvestments"


def test_block_text_appends_skeleton():
    assert block_text("anand builders") == "anand builders and bldrs"  # "nn" collapses
