"""Normalisation v1: examples from EDA section F and the P1 brief."""
import glob
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ber.normalize.addresses import normalize_address as A
from ber.normalize.dictionaries import DICT_DIR
from ber.normalize.names import normalize_name as N
from ber.normalize.run import NORM_COLS, normalize_frame


# ----------------------------------------------------------------- names

def test_reordered_private_limited():
    r = N("Black Solutions Limited Private")
    assert r["name_core"] == "black solutions"
    assert r["legal_class"] == ["PVT_LTD"]


def test_plus_as_and_and_digit_typos():
    assert N("Mcc0y + Coghill")["name_core"] == "mccoy and coghill"
    assert N("Mccoy & Coghill")["name_core"] == "mccoy and coghill"
    assert N("5ai Services  Private Limited")["name_core"] == "sai services"
    assert N("Capita1 End LLC")["name_core"] == "capital end"


def test_honorific_and_domain():
    r = N("M/s Marketinghitechprivate.Com")
    assert r["name_core"] == "marketinghitechprivate"
    assert r["is_domain"] is True
    assert N("www.servicesprivatemercantile.co.in")["name_core"] == "servicesprivatemercantile"


def test_aka_alias():
    r = N("Synpyra aka Horizon Infrastructure Brands LLC")
    assert r["legal_class"] == ["LLC"]
    assert "synpyra" in r["name_alt"]
    assert "horizon infrastructure brands" in r["name_alt"]
    assert N("Calodrexaria fka Piedmont Crystal Paceline")["name_alt"][-1] == "piedmont crystal paceline"


def test_leading_junk_honorific_brackets():
    r = N("Dr ... Bohra  Holding (India) Pvt Limited")
    assert r["name_core"] == "bohra holding india"
    assert "bohra holding" in r["name_alt"]
    assert r["legal_class"] == ["PVT_LTD"]
    assert N("Clm Overseas Private [Limited]")["name_core"] == "clm overseas"


def test_legal_forms_multiscript_and_french():
    assert N("लिमिटेड प्राइवेट गोल्ड")["legal_class"] == ["PVT_LTD"]
    assert N("प्रीमियर प्रोडक्ट्स एलएलपी")["legal_class"] == ["LLP"]
    assert N("Gagny (France) Club (S.A.R.L.)")["legal_class"] == ["SARL"]
    assert N("Kempf Dynamic Spring (Pllc)")["legal_class"] == ["PLLC"]
    assert N("Teague Sterling Return L.L.C.")["name_core"] == "teague sterling return"
    assert N("Pix Union SAS")["legal_class"] == ["SAS"]
    assert N("Prívate Kg Limited")["legal_class"] == ["PVT_LTD"]


def test_script_detection():
    assert N("லிமிடெட்")["name_script"] == "tamil"
    assert N("ప్రైవేట్ లిమిటెడ్")["name_script"] == "telugu"
    assert N("Acme")["name_script"] == "latin"
    assert N("")["name_script"] == "none"


def test_name_that_is_only_a_legal_word_is_kept():
    assert N("Co")["name_core"] == "co"


# ----------------------------------------------------------------- addresses

def test_us_street_unit():
    r = A("1708 Usher Pl, Fl. 0")
    assert r["addr_core"] == "1708 usher place floor 0"
    assert r["house_no"] == "1708"


def test_landmark_extraction():
    r = A("Near Nalanda Buddha Vihar, Jaitala")
    assert r["landmark"] == "nalanda buddha vihar"
    assert "nalanda" not in r["addr_core"]
    assert r["addr_core"] == "jaitala"


def test_french_bis_and_rue():
    r = A("12 bis r. de la Paix")
    assert r["addr_core"] == "12 bis rue de la paix"
    assert r["house_no"] == "12bis"
    assert A("15 R FRANÇOIS MADIOT, SAINT-NAZAIRE")["addr_core"].startswith("15 rue francois madiot")
    assert A("N° 6 RUE DU FAISAN, 2EME ETAGE")["addr_core"] == "6 rue du faisan 2nd etage"


def test_indian_initials_not_turned_into_rue():
    assert A("No 5 R K Road, Chennai")["addr_core"] == "5 r k road chennai"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("15 St Paul St, Saint Paul, MN", "15 saint paul street saint paul mn"),
        ("8C Dickens St, Jacksonville", "8c dickens street jacksonville"),
        ("St-Nazaire", "saint nazaire"),
        ("113 56th St, Chicago", "113 56th street chicago"),
        ("Ste 200, 10 Main St", "suite 200 10 main street"),
        ("Ste Foy", "sainte foy"),
        ("55 Asylum St, Hartford, CT", "55 asylum street hartford ct"),
    ],
)
def test_st_saint_vs_street(raw, expected):
    assert A(raw)["addr_core"] == expected


def test_regions_both_directions_and_native_script():
    assert A("Mack Rd, Haltom City, Texas")["region"] == "tx"
    assert A("Mack Rd, Haltom City, TX")["region"] == "tx"
    assert A("Kolkata, West Bengal")["region"] == "in_wb"
    assert A("Kolkata, WB")["region"] == "in_wb"
    assert A("Pune, महाराष्ट्र")["region"] == "in_mh"
    assert "kolkata" in A("55/2 Pura Das Rd, Calcutta, WB")["addr_core"]


def test_numbers_house_postcode():
    r = A("16306 EDWARDS RD, BOWLING GREEN TOWNSHIP, VA 24153")
    assert r["house_no"] == "16306"
    assert r["postcode_like"] == ["24153"]
    assert r["nums"] == ["16306", "24153"]
    assert A("#0043 EQ, S. NO. 43")["house_no"] == "43"
    assert A("C/O X, 500-502 Main St")["house_no"] == "500-502"
    assert A("Bangalore, Karnataka, 560001")["postcode_like"] == ["560001"]


@pytest.mark.parametrize("raw", ["", "   ", "<NULL>, N/A", "NULL"])
def test_empty_address_is_missing(raw):
    r = A(raw)
    assert r["addr_missing"] is True
    assert r["addr_core"] == "" and r["house_no"] == "" and r["nums"] == []


def test_placeholder_inside_address_is_dropped():
    r = A("216 Phoenix St, <NULL>, Payson, Arizona")
    assert "null" not in r["addr_core"] and r["addr_missing"] is False


# ----------------------------------------------------------------- plumbing

def test_yaml_values_are_all_strings():
    """PyYAML turns bare no/on/null into bool/None; every list item must stay a string."""
    for f in glob.glob(str(DICT_DIR / "*.yaml")):
        data = yaml.safe_load(Path(f).read_text(encoding="utf-8"))

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    assert isinstance(k, str), (f, k)
                    walk(v)
            elif isinstance(o, list):
                for x in o:
                    assert isinstance(x, str), (f, x)

        walk(data)


def test_normalize_frame_parallel_matches_serial():
    df = pd.DataFrame(
        {
            "entity_id": [f"S2-{i}" for i in range(6)],
            "business_name": ["Acme Pvt Ltd", "Mcc0y + Coghill", "", "Foo.com", "Dr X Y", "Z LLC"],
            "business_address": ["1 Main St", "", "Near Temple, Pune", "12 bis r. de la Paix", "x", "PO Box 5, TX"],
            "country": ["A", "B", "A", "C", "B", "A"],
        }
    )
    a = normalize_frame(df, chunk=2, processes=2)
    b = normalize_frame(df, chunk=10, processes=1)
    assert list(a.columns) == NORM_COLS
    pd.testing.assert_frame_equal(a, b)


def test_abbreviations_found_by_unknown_token_report():
    assert A("Appt 4, 12 Rue Victor Hugo")["addr_core"].startswith("apartment 4")
    assert "house" in A("HN 45, Ram Nagar").get("addr_core").split()
    assert "township" in A("Bowling Green Twp, VA").get("addr_core").split()
    assert A("Cour2 Rue Pasteur")["addr_core"].startswith("cour 2")
    assert A("Block B3, Sector 9")["addr_core"] == "block b3 sector 9"
