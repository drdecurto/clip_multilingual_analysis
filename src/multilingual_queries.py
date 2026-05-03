#!/usr/bin/env python3
"""
Multilingual driving-concept queries for dense-CLIP grounding.

Same 13 languages as the parent project (`vlm_energy_signatures_multilingual`):
  ar, ca, de, en, es, eu, fr, it, lb, pt, ru, zh-CN, zh-TW

Concepts chosen to match BDD100K's object taxonomy.

Two prompting modes are provided:
    bare   — noun only                    ("car", "coche", "Auto")
    indef  — natural indefinite phrasing  ("a car", "un coche", "ein Auto")

`indef` is the default (matches OneMap's `"a " + query_text[0]` convention). For
strict cross-language comparability you may prefer `bare`.
"""

from typing import Dict, List

# =============================================================================
# Language table (mirrors the v3 evaluator in the parent project)
# =============================================================================

LANGUAGE_INFO: Dict[str, Dict[str, str]] = {
    "ar":    {"name": "Arabic",               "native": "العربية",        "family": "Semitic",  "low_resource": True},
    "ca":    {"name": "Catalan",              "native": "Català",          "family": "Romance",  "low_resource": False},
    "de":    {"name": "German",               "native": "Deutsch",         "family": "Germanic", "low_resource": False},
    "en":    {"name": "English",              "native": "English",         "family": "Germanic", "low_resource": False},
    "es":    {"name": "Spanish",              "native": "Español",         "family": "Romance",  "low_resource": False},
    "eu":    {"name": "Basque",               "native": "Euskara",         "family": "Isolate",  "low_resource": True},
    "fr":    {"name": "French",               "native": "Français",        "family": "Romance",  "low_resource": False},
    "it":    {"name": "Italian",              "native": "Italiano",        "family": "Romance",  "low_resource": False},
    "lb":    {"name": "Luxembourgish",        "native": "Lëtzebuergesch",  "family": "Germanic", "low_resource": True},
    "pt":    {"name": "Portuguese",           "native": "Português",       "family": "Romance",  "low_resource": False},
    "ru":    {"name": "Russian",              "native": "Русский",         "family": "Slavic",   "low_resource": False},
    "zh-CN": {"name": "Chinese (Simplified)",  "native": "简体中文",         "family": "Sinitic",  "low_resource": False},
    "zh-TW": {"name": "Chinese (Traditional)", "native": "繁體中文",         "family": "Sinitic",  "low_resource": False},
}

ALL_LANGUAGES: List[str] = list(LANGUAGE_INFO.keys())

LOW_RESOURCE_LANGS: List[str] = [l for l, info in LANGUAGE_INFO.items() if info["low_resource"]]
# → ["ar", "eu", "lb"]   (the three double-penalty languages)


# =============================================================================
# Concept translations (bare noun form)
# =============================================================================

CONCEPTS_BARE: Dict[str, Dict[str, str]] = {
    "car": {
        "ar": "سيارة", "ca": "cotxe", "de": "Auto", "en": "car", "es": "coche",
        "eu": "autoa", "fr": "voiture", "it": "automobile", "lb": "Auto",
        "pt": "carro", "ru": "машина", "zh-CN": "汽车", "zh-TW": "汽車",
    },
    "truck": {
        "ar": "شاحنة", "ca": "camió", "de": "Lastwagen", "en": "truck", "es": "camión",
        "eu": "kamioia", "fr": "camion", "it": "camion", "lb": "Camion",
        "pt": "caminhão", "ru": "грузовик", "zh-CN": "卡车", "zh-TW": "卡車",
    },
    "bus": {
        "ar": "حافلة", "ca": "autobús", "de": "Bus", "en": "bus", "es": "autobús",
        "eu": "autobusa", "fr": "autobus", "it": "autobus", "lb": "Bus",
        "pt": "ônibus", "ru": "автобус", "zh-CN": "公交车", "zh-TW": "公車",
    },
    "person": {
        "ar": "شخص", "ca": "persona", "de": "Person", "en": "person", "es": "persona",
        "eu": "pertsona", "fr": "personne", "it": "persona", "lb": "Persoun",
        "pt": "pessoa", "ru": "человек", "zh-CN": "人", "zh-TW": "人",
    },
    "pedestrian": {
        "ar": "مشاة", "ca": "vianant", "de": "Fußgänger", "en": "pedestrian", "es": "peatón",
        "eu": "oinezkoa", "fr": "piéton", "it": "pedone", "lb": "Foussgänger",
        "pt": "pedestre", "ru": "пешеход", "zh-CN": "行人", "zh-TW": "行人",
    },
    "traffic_light": {
        "ar": "إشارة مرور", "ca": "semàfor", "de": "Ampel", "en": "traffic light",
        "es": "semáforo", "eu": "semaforoa", "fr": "feu de circulation",
        "it": "semaforo", "lb": "Verkéiersluucht", "pt": "semáforo",
        "ru": "светофор", "zh-CN": "交通灯", "zh-TW": "交通燈",
    },
    "traffic_sign": {
        "ar": "علامة مرور", "ca": "senyal de trànsit", "de": "Verkehrsschild",
        "en": "traffic sign", "es": "señal de tráfico", "eu": "trafiko seinalea",
        "fr": "panneau de signalisation", "it": "segnale stradale",
        "lb": "Verkéierszeechen", "pt": "sinal de trânsito",
        "ru": "дорожный знак", "zh-CN": "交通标志", "zh-TW": "交通標誌",
    },
    "bicycle": {
        "ar": "دراجة", "ca": "bicicleta", "de": "Fahrrad", "en": "bicycle",
        "es": "bicicleta", "eu": "bizikleta", "fr": "vélo", "it": "bicicletta",
        "lb": "Vëlo", "pt": "bicicleta", "ru": "велосипед",
        "zh-CN": "自行车", "zh-TW": "自行車",
    },
    "motorcycle": {
        "ar": "دراجة نارية", "ca": "motocicleta", "de": "Motorrad", "en": "motorcycle",
        "es": "motocicleta", "eu": "motozikleta", "fr": "moto", "it": "motocicletta",
        "lb": "Motocikel", "pt": "motocicleta", "ru": "мотоцикл",
        "zh-CN": "摩托车", "zh-TW": "摩托車",
    },
    "road": {
        "ar": "طريق", "ca": "carretera", "de": "Straße", "en": "road",
        "es": "carretera", "eu": "errepidea", "fr": "route", "it": "strada",
        "lb": "Strooss", "pt": "estrada", "ru": "дорога",
        "zh-CN": "道路", "zh-TW": "道路",
    },
    "building": {
        "ar": "مبنى", "ca": "edifici", "de": "Gebäude", "en": "building",
        "es": "edificio", "eu": "eraikina", "fr": "bâtiment", "it": "edificio",
        "lb": "Gebai", "pt": "edifício", "ru": "здание",
        "zh-CN": "建筑物", "zh-TW": "建築物",
    },
}


# =============================================================================
# Indefinite article templates — natural phrasing per language
# =============================================================================

# Each value is (masculine_sg, feminine_sg) or a single form for languages without
# article agreement. Used with `INDEF_ARTICLE_ASSIGNMENT` below.
_INDEF_ARTICLES: Dict[str, Dict[str, str]] = {
    "ar":    {"default": ""},                                      # Arabic has no indef. article
    "ca":    {"m": "un",    "f": "una"},
    "de":    {"m": "ein",   "f": "eine",  "n": "ein"},
    "en":    {"default": "a"},
    "es":    {"m": "un",    "f": "una"},
    "eu":    {"default": ""},                                      # Basque: indef. via bare form
    "fr":    {"m": "un",    "f": "une"},
    "it":    {"m": "un",    "f": "una"},
    "lb":    {"m": "en",    "f": "eng",   "n": "en"},
    "pt":    {"m": "um",    "f": "uma"},
    "ru":    {"default": ""},                                      # Russian has no articles
    "zh-CN": {"default": "一个"},                                   # 一个 (generic classifier+number)
    "zh-TW": {"default": "一個"},
}

# Gender assignment per concept per language (best effort, checked)
# Keys: "m", "f", "n" or "default"
_GENDER: Dict[str, Dict[str, str]] = {
    "car":            {"ca": "m", "de": "n", "es": "m", "fr": "f", "it": "f", "lb": "n", "pt": "m"},
    "truck":          {"ca": "m", "de": "m", "es": "m", "fr": "m", "it": "m", "lb": "m", "pt": "m"},
    "bus":            {"ca": "m", "de": "m", "es": "m", "fr": "m", "it": "m", "lb": "m", "pt": "m"},
    "person":         {"ca": "f", "de": "f", "es": "f", "fr": "f", "it": "f", "lb": "f", "pt": "f"},
    "pedestrian":     {"ca": "m", "de": "m", "es": "m", "fr": "m", "it": "m", "lb": "m", "pt": "m"},
    "traffic_light":  {"ca": "m", "de": "f", "es": "m", "fr": "m", "it": "m", "lb": "f", "pt": "m"},
    "traffic_sign":   {"ca": "m", "de": "n", "es": "f", "fr": "m", "it": "m", "lb": "n", "pt": "m"},
    "bicycle":        {"ca": "f", "de": "n", "es": "f", "fr": "m", "it": "f", "lb": "m", "pt": "f"},
    "motorcycle":     {"ca": "f", "de": "n", "es": "f", "fr": "f", "it": "f", "lb": "m", "pt": "f"},
    "road":           {"ca": "f", "de": "f", "es": "f", "fr": "f", "it": "f", "lb": "f", "pt": "f"},
    "building":       {"ca": "m", "de": "n", "es": "m", "fr": "m", "it": "m", "lb": "m", "pt": "m"},
}


def _indef_article(concept: str, lang: str) -> str:
    """Return the natural indefinite article for (concept, lang) or '' if none."""
    articles = _INDEF_ARTICLES.get(lang, {})
    if "default" in articles:
        return articles["default"]
    gender = _GENDER.get(concept, {}).get(lang, "m")
    return articles.get(gender, articles.get("m", ""))


def render_query(concept: str, lang: str, template: str = "indef") -> str:
    """Return the rendered query string for (concept, lang)."""
    if concept not in CONCEPTS_BARE:
        raise KeyError(f"Unknown concept '{concept}'. Known: {list(CONCEPTS_BARE)}")
    if lang not in LANGUAGE_INFO:
        raise KeyError(f"Unknown language '{lang}'. Known: {ALL_LANGUAGES}")

    noun = CONCEPTS_BARE[concept][lang]
    if template == "bare":
        return noun
    if template == "indef":
        art = _indef_article(concept, lang)
        if not art:
            return noun
        # CJK languages conventionally use no space between classifier and noun.
        sep = "" if lang in {"zh-CN", "zh-TW"} else " "
        return f"{art}{sep}{noun}"
    raise ValueError(f"Unknown template '{template}' (use 'bare' or 'indef')")


def all_queries(concepts: List[str], languages: List[str],
                template: str = "indef") -> Dict[str, Dict[str, str]]:
    """Return {concept: {lang: query_string}} for the chosen subset."""
    return {
        c: {l: render_query(c, l, template) for l in languages}
        for c in concepts
    }


ALL_CONCEPTS: List[str] = list(CONCEPTS_BARE.keys())


if __name__ == "__main__":
    # Smoke test — print a small grid of rendered queries
    sample_concepts = ["car", "pedestrian", "traffic_light"]
    sample_langs = ["en", "es", "fr", "de", "ar", "eu", "lb", "zh-CN"]
    print(f"Languages: {len(ALL_LANGUAGES)}   Concepts: {len(ALL_CONCEPTS)}")
    print(f"Low-resource: {LOW_RESOURCE_LANGS}")
    print()
    for c in sample_concepts:
        print(f"• {c}")
        for l in sample_langs:
            q_bare = render_query(c, l, "bare")
            q_indef = render_query(c, l, "indef")
            print(f"    {l:5s}  bare='{q_bare}'   indef='{q_indef}'")
        print()
