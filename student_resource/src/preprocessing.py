"""
Text preprocessing for Business Entity Resolution.
Normalizes business names, addresses, and country values.
"""
import re
import unicodedata
from typing import Optional


# --- Legal suffix normalization map ---
LEGAL_SUFFIXES = {
    # Full forms -> abbreviated standard
    "private limited": "pvt ltd",
    "pvt limited": "pvt ltd",
    "private ltd": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "pvt. ltd.": "pvt ltd",
    "pvt. ltd": "pvt ltd",
    "pvt.ltd.": "pvt ltd",
    "pvt.ltd": "pvt ltd",
    "p ltd": "pvt ltd",
    "limited": "ltd",
    "corporation": "corp",
    "incorporated": "inc",
    "company": "co",
    "llc": "llc",
    "l.l.c.": "llc",
    "l.l.c": "llc",
    "llp": "llp",
    "l.l.p.": "llp",
    "l.l.p": "llp",
    "plc": "plc",
    "p.l.c.": "plc",
    "gmbh": "gmbh",
    "sarl": "sarl",
    "s.a.r.l.": "sarl",
    "s.a.r.l": "sarl",
    "sa": "sa",
    "s.a.": "sa",
    "s.a": "sa",
    "sas": "sas",
    "s.a.s.": "sas",
    "s.a.s": "sas",
    "ag": "ag",
    "a.g.": "ag",
    "nv": "nv",
    "n.v.": "nv",
    "bv": "bv",
    "b.v.": "bv",
    "pty": "pty",
    "pty.": "pty",
    "pty ltd": "pty ltd",
    "pty. ltd.": "pty ltd",
    "societe anonyme": "sa",
    "societe a responsabilite limitee": "sarl",
    "eurl": "eurl",
    "sasu": "sasu",
    "sci": "sci",
    "snc": "snc",
    "scs": "scs",
    "sca": "sca",
    "se": "se",
    "groupe": "groupe",
    "et cie": "et cie",
    "cie": "cie",
}

# Common street abbreviations
STREET_ABBREVS = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "place": "pl",
    "circle": "cir",
    "highway": "hwy",
    "parkway": "pkwy",
    "square": "sq",
    "terrace": "ter",
    "nagar": "ngr",
    "marg": "marg",
    "path": "path",
    "floor": "flr",
    "building": "bldg",
    "apartment": "apt",
    "suite": "ste",
    "number": "no",
    "district": "dist",
    "sector": "sec",
    "phase": "ph",
    "block": "blk",
    "cross": "cross",
    "main": "main",
    "near": "nr",
    "opposite": "opp",
}

# Country normalization
COUNTRY_MAP = {
    "us": "us",
    "usa": "us",
    "u.s.": "us",
    "u.s.a.": "us",
    "united states": "us",
    "united states of america": "us",
    "america": "us",
    "india": "india",
    "in": "india",
    "ind": "india",
    "france": "france",
    "fr": "france",
    "fra": "france",
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII-compatible form."""
    if not text:
        return ""
    # NFKD decomposition + ASCII encoding
    text = unicodedata.normalize("NFKD", text)
    # Keep the original if ASCII conversion loses too much
    try:
        ascii_text = text.encode("ascii", "ignore").decode("ascii")
        # Only use ASCII if we didn't lose more than 20% of characters
        if len(ascii_text) >= len(text) * 0.8:
            return ascii_text
    except Exception:
        pass
    return text


def clean_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces, strip."""
    return re.sub(r"\s+", " ", text).strip()


def remove_punctuation(text: str, keep_chars: str = "") -> str:
    """Remove punctuation except specified characters."""
    pattern = f"[^a-z0-9\\s{re.escape(keep_chars)}]"
    return re.sub(pattern, " ", text)


def normalize_name(name: Optional[str]) -> str:
    """Normalize a business name for matching.
    
    Steps:
    1. Lowercase
    2. Unicode normalization
    3. Replace & with 'and'
    4. Remove punctuation
    5. Normalize legal suffixes
    6. Collapse whitespace
    
    Args:
        name: Raw business name
        
    Returns:
        Normalized business name string
    """
    if name is None or (isinstance(name, float) and str(name) == "nan"):
        return ""
    
    name = str(name).lower().strip()
    if not name:
        return ""
    
    # Unicode normalization
    name = normalize_unicode(name)
    
    # Replace & with and
    name = name.replace("&", " and ")
    name = name.replace("+", " and ")
    
    # Remove periods from abbreviations but keep the letters
    # e.g., "U.S.A." -> "usa", "P.V.T." -> "pvt"
    name = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", name)
    
    # Remove remaining punctuation
    name = remove_punctuation(name)
    
    # Collapse whitespace
    name = clean_whitespace(name)
    
    # Normalize legal suffixes
    # Sort by length descending to match longer phrases first
    for full, short in sorted(LEGAL_SUFFIXES.items(), key=lambda x: -len(x[0])):
        # Match at word boundaries
        pattern = r"\b" + re.escape(full) + r"\b"
        name = re.sub(pattern, short, name)
    
    # Final cleanup
    name = clean_whitespace(name)
    return name


def normalize_name_aggressive(name: Optional[str]) -> str:
    """More aggressive normalization - removes legal suffixes entirely.
    Useful for blocking/candidate generation where we want broader matching.
    """
    norm = normalize_name(name)
    if not norm:
        return ""
    
    # Remove common legal suffixes entirely for broader matching
    suffixes_to_remove = [
        "pvt ltd", "ltd", "corp", "inc", "co", "llc", "llp", "plc",
        "gmbh", "sarl", "sa", "sas", "ag", "nv", "bv", "pty ltd", "pty",
        "eurl", "sasu", "sci", "snc", "scs", "sca", "se", "groupe",
        "et cie", "cie",
    ]
    for suffix in sorted(suffixes_to_remove, key=len, reverse=True):
        pattern = r"\b" + re.escape(suffix) + r"\b"
        norm = re.sub(pattern, "", norm)
    
    return clean_whitespace(norm)


def normalize_address(address: Optional[str]) -> str:
    """Normalize a business address for matching.
    
    Steps:
    1. Lowercase
    2. Unicode normalization
    3. Normalize street abbreviations
    4. Remove excess punctuation (keep numbers)
    5. Collapse whitespace
    
    Args:
        address: Raw business address
        
    Returns:
        Normalized address string
    """
    if address is None or (isinstance(address, float) and str(address) == "nan"):
        return ""
    
    address = str(address).lower().strip()
    if not address:
        return ""
    
    # Unicode normalization
    address = normalize_unicode(address)
    
    # Replace & with and
    address = address.replace("&", " and ")
    
    # Remove periods from abbreviations
    address = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", address)
    
    # Remove punctuation but keep hyphens in addresses (e.g., ZIP codes like 400-001)
    address = remove_punctuation(address, keep_chars="-")
    
    # Collapse whitespace
    address = clean_whitespace(address)
    
    # Normalize street abbreviations
    for full, short in sorted(STREET_ABBREVS.items(), key=lambda x: -len(x[0])):
        pattern = r"\b" + re.escape(full) + r"\b"
        address = re.sub(pattern, short, address)
    
    return clean_whitespace(address)


def normalize_country(country: Optional[str]) -> str:
    """Normalize a country value.
    
    Args:
        country: Raw country value
        
    Returns:
        Normalized country string
    """
    if country is None or (isinstance(country, float) and str(country) == "nan"):
        return ""
    
    country = str(country).lower().strip()
    country = normalize_unicode(country)
    country = remove_punctuation(country)
    country = clean_whitespace(country)
    
    return COUNTRY_MAP.get(country, country)


def extract_tokens(text: str) -> list:
    """Extract word tokens from normalized text."""
    if not text:
        return []
    return text.split()


def extract_numeric_tokens(text: str) -> list:
    """Extract numeric tokens from text (e.g., ZIP codes, street numbers)."""
    if not text:
        return []
    return re.findall(r"\b\d+\b", text)


def extract_name_key(name: str) -> str:
    """Create a blocking key from a business name.
    Uses first 3 characters of the aggressive normalization.
    """
    norm = normalize_name_aggressive(name)
    tokens = extract_tokens(norm)
    if not tokens:
        return ""
    # Use first significant token (skip very short ones)
    for t in tokens:
        if len(t) >= 3:
            return t[:4]
    return tokens[0][:4] if tokens else ""


def get_name_tokens_set(name: str) -> set:
    """Get set of tokens from normalized name."""
    return set(extract_tokens(normalize_name(name)))


def get_name_bigrams(name: str) -> set:
    """Get character bigrams for fuzzy matching."""
    norm = normalize_name(name)
    if len(norm) < 2:
        return set()
    return {norm[i:i+2] for i in range(len(norm) - 1)}
