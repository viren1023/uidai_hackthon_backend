import re
from num2words import num2words
import csv
from io import StringIO
import os
from rapidfuzz import process, fuzz
from datetime import datetime
import json
from app.utils.database import biometric_collection
import pandas as pd

# CONSTANTS
with open("app/JSON/location.json", "r", encoding="utf-8") as f:
    MASTER_DATA = json.load(f)
STANDARD_COLUMNS = {"date", "state", "district", "pincode"}
STATE_LOWER_MAP = {s.lower(): s for s in MASTER_DATA.keys()}
    
ALL_DISTRICTS = []
DISTRICT_TO_STATE = {}
for state, districts in MASTER_DATA.items():
    for district in districts:
        ALL_DISTRICTS.append(district)
        DISTRICT_TO_STATE[district] = state

DISTRICT_LOWER_MAP = {d.lower(): d for d in ALL_DISTRICTS}
ALIAS_FILE = 'app/JSON/district_state_aliases.json'
# Directional translations
DIRECTIONAL_MAP = {
    'north': 'uttar',
    'south': 'dakshin',
    'east': 'purba',
    'west': 'paschim'
}

def replace_numbers_with_words(text):
    def convert(match):
        return num2words(int(match.group())).replace("-", " ")
    return re.sub(r'\b\d+\b', convert, text)

def normalize_text(text):
    """
    Only keeps text and removes all other context
    """
    text = text.lower().strip()
    text = text.replace("&", "and")
    # text = replace_numbers_with_words(text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

async def savetoDB(canonical_events):
    # await biometric_collection.create_index(
    # [("date", 1), ("state", 1), ("district", 1), ("pincode", 1)],
    # unique=True)
    if canonical_events:
        await biometric_collection.insert_many(canonical_events)
        # print(f"Inserted IDs: {result.inserted_ids}")
    pass


async def process_uidai_csv(contents, source):
    """
    Takes raw CSV file bytes
    Returns list of canonical event records
    """

    # text = contents.decode("utf-8", errors="ignore")
    # reader = csv.DictReader(StringIO(text))
    
    df = pd.read_csv(StringIO(contents.decode("utf-8", errors="ignore")))
    df = df.drop_duplicates(keep="first")
    rows = df.to_dict('records')

    canonical_events = []

    for row in rows:

        raw_date = row["date"]
        raw_state = row["state"]
        raw_district = row["district"]
        raw_pincode = row["pincode"]

        state_norm = normalize_text(raw_state)
        district_norm = normalize_text(raw_district)

        state, dist, score = get_fuzzy_match(state_norm, district_norm)

        event = {
            "date": raw_date,
            "state": state,
            "district": dist,
            "pincode": raw_pincode,
            "mertics": {
                    key: int(float(value)) if value not in (None, "", " ") else 0
                    for key, value in row.items()
                    if key.lower() not in STANDARD_COLUMNS
            },
            "metadata": {
                "soruce_dataset": source,
                "raw_state": raw_state,
                "raw_district": raw_district,
                "dist_confidence_score": score,
                "processed_at": datetime.now().isoformat(),
            }
        }
        canonical_events.append(event)
    print("process_uidai_csv")
    await savetoDB(canonical_events)
    print("data save finish")
    return canonical_events

# async def process_uidai_csv(contents, source):
#     """
#     Takes raw CSV file bytes, merges duplicates by date+state+district+pincode,
#     sums numeric metrics, and saves to MongoDB.
#     """
#     text = contents.decode("utf-8", errors="ignore")
#     reader = csv.DictReader(StringIO(text))

#     # Dictionary to merge events
#     events_dict = {}  # key = (date, state, district, pincode)

#     for row in reader:
#         raw_date = row["date"]
#         raw_state = row["state"]
#         raw_district = row["district"]
#         raw_pincode = row["pincode"]

#         state_norm = normalize_text(raw_state)
#         district_norm = normalize_text(raw_district)
#         state, dist, score = get_fuzzy_match(state_norm, district_norm)

#         # Convert metrics to int when possible
#         metrics = {}
#         for k, v in row.items():
#             if k.lower() not in STANDARD_COLUMNS:
#                 try:
#                     metrics[k] = int(v)
#                 except (ValueError, TypeError):
#                     metrics[k] = v  # keep as string if not numeric

#         key = (raw_date, state, dist, raw_pincode)

#         if key in events_dict:
#             # Merge numeric metrics
#             for k, v in metrics.items():
#                 if isinstance(v, int):
#                     events_dict[key]["metrics"][k] = events_dict[key]["metrics"].get(k, 0) + v
#                 else:
#                     events_dict[key]["metrics"][k] = v  # overwrite non-numeric
#         else:
#             # Create new event
#             events_dict[key] = {
#                 "date": datetime.strptime(raw_date, "%Y-%m-%d"),  # store as datetime
#                 "state": state,
#                 "district": dist,
#                 "pincode": raw_pincode,
#                 "metrics": metrics,
#                 "metadata": {
#                     "source_dataset": source,
#                     "raw_state": raw_state,
#                     "raw_district": raw_district,
#                     "dist_confidence_score": score,
#                     "processed_at": datetime.now(),
#                 }
#             }

#     canonical_events = list(events_dict.values())

#     print(f"Processed {len(canonical_events)} unique events")
#     await savetoDB(canonical_events)
#     print("Data saved to MongoDB")
#     return canonical_events


# Load or initialize aliases
def load_aliases():
    if os.path.exists(ALIAS_FILE):
        with open(ALIAS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'states': {}, 'districts': {}}

def save_aliases(aliases):
    with open(ALIAS_FILE, 'w', encoding='utf-8') as f:
        json.dump(aliases, f, indent=2, ensure_ascii=False)

# Global aliases dictionary
ALIASES = load_aliases()

# Build reverse lookup for fast access (correct_name -> list of aliases)
def build_reverse_lookup(aliases_dict):
    """Convert from {correct: [alias1, alias2]} to {alias: correct}"""
    reverse = {}
    for correct_name, alias_list in aliases_dict.items():
        # Add the correct name itself
        reverse[correct_name.lower()] = correct_name
        # Add all aliases
        for alias in alias_list:
            reverse[alias.lower()] = correct_name
    return reverse

# Create reverse lookups for O(1) access
STATE_ALIAS_LOOKUP = build_reverse_lookup(ALIASES.get('states', {}))
DISTRICT_ALIAS_LOOKUP = build_reverse_lookup(ALIASES.get('districts', {}))

def translate_directional(text):
    """Translate North/South/East/West to local equivalents and vice versa"""
    words = text.lower().split()
    if not words:
        return text
    
    first_word = words[0]
    
    # English to local
    if first_word in DIRECTIONAL_MAP:
        return DIRECTIONAL_MAP[first_word] + ' ' + ' '.join(words[1:])
    
    # Local to English
    for eng, local in DIRECTIONAL_MAP.items():
        if first_word == local:
            return eng + ' ' + ' '.join(words[1:])
    
    return text

def add_to_aliases(correct_name, raw_name, alias_type='districts'):
    """Add a new alias to the correct structure"""
    global STATE_ALIAS_LOOKUP, DISTRICT_ALIAS_LOOKUP
    
    raw_name_lower = raw_name.lower()
    correct_name_clean = correct_name  # Keep original casing
    
    # Initialize if doesn't exist
    if correct_name_clean not in ALIASES[alias_type]:
        ALIASES[alias_type][correct_name_clean] = []
    
    # Add alias if not already there
    if raw_name_lower not in ALIASES[alias_type][correct_name_clean]:
        ALIASES[alias_type][correct_name_clean].append(raw_name_lower)
        save_aliases(ALIASES)
        
        # Update reverse lookup
        if alias_type == 'states':
            STATE_ALIAS_LOOKUP[raw_name_lower] = correct_name_clean
        else:
            DISTRICT_ALIAS_LOOKUP[raw_name_lower] = correct_name_clean

def get_fuzzy_match(raw_state, raw_district, learn=True, min_confidence=60):
    """
    Takes raw state and district (already cleaned and lowercased)
    Returns its correct matching state and district with confidence score
    
    Args:
        raw_state: Input state name
        raw_district: Input district name
        learn: If True, saves successful matches to aliases (default: True)
        min_confidence: Minimum confidence score to accept a match (default: 60)
    """
    
    # Normalize inputs
    raw_state = str(raw_state).strip().lower()
    raw_district = str(raw_district).strip().lower()
    
    # STEP 0: Check aliases first (instant lookup)
    state_from_alias = STATE_ALIAS_LOOKUP.get(raw_state)
    district_from_alias = DISTRICT_ALIAS_LOOKUP.get(raw_district)
    
    if state_from_alias and district_from_alias:
        return state_from_alias, district_from_alias, 100.0
    
    # STEP 1: Try exact state match
    matched_state_key = STATE_LOWER_MAP.get(raw_state)
    if not matched_state_key and state_from_alias:
        matched_state_key = state_from_alias
    
    # STEP 2: Fuzzy state matching
    if not matched_state_key:
        best_state_score = 0
        for official_state in MASTER_DATA.keys():
            max_score = max(
                fuzz.ratio(raw_state, official_state.lower()),
                fuzz.partial_ratio(raw_state, official_state.lower()),
                fuzz.token_sort_ratio(raw_state, official_state.lower())
            )
            if max_score > best_state_score:
                best_state_score = max_score
                matched_state_key = official_state
        
        if best_state_score < 70:
            return raw_state, raw_district, 0.0
        
        # Learn this state mapping
        if learn and best_state_score >= 85:
            add_to_aliases(matched_state_key, raw_state, 'states')
    
    # STEP 3: Check district alias
    if district_from_alias:
        correct_state = DISTRICT_TO_STATE.get(district_from_alias, matched_state_key)
        return correct_state, district_from_alias, 100.0
    
    # STEP 4: Try exact district match
    matched_district = DISTRICT_LOWER_MAP.get(raw_district)
    if matched_district:
        correct_state = DISTRICT_TO_STATE.get(matched_district, matched_state_key)
        if learn:
            add_to_aliases(matched_district, raw_district, 'districts')
        return correct_state, matched_district, 100.0
    
    # STEP 5: Try directional translation
    translated_district = translate_directional(raw_district)
    if translated_district != raw_district:
        matched_district = DISTRICT_LOWER_MAP.get(translated_district)
        if matched_district:
            correct_state = DISTRICT_TO_STATE.get(matched_district, matched_state_key)
            if learn:
                add_to_aliases(matched_district, raw_district, 'districts')
            return correct_state, matched_district, 100.0
    
    # STEP 6: Fuzzy district matching within state
    best_district = None
    best_score = 0
    
    districts_in_state = MASTER_DATA.get(matched_state_key, [])
    
    for official_district in districts_in_state:
        official_lower = official_district.lower()
        
        # Test both original and translated versions
        for test_name in [raw_district, translated_district]:
            ratio_score = fuzz.ratio(test_name, official_lower)
            partial_score = fuzz.partial_ratio(test_name, official_lower)
            token_sort_score = fuzz.token_sort_ratio(test_name, official_lower)
            token_set_score = fuzz.token_set_ratio(test_name, official_lower)
            
            current_score = max(ratio_score, partial_score, token_sort_score, token_set_score)
            
            if current_score > best_score:
                best_score = current_score
                best_district = official_district
    
    # Accept match if above threshold
    if best_score >= min_confidence:
        # Learn high-confidence matches
        if learn and best_score >= 80:
            add_to_aliases(best_district, raw_district, 'districts')
        return matched_state_key, best_district, best_score
    
    # STEP 7: Global search across all districts (only if local match was poor)
    if best_score < 70:
        best_global_district = None
        best_global_score = 0
        best_global_state = matched_state_key
        
        for official_district in ALL_DISTRICTS:
            official_lower = official_district.lower()
            
            for test_name in [raw_district, translated_district]:
                ratio_score = fuzz.ratio(test_name, official_lower)
                partial_score = fuzz.partial_ratio(test_name, official_lower)
                token_sort_score = fuzz.token_sort_ratio(test_name, official_lower)
                token_set_score = fuzz.token_set_ratio(test_name, official_lower)
                
                current_score = max(ratio_score, partial_score, token_sort_score, token_set_score)
                
                if current_score > best_global_score:
                    best_global_score = current_score
                    best_global_district = official_district
                    best_global_state = DISTRICT_TO_STATE[official_district]
        
        # Use global match if it's significantly better
        if best_global_score >= 75 and best_global_score > best_score:
            if learn and best_global_score >= 85:
                add_to_aliases(best_global_district, raw_district, 'districts')
                add_to_aliases(best_global_state, raw_state, 'states')
            return best_global_state, best_global_district, best_global_score
    
    # Return the best match we found
    if best_score >= min_confidence:
        return matched_state_key, best_district, best_score
    
    return matched_state_key, raw_district, 0.0

# Helper function to manually add aliases
def add_alias(raw_name, correct_name, alias_type='districts'):
    """Manually add an alias to the learning system"""
    add_to_aliases(correct_name, raw_name, alias_type)
    print(f"Added {alias_type[:-1]} alias: '{raw_name}' → '{correct_name}'")

# Helper to view learned aliases
def show_aliases():
    """Display all learned aliases"""
    print("\n=== Learned State Aliases ===")
    for correct, aliases in ALIASES['states'].items():
        print(f"  {correct}:")
        for alias in aliases:
            print(f"    - {alias}")
    
    print("\n=== Learned District Aliases ===")
    for correct, aliases in ALIASES['districts'].items():
        print(f"  {correct}:")
        for alias in aliases:
            print(f"    - {alias}")
    
    total_state_aliases = sum(len(v) for v in ALIASES['states'].values())
    total_district_aliases = sum(len(v) for v in ALIASES['districts'].values())
    print(f"\nTotal: {len(ALIASES['states'])} states ({total_state_aliases} aliases), {len(ALIASES['districts'])} districts ({total_district_aliases} aliases)")