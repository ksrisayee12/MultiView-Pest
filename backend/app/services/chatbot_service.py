"""
backend/app/services/chatbot_service.py

Dual-Engine AI Chatbot for AgriHealth:
  1. General Plant & Agronomy Assistant (Strictly guardrailed to plant science, botany, crops, soil, pests).
  2. Clinical Disease & Treatment Consultant (Context-locked to the diagnosed disease and plant session).

Powered by Groq API (llama-3.3-70b-versatile) with graceful offline clinical fallback when API key is pending.
"""

import os
import re
import json
import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Load local disease recommendations as grounded clinical baseline
_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
_REC_YAML = _BASE_DIR / "configs" / "recommendations.yaml"

# Local knowledgebase for disease pathology, cause, and cure
_DISEASE_KNOWLEDGEBASE: Dict[str, Dict[str, Any]] = {
    "early_blight": {
        "display_name": "Early Blight (Alternaria solani)",
        "pathogen": "Alternaria solani (Fungal Pathogen)",
        "causes": (
            "Overwinters in infected crop residue, soil, or solanaceous weeds. "
            "Favored by warm temperatures (24-29°C / 75-85°F) and alternating wet and dry periods. "
            "Spreads primarily through rain splash, overhead irrigation, and windblown conidia."
        ),
        "symptoms": (
            "Characteristic concentric brown-to-black rings ('target board' pattern) on older foliage, "
            "surrounded by a chlorotic yellow halo. Causes progressive lower defoliation."
        ),
        "treatment_cultural": [
            "Prune and safely dispose of all lower foliage showing concentric target spots (do not compost).",
            "Discontinue overhead sprinkler irrigation immediately; switch to drip or base soaker lines.",
            "Sterilize shears between cuts using 70% isopropyl alcohol."
        ],
        "treatment_organic": [
            "Apply copper octanoate (copper soap fungicide) or liquid copper at 7-10 day intervals.",
            "Foliar spray with Bacillus amyloliquefaciens or Bacillus subtilis bio-fungicide.",
            "Drench root zone with Trichoderma harzianum to suppress soil-borne fungal inocula."
        ],
        "treatment_chemical": [
            "Chlorothalonil (Bravo/Daconil) foliar spray as a multi-site preventative protectant.",
            "Mancozeb (FRAC M03) applied prior to anticipated humid rain events.",
            "Azoxystrobin or Pyraclostrobin (FRAC 11) for systemic curative suppression in severe outbreaks."
        ],
        "prevention": (
            "Rotate crops with non-solanaceous species (corn, beans) on a 3-year cycle. "
            "Apply 2-3 inches of straw or organic mulch to prevent rain-splash from soil onto lower leaves. "
            "Ensure 24-36 inch plant spacing to optimize foliar airflow."
        )
    },
    "late_blight": {
        "display_name": "Late Blight (Phytophthora infestans)",
        "pathogen": "Phytophthora infestans (Oomycete Water Mold)",
        "causes": (
            "Devastating water mold spreading rapidly under cool (15-21°C / 60-70°F), persistently wet (>90% RH) conditions. "
            "Biflagellate zoospores swim in surface moisture films and can kill entire plantings within 7-14 days."
        ),
        "symptoms": (
            "Large, irregular water-soaked pale-green to brown lesions on leaves and stems. "
            "White velvety fungal down/spores visible on the leaf undersides in high humidity. Fruit develops greasy brown rot."
        ),
        "treatment_cultural": [
            "Immediately bag and remove severely infected plants in dry weather to limit spore dispersal.",
            "Destroy volunteer tomato and potato cull piles within 500 meters of the field.",
            "Keep foliage completely dry; greenhouse vent fans should run continuously to drop relative humidity below 80%."
        ],
        "treatment_organic": [
            "Fixed copper hydroxide sprays applied preventatively before rain events.",
            "Potassium phosphite (phosphonate) foliar drench to trigger systemic acquired resistance (SAR)."
        ],
        "treatment_chemical": [
            "Mandipropamid (Revus) or Cyazofamid (Ranman) - highly targeted oomycete penetrants.",
            "Mefenoxam / Metalaxyl where resistant strains are not documented.",
            "Fluopicolide (Presidio) tank-mixed with protectant chlorothalonil."
        ],
        "prevention": (
            "Plant late-blight resistant cultivars (e.g., Mountain Magic, Defiant Ph-R, Iron Lady). "
            "Track local BLITECAST or university disease forecasting alerts. Never save seed from infected crops."
        )
    },
    "septoria_leaf_spot": {
        "display_name": "Septoria Leaf Spot (Septoria lycopersici)",
        "pathogen": "Septoria lycopersici (Fungal Ascomycete)",
        "causes": (
            "Pycnidia release conidia in gelatinous matrices during rainy spells. "
            "Optimal growth at 20-25°C with leaf wetness periods exceeding 8 hours. "
            "Insects, tools, and farm workers also physically vector pycnidiospores."
        ),
        "symptoms": (
            "Numerous small (2-3 mm) circular spots with dark brown margins and sunken ash-gray centers. "
            "Tiny black specks (pycnidia fruiting bodies) visible in the center of mature spots."
        ),
        "treatment_cultural": [
            "Remove diseased bottom leaves up to the first healthy fruit cluster.",
            "Avoid touching foliage when wet with dew or irrigation.",
            "Disinfect stakes and trellising wire with 10% sodium hypochlorite before reuse."
        ],
        "treatment_organic": [
            "Bordeaux mixture (copper sulfate + hydrated lime).",
            "Sulfur-based dusts or bio-rational Serenade (Bacillus subtilis)."
        ],
        "treatment_chemical": [
            "Chlorothalonil preventative spray every 7 to 10 days during wet weather.",
            "Boscalid (Endura) or Difenoconazole (Quadris Top) for targeted curative inhibition."
        ],
        "prevention": (
            "Two-year minimum crop rotation. Thorough autumn tillage or residue removal to bury overwintered pycnidia."
        )
    },
    "bacterial_spot": {
        "display_name": "Bacterial Spot (Xanthomonas perforans / euvesicatoria)",
        "pathogen": "Xanthomonas spp. (Gram-Negative Plant Bacterium)",
        "causes": (
            "Introduced via contaminated seed or infected transplants. "
            "Enters through natural stomata and mechanical wounds. Propagated by driven rain, warm humid weather (24-30°C)."
        ),
        "symptoms": (
            "Small (less than 3 mm), angular, dark brown greasy-looking spots on leaves and stems. "
            "Leaf spots rarely exceed 3 mm; centers do not turn gray. Severe infections cause foliar drop and sunscalded fruit."
        ),
        "treatment_cultural": [
            "Do not work in fields when foliage is wet.",
            "Eliminate solanaceous weeds (nightshade, horsenettle) which serve as wild bacterial reservoirs."
        ],
        "treatment_organic": [
            "Copper hydroxide combined with mancozeb (copper-mancozeb synergism breaks bacterial resistance).",
            "Actigard (Acibenzolar-S-methyl) plant defense activator.",
            "Bacteriophage biopesticides (AgriPhage) specific to Xanthomonas strains."
        ],
        "treatment_chemical": [
            "Streptomycin sprays permitted only in seedbed nurseries prior to transplant.",
            "Copper-based bactericides combined with ethylene bisdithiocarbamate (EBDC)."
        ],
        "prevention": (
            "Use only certified hot-water treated or hydrochloric acid-treated seeds. "
            "Select resistant cultivars containing Bs2, Bs3, or Bs4 genes."
        )
    },
    "yellow_leaf_curl": {
        "display_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "pathogen": "Begomovirus (Whitefly-transmitted circular ssDNA Geminivirus)",
        "causes": (
            "Strictly transmitted by the sweetpotato/silverleaf whitefly (Bemisia tabaci). "
            "Not seed-borne or sap-transmissible by pruners. High whitefly populations in warm weather drive explosive transmission."
        ),
        "symptoms": (
            "Stunted, erect, bushy plant habit. Leaflets are severely reduced in size, curled upward with marginal chlorosis (yellowing). "
            "Heavy flower abscission; fruit set fails almost completely after infection."
        ),
        "treatment_cultural": [
            "Rogue out (uproot and bag) infected plants immediately upon symptom detection to halt whitefly feeding reservoirs.",
            "Install yellow sticky cards at canopy height (1 trap per 100 sq meters) for population monitoring.",
            "Use 50-mesh insect-exclusion netting over greenhouse ventilation."
        ],
        "treatment_organic": [
            "Foliar sprays of cold-pressed neem oil (1% emulsion) or horticultural insecticidal soaps targeting whitefly nymphs.",
            "Release biological parasitic wasps (Encarsia formosa or Eretmocerus eremicus).",
            "Apply reflective silver mulches to disorient incoming flying whiteflies."
        ],
        "treatment_chemical": [
            "Systemic neonicotinoids (Imidacloprid, Thiamethoxam) at transplant drench.",
            "Spiromesifen (Oberon) or Pyriproxyfen (Knack) insect growth regulators targeting whitefly egg and nymph stages."
        ],
        "prevention": (
            "Plant TYLCV-resistant tomato varieties bearing Ty-1, Ty-2, or Ty-3 resistance loci. "
            "Maintain a strict 2-month host-free crop break between tomato cultivation seasons."
        )
    },
    "healthy": {
        "display_name": "Healthy Foliage (Optimum Vitality)",
        "pathogen": "None detected (Normal Botanical Function)",
        "causes": "Balanced nutrition, optimal irrigation, adequate spacing, and absence of pathogenic foliar lesions.",
        "symptoms": "Uniform deep green leaf lamina, turgid petioles, normal leaflet margins without necrotic spotting, viral chlorosis, or insect feeding holes.",
        "treatment_cultural": [
            "Maintain standard drip irrigation schedule (1.5 to 2.0 inches per week depending on soil evapotranspiration).",
            "Ensure regular balanced fertilization (N-P-K 5-10-10 or 10-10-10 plus trace calcium to prevent deficiency).",
            "Sucker lower indeterminate growth to encourage upward airflow."
        ],
        "treatment_organic": [
            "Apply bi-weekly compost tea or fish hydrolysate foliar misting to maintain beneficial phyllosphere microbiota.",
            "Preventative potassium silicate foliar drench to thicken plant epidermal cell walls."
        ],
        "treatment_chemical": [
            "No chemical intervention needed or advised."
        ],
        "prevention": (
            "Continue routine weekly scoutings of upper and underside foliage. "
            "Keep soil mulch barrier intact to prevent future soil-borne pathogen splash."
        )
    },
    "pest_chewing_damage": {
        "display_name": "Foliar Pest Chewing & Defoliation Damage",
        "pathogen": "Chewing Insect Herbivores (Caterpillars / Flea Beetles / Cutworms / Hornworms)",
        "causes": (
            "Direct foliar consumption by phytophagous insect larvae (e.g. Spodoptera armyworms, loopers, Manduca hornworms) "
            "or adult flea beetles piercing leaf laminae and creating physical feeding holes."
        ),
        "symptoms": (
            "Irregular circular perforations through the leaf blade (shot-hole damage), ragged scalloped leaf margins, "
            "and localized insect frass droppings, without concentric fungal rings or bacterial water-soaked lesions."
        ),
        "treatment_cultural": [
            "Hand-pick visible caterpillars and larvae from the undersides of leaves during early morning or twilight hours.",
            "Install insect exclusion netting (mesh size < 0.8 mm) or floating row covers over beds to prevent oviposition.",
            "Weed host areas surrounding planting beds and till soil lightly post-harvest to expose pupae to predatory birds."
        ],
        "treatment_organic": [
            "Foliar spray of Bacillus thuringiensis (Bt subsp. kurstaki) @ 2.0 - 2.5 g/L during active early-instar larval feeding.",
            "Cold-pressed Neem Oil (0.5% Azadirachtin) @ 5 mL/L with mild surfactant for feeding deterrence and growth inhibition.",
            "Spinosad biological spray @ 0.5 mL/L targeting canopy undersides where young larvae feed."
        ],
        "treatment_chemical": [
            "Chlorantraniliprole (Coragen / Rynaxypyr) @ 0.3 mL/L for targeted ovi-larvicidal protection with low beneficial insect impact.",
            "Emamectin Benzoate (5% SG) @ 0.4 g/L for rapid knock-down of chewing lepidopteran caterpillars.",
            "Lambda-cyhalothrin @ 1.0 mL/L as a rescue foliar treatment if severe defoliation exceeds 20% of canopy."
        ],
        "prevention": (
            "Erect pheromone and yellow sticky traps to monitor adult moth and beetle flights before egg laying begins. "
            "Encourage natural predators including parasitic wasps (Trichogramma), predatory stink bugs, and assassin bugs."
        )
    },
    "pepper_bacterial_spot": {
        "display_name": "Pepper Bacterial Spot (Xanthomonas euvesicatoria)",
        "pathogen": "Xanthomonas campestris pv. vesicatoria / X. euvesicatoria",
        "causes": (
            "Seed-borne or wind-splashed bacterial pathogen thriving in warm (24-30°C), humid, and wet rainy conditions."
        ),
        "symptoms": (
            "Small, water-soaked, circular to irregular dark brown spots on pepper leaves that turn necrotic with raised margins and chlorotic halos, causing premature defoliation."
        ),
        "treatment_cultural": [
            "Use certified pathogen-free pepper seeds and hot-water treated seeds (50°C for 25 minutes).",
            "Avoid overhead sprinkler irrigation; convert to subsurface drip irrigation to keep foliage completely dry.",
            "Immediately rogue out and destroy severely infected pepper plants; do not compost."
        ],
        "treatment_organic": [
            "Fixed copper hydroxide / copper octanoate sprays @ 2-3 g/L weekly during wet periods.",
            "Bacillus subtilis / B. amyloliquefaciens foliar biofungicides as protective bio-barriers."
        ],
        "treatment_chemical": [
            "Copper Hydroxide + Mancozeb tank mix (2.5 g/L + 2.0 g/L) to manage copper-tolerant Xanthomonas strains.",
            "Kasugamycin or Streptomycin sulphate where permitted by local agricultural regulations for greenhouse transplants."
        ],
        "prevention": (
            "Enforce strict 2-3 year rotation away from solanaceous crops (peppers, tomatoes, eggplants). "
            "Disinfect staking, cages, pruning shears, and greenhouse trays with 10% sodium hypochlorite."
        )
    },
    "pepper_healthy": {
        "display_name": "Healthy Pepper / Chilli Foliage (Optimum Vitality)",
        "pathogen": "None detected (Healthy Pepper / Chilli Foliage)",
        "causes": "Optimal soil nutrients, balanced nitrogen-phosphorus-potassium, and proper solar radiation.",
        "symptoms": "Vibrant green, uniform leaf blades without chlorosis, necrotic spotting, or feeding perforations.",
        "treatment_cultural": [
            "Maintain consistent soil moisture and adequate calcium nutrition to prevent blossom-end rot.",
            "Provide light shade netting during extreme midday heat peaks above 35°C."
        ],
        "treatment_organic": [
            "Apply balanced organic seaweed extract and fish emulsion every 2 weeks.",
            "Maintain neem oil preventative barrier on foliage."
        ],
        "treatment_chemical": [
            "No chemical intervention required."
        ],
        "prevention": (
            "Routine canopy scouting and prophylactic neem oil maintenance. Maintain clean weed-free perimeter."
        )
    },
    "potato_early_blight": {
        "display_name": "Potato Early Blight (Alternaria solani)",
        "pathogen": "Alternaria solani (Fungal Deuteromycete)",
        "causes": (
            "Fungal spores surviving on crop debris, spread by wind, insects, and rain splashing under alternating wet and dry conditions."
        ),
        "symptoms": (
            "Dark brown to black lesions with characteristic concentric target-board rings on older potato leaflets, surrounded by yellow chlorotic halos."
        ),
        "treatment_cultural": [
            "Ensure adequate nitrogen and potassium fertility to avoid plant stress; destroy infected haulms before tuber harvest.",
            "Maintain wide plant spacing to accelerate canopy drying."
        ],
        "treatment_organic": [
            "Bordeaux mixture (1%) or bio-control with Trichoderma harzianum soil and foliar drench.",
            "Liquid copper octanoate protectant."
        ],
        "treatment_chemical": [
            "Azoxystrobin (250 SC) @ 1 mL/L or Difenoconazole @ 0.5 mL/L rotated with Chlorothalonil protectant.",
            "Mancozeb 75% WP @ 2.0 g/L preventative spray."
        ],
        "prevention": (
            "Plant certified disease-free seed tubers and adhere to a 3-year potato rotation away from nightshade crops."
        )
    },
    "potato_late_blight": {
        "display_name": "Potato Late Blight (Phytophthora infestans)",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "causes": (
            "Cool (15-20°C), saturated, high-humidity weather with rain or persistent dew."
        ),
        "symptoms": (
            "Rapidly expanding water-soaked pale-green to brown lesions on potato foliage with white fluffy fungal mildew on leaf undersides in high humidity."
        ),
        "treatment_cultural": [
            "Hill up potato rows with soil to form a thick barrier protecting underground tubers from wash-down sporangia; burn blighted vines before digging.",
            "Destroy volunteer potato plants and cull piles within 1 km."
        ],
        "treatment_organic": [
            "Preventative copper hydroxide sprays @ 2.5 g/L every 5-7 days under wet forecast.",
            "Potassium phosphite foliar SAR stimulator."
        ],
        "treatment_chemical": [
            "Metalaxyl-M + Mancozeb (Ridomil Gold) @ 2.5 g/L or Dimethomorph @ 1.5 g/L curative systemic spray.",
            "Cyazofamid or Mandipropamid anti-oomycete applications."
        ],
        "prevention": (
            "Plant late-blight resistant potato cultivars and monitor regional blight warning networks."
        )
    },
    "potato_healthy": {
        "display_name": "Healthy Potato Foliage (Optimum Vitality)",
        "pathogen": "None detected (Healthy Potato Foliage)",
        "causes": "Optimal tuber bulking conditions with balanced soil nutrition.",
        "symptoms": "Lush compound leaves, uniform green coloration, and vigorous vegetative growth without lesion blight.",
        "treatment_cultural": [
            "Regular hilling up around potato stems to shield tubers and retain moisture.",
            "Maintain moderate, consistent drip fertigation."
        ],
        "treatment_organic": [
            "Compost tea root zone conditioning."
        ],
        "treatment_chemical": [
            "No chemical treatment required."
        ],
        "prevention": (
            "Strict 3-year crop rotation and use of certified seed tubers."
        )
    },
    "leaf_mold": {
        "display_name": "Leaf Mold (Passalora fulva)",
        "pathogen": "Passalora fulva (syn. Cladosporium fulvum)",
        "causes": "High relative humidity (>85%) and poor air circulation in greenhouse or dense canopy environments.",
        "symptoms": "Pale greenish-yellow spots on upper leaf surfaces matching olive-green to velvety brown mold growth on lower leaf surfaces.",
        "treatment_cultural": [
            "Increase greenhouse ventilation and run exhaust fans to reduce relative humidity below 80%.",
            "Prune lower canopy leaves and thin excessive foliage to increase light penetration."
        ],
        "treatment_organic": [
            "Potassium bicarbonate (3-5 g/L) or copper octanoate foliar spray.",
            "Bacillus subtilis bio-fungicide."
        ],
        "treatment_chemical": [
            "Difenoconazole @ 0.5 mL/L or Cyprodinil + Fludioxonil protectant.",
            "Chlorothalonil preventative barrier."
        ],
        "prevention": (
            "Plant resistant varieties containing Cf resistance genes; keep canopy dry through night hours."
        )
    },
    "spider_mites": {
        "display_name": "Two-Spotted Spider Mite Infestation (Tetranychus urticae)",
        "pathogen": "Tetranychus urticae (Phytophagous Arachnid)",
        "causes": "Hot, dry, dusty environmental conditions triggering rapid spider mite population growth.",
        "symptoms": "Fine pale-yellow stippling / chlorotic speckling on upper leaf surfaces, fine silken webbing under leaves, leaf bronzing.",
        "treatment_cultural": [
            "Misting or sprinkler washing of dusty roadways and garden margins to lower dust stress.",
            "Prune and dispose of heavily webbed shoot tips immediately."
        ],
        "treatment_organic": [
            "Release predatory mites (Phytoseiulus persimilis or Neoseiulus californicus).",
            "Cold-pressed neem oil (1%) or insecticidal potassium salts of fatty acids."
        ],
        "treatment_chemical": [
            "Spiromesifen (Oberon) @ 1 mL/L or Abamectin (1.8% EC) @ 0.5 mL/L targeting leaf undersides.",
            "Bifenazate (Acramite) selective miticide."
        ],
        "prevention": (
            "Avoid broad-spectrum synthetic pyrethroids that destroy beneficial mite predators."
        )
    },
    "target_spot": {
        "display_name": "Target Spot (Corynespora cassiicola)",
        "pathogen": "Corynespora cassiicola (Fungal Pathogen)",
        "causes": "Warm temperatures (20-28°C) accompanied by extended periods of high humidity and leaf moisture.",
        "symptoms": "Pinpoint brown necrotic lesions expanding into circular spots with concentric rings and distinct yellow halos.",
        "treatment_cultural": [
            "Improve row spacing and prune lower leaves to optimize canopy drying.",
            "Avoid overhead irrigation."
        ],
        "treatment_organic": [
            "Copper oxychloride @ 2.5 g/L or sulfur-based protectants.",
            "Bio-fungicides based on Bacillus amyloliquefaciens."
        ],
        "treatment_chemical": [
            "Pyraclostrobin + Boscalid @ 1 g/L or Azoxystrobin @ 1 mL/L.",
            "Chlorothalonil protectant."
        ],
        "prevention": (
            "Implement a 2-year non-solanaceous crop rotation and maintain clean organic mulch."
        )
    },
    "mosaic_virus": {
        "display_name": "Mosaic Virus (ToMV / TMV)",
        "pathogen": "Tobacco Mosaic / Tomato Mosaic Tobamovirus",
        "causes": "Mechanical transmission via workers' hands, contaminated tools, tobacco products, and seeds.",
        "symptoms": "Mottled light and dark green mosaic patterns on foliage, leaf distortion, blistering, shoestring leaves, and stunted growth.",
        "treatment_cultural": [
            "Immediately rogue out infected plants in sealed plastic bags; prohibit smoking or tobacco handling around plants.",
            "Disinfect all tools and stakes in 20% nonfat dry milk or 10% TSP solution."
        ],
        "treatment_organic": [
            "Milk spray (20% nonfat milk in water) during handling to inactivate tobamovirus particles on hands."
        ],
        "treatment_chemical": [
            "No chemical cure exists for plant viruses; infected plants must be eliminated to protect adjacent crops."
        ],
        "prevention": (
            "Select certified virus-resistant varieties (Tm-2 or Tm-2^2 resistance alleles)."
        )
    },
    "unclassified_foliar_damage": {
        "display_name": "Unclassified Foliar Stress / Feeding Damage",
        "pathogen": "Mechanical Abrasion, Environmental Stress, or Incipient Pest Feeding",
        "causes": "Physical leaf wounding, wind/hail friction, or non-specific foliar abiotic stress.",
        "symptoms": "Atypical leaf tears, physical perforations, or superficial chlorosis without concentric fungal rings or bacterial halos.",
        "treatment_cultural": [
            "Inspect surrounding canopy for active insect presence or physical abrasion sources.",
            "Ensure regular irrigation and avoid drought or nutrient extremes."
        ],
        "treatment_organic": [
            "Apply preventative cold-pressed neem oil @ 5 mL/L as a protective feeding barrier."
        ],
        "treatment_chemical": [
            "No chemical fungicides needed unless clear fungal sporulation develops."
        ],
        "prevention": (
            "Maintain balanced fertigation and re-inspect foliage in 48 hours to confirm condition stability."
        )
    }
}


def get_condition_profile(condition_key: str, crop_type: str = "tomato") -> Dict[str, Any]:
    """
    Returns the comprehensive clinical pathology profile for a condition key,
    dynamically adapted to the specified crop type so non-tomato crops never
    receive hardcoded tomato nomenclature.
    """
    c_lower = (crop_type or "tomato").lower()
    crop_display = (
        "Pepper / Chilli" if any(p in c_lower for p in ("pepper", "chilli", "capsicum"))
        else "Potato" if "potato" in c_lower
        else "Tomato" if "tomato" in c_lower
        else "Plant"
    )

    # 1. Match direct key
    if condition_key in _DISEASE_KNOWLEDGEBASE:
        kb = dict(_DISEASE_KNOWLEDGEBASE[condition_key])
    else:
        # Check normalized aliases
        alias_map = {
            "earlyblight": "early_blight",
            "lateblight": "late_blight",
            "septoria": "septoria_leaf_spot",
            "bacterialspot": "bacterial_spot",
            "tylcv": "yellow_leaf_curl",
            "leafmold": "leaf_mold",
            "spidermites": "spider_mites",
            "targetspot": "target_spot",
            "mosaic": "mosaic_virus",
            "chewing": "pest_chewing_damage",
            "holes": "pest_chewing_damage",
            "pest": "pest_chewing_damage",
        }
        matched_k = alias_map.get(condition_key.replace("_", "").lower(), "healthy")
        kb = dict(_DISEASE_KNOWLEDGEBASE.get(matched_k, _DISEASE_KNOWLEDGEBASE["healthy"]))

    # 2. Dynamic crop adaptation for healthy and general profiles
    if condition_key in ("healthy", "pepper_healthy", "potato_healthy"):
        kb["display_name"] = f"Healthy {crop_display} Foliage (Optimum Vitality)"
        kb["pathogen"] = f"None detected (Healthy {crop_display} Foliage)"
    elif condition_key == "pest_chewing_damage":
        kb["display_name"] = f"{crop_display} Pest Chewing & Defoliation Damage"
    elif condition_key == "unclassified_foliar_damage":
        kb["display_name"] = f"{crop_display} Foliar Stress / Irregularity"

    return kb



def _get_groq_api_key() -> str:
    """Retrieve Groq API key from environment, stripping placeholders and quotes."""
    from backend.app.config.settings import settings
    key = (settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")).strip().strip('"').strip("'")
    if not key or key.lower() in ("placeholder", "your_groq_api_key_here", "gsk_placeholder", "none"):
        return ""
    return key


# ---------------------------------------------------------------------------
# Strict Guardrail Verifiers
# ---------------------------------------------------------------------------
_PLANT_TERMS = {
    "plant", "leaf", "leaves", "crop", "tomato", "foliage", "disease", "pest",
    "fungus", "fungicide", "bacteria", "bacterial", "virus", "viral", "soil",
    "fertilizer", "compost", "irrigation", "water", "watering", "blight",
    "septoria", "mildew", "rot", "stem", "root", "flower", "fruit", "harvest",
    "pruning", "aphid", "caterpillar", "whitefly", "beetle", "cutworm", "pot",
    "garden", "greenhouse", "agriculture", "agronomy", "botany", "seed",
    "seedling", "chlorosis", "necrosis", "lesion", "spore", "pathogen", "cure",
    "spray", "neem", "copper", "mancozeb", "chlorothalonil", "nitrogen", "calcium",
    "phosphorus", "potassium", "ph", "mulch", "yield", "canopy", "stunt", "wilt",
    # Crop-type expansions (pepper, chilli, potato, etc.)
    "pepper", "chilli", "chili", "capsicum", "potato", "paprika", "eggplant",
    "nightshade", "solanaceous", "solanum", "lycopersicon", "tuberosum",
    "herbicide", "insecticide", "miticide", "fungal", "oomycete", "nematode",
    "drip", "fertigation", "mulching", "intercrop", "rotation", "tillage",
    "horticulture", "defoliation", "perforation", "feeding", "infestation",
    "armyworm", "hornworm", "flea", "cutworm", "thrip", "mite", "whitefly",
}

_FORBIDDEN_DOMAINS = [
    r"\b(code|python|javascript|java|c\+\+|html|css|sql|function|script|algorithm)\b",
    r"\b(crypto|bitcoin|ethereum|forex|stock|market|trade|investment)\b",
    r"\b(movie|actor|film|song|music|hollywood|celebrity)\b",
    r"\b(politics|election|president|biden|trump|congress|parliament)\b",
    r"\b(recipe|cake|pizza|burger|cocktail|bake|cook)\b",
    r"\b(football|soccer|cricket|nba|nfl|tennis|match|game)\b",
]

def is_plant_related(text: str) -> bool:
    """Evaluates whether the user's prompt is within plant, botanical, and agronomic domain."""
    clean = text.lower()
    
    # Check for direct plant keywords or stems (handles plurals: plants, tomatoes, pests, weeds)
    words = set(re.findall(r"\b[a-z]{3,}\b", clean))
    has_plant_word = any(any(w.startswith(t) or (len(w) >= 4 and t.startswith(w)) for t in _PLANT_TERMS) for w in words)
    
    # Check for obvious forbidden non-plant topics
    for pattern in _FORBIDDEN_DOMAINS:
        if re.search(pattern, clean) and not has_plant_word:
            return False
            
    # If the user asks general agronomic queries
    if has_plant_word:
        return True
        
    # Catch greetings and basic assistant interactions
    if re.search(r"^(hi|hello|hey|help|who are you|what can you do)", clean):
        return True
        
    return False


# ---------------------------------------------------------------------------
# Groq Candidate Models & Resilient Fallback Engine
# ---------------------------------------------------------------------------
_GROQ_CANDIDATE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]


async def _call_groq_with_fallback(
    api_key: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1000
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Executes chat completion with automatic model fallback cascading if a model is unavailable.
    Returns: (reply_text, active_model_name, error_message)
    """
    preferred = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
    models_to_try = [preferred] + [m for m in _GROQ_CANDIDATE_MODELS if m != preferred]
    
    last_error = ""
    async with httpx.AsyncClient(timeout=35.0) as client:
        for model in models_to_try:
            try:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": "AgriHealth-PrecisionClinic/1.0",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return content, model, None
                elif resp.status_code == 404 or "model_not_found" in resp.text:
                    last_error = f"Model '{model}' not found on endpoint"
                    continue
                else:
                    last_error = f"Status {resp.status_code}: {resp.text}"
            except Exception as e:
                last_error = str(e)
                continue

    return None, None, last_error


# ---------------------------------------------------------------------------
# General Plant Chatbot Handler
# ---------------------------------------------------------------------------
async def query_general_plant_bot(message: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Answers general plant, crop, soil, and cultivation questions.
    Strictly guardrailed against non-plant topics.
    """
    history = history or []
    
    # 1. Guardrail verification
    if not is_plant_related(message):
        return {
            "reply": (
                "🌿 **AgriHealth Plant Advisory Guardrail Notice**\n\n"
                "I am **AgriHealth Advisor**, specialized exclusively in botanical science, crop health, "
                "agronomy, soil fertility, and pest management.\n\n"
                "I cannot answer questions outside the plant and agricultural domain. "
                "Please ask me about crops, disease diagnostics, fertilizing schedules, pest control, or soil management!"
            ),
            "guardrail_triggered": True,
            "engine": "guardrail_filter",
            "model": "rule_based"
        }

    api_key = _get_groq_api_key()
    
    # 2. Check if Groq API key is present
    if not api_key:
        return {
            "reply": (
                "🌿 **AgriHealth Botanical Assistant (Offline Mode)**\n\n"
                f"Thank you for your agronomic question: *\"{message}\"*\n\n"
                "**Quick Guidance:**\n"
                "- Ensure consistent soil moisture (avoid wetting foliage to limit fungal spores).\n"
                "- Maintain balanced nutrition (NPK) with micronutrients (Ca, Mg, Fe).\n"
                "- Inspect leaf undersides regularly for early vector presence (aphids, whiteflies).\n\n"
                "> 💡 **To unlock full conversational AI:**\n"
                "> Paste your free Groq API key into `.env`:\n"
                "> `GROQ_API_KEY=\"gsk_...\"`\n"
                "> and restart the application."
            ),
            "guardrail_triggered": False,
            "engine": "offline_fallback",
            "model": "local_botanical_guide"
        }

    # 3. Call Groq API via httpx
    system_prompt = (
        "You are AgriHealth Advisor, an authoritative agricultural scientist, botanical researcher, "
        "and field agronomist.\n\n"
        "STRICT GUARDRAIL RULES:\n"
        "1. You ONLY answer questions related to plants, botany, crop cultivation, agricultural pests, "
        "soil chemistry, plant pathology, irrigation, fertilizers, and greenhouse farming.\n"
        "2. If the user asks ANY question outside of plants and agriculture, you MUST decline with:\n"
        "'I am AgriHealth Advisor, specialized exclusively in plant science and crop health. I cannot answer queries outside the botanical and agricultural domain.'\n\n"
        "MANDATORY OUTPUT FORMATTING RULES:\n"
        "- Structure your responses using rich Markdown: '## Primary Topics' and '### Subsections'.\n"
        "- Use bold attribute lead-ins: '- **Technique:** ...', '- **Dosage:** ...', '- **Timing:** ...'.\n"
        "- When presenting plant lists, schedules, or spray programs, ALWAYS format them as clean Markdown tables with headers, ensuring every table row is on its own separate line:\n"
        "  | Crop / Stage | Recommendation | Method / Dosage | Timing & Benefits |\n"
        "  |---|---|---|---|\n"
        "  | Item 1 | Description | Application rate | When to apply |\n"
        "- Use checkboxes for action checklists: '- [ ] Practical step'.\n"
        "- Never concatenate table rows onto one line. Use double line breaks between paragraphs."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-6:]:
        messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
    messages.append({"role": "user", "content": message})

    reply, used_model, err = await _call_groq_with_fallback(
        api_key=api_key,
        messages=messages,
        temperature=0.25,
        max_tokens=800,
    )
    if reply:
        return {
            "reply": reply,
            "guardrail_triggered": False,
            "engine": "groq_api",
            "model": used_model,
        }
    else:
        return {
            "reply": (
                f"🌿 **AgriHealth Botanical Assistant**\n\n"
                f"We are temporarily routing through our localized clinical knowledgebase.\n\n"
                f"**Regarding your agronomic question:** *\"{message}\"*\n\n"
                "- Maintain balanced irrigation with strict morning watering at soil level to prevent fungal spore dispersal.\n"
                "- Monitor nitrogen-phosphorus-potassium balance and ensure trace calcium to avoid blossom end issues.\n"
                "- Scout foliar leaflets weekly for early signs of necrotic or chlorotic spots.\n\n"
                f"*Note: Groq model cascade notice ({err}). Operating on local botanical guide.*"
            ),
            "guardrail_triggered": False,
            "engine": "offline_fallback",
            "model": "local_botanical_guide",
        }


# ---------------------------------------------------------------------------
# Clinical Disease & Treatment Consultant Handler
# ---------------------------------------------------------------------------
async def query_disease_treatment_bot(
    message: str,
    context: Dict[str, Any],
    history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Provides clinical treatment consultation strictly locked to the diagnosed condition.
    Explains:
      - Pathogen biology & etiology
      - Why it happened / causes
      - Step-by-step treatment protocol (cultural, organic, chemical)
      - Preventative barrier management
    """
    history = history or []
    
    # Extract clinical context
    disease_key = str(context.get("disease_name", "healthy")).strip().lower()
    crop_type = context.get("crop_type", "Tomato")
    kb = get_condition_profile(disease_key, crop_type=crop_type)
    confidence = context.get("confidence", 0.0)
    severity = context.get("severity", "Moderate")
    lesion_pct = context.get("lesion_percentage", 12.5)
    affected_views = context.get("affected_views", ["front", "underside"])

    api_key = _get_groq_api_key()
    
    # If no Groq API key is configured, return the rich localized clinical protocol
    if not api_key:
        cultural_str = "\n".join([f"  • {item}" for item in kb["treatment_cultural"]])
        organic_str = "\n".join([f"  • {item}" for item in kb["treatment_organic"]])
        chemical_str = "\n".join([f"  • {item}" for item in kb["treatment_chemical"]])
        
        reply = (
            f"🔬 **Clinical Pathologist Prescription: {kb['display_name']}**\n\n"
            f"**Diagnostic Baseline:**\n"
            f"- **Host Crop:** {crop_type.title()} | **Assessed Severity:** {severity} ({lesion_pct:.1f}% foliar area)\n"
            f"- **Confidence Interval:** {confidence * 100:.1f}%\n"
            f"- **Observed Viewpoints:** {', '.join(affected_views)}\n\n"
            f"### 🦠 1. Causal Pathogen & Etiology\n"
            f"- **Pathogen:** {kb['pathogen']}\n"
            f"- **What Caused This:** {kb['causes']}\n"
            f"- **Symptom Signs:** {kb['symptoms']}\n\n"
            f"### 📋 2. Actionable Treatment Protocol\n"
            f"**A. Immediate Cultural & Sanitation Actions:**\n{cultural_str}\n\n"
            f"**B. Certified Organic & Biological Controls:**\n{organic_str}\n\n"
            f"**C. Targeted Chemical Interventions (Registered Actives):**\n{chemical_str}\n\n"
            f"### 🛡️ 3. Long-Term Prevention\n"
            f"{kb['prevention']}\n\n"
            f"---\n"
            f"> 💡 *Notice: Currently operating in Clinical Knowledgebase Mode. Add your `GROQ_API_KEY` in `.env` to enable real-time dynamic interactive dialogue with the specialist.*"
        )
        return {
            "reply": reply,
            "disease": kb["display_name"],
            "engine": "clinical_knowledgebase",
            "model": "agronomic_treatment_catalog"
        }

    # Groq API System Prompt Context-Locked to this disease
    system_prompt = (
        f"You are the Chief Agricultural Plant Pathologist at AgriHealth Diagnostic Laboratories.\n"
        f"You are providing an expert clinical consultation for a verified diagnosis:\n\n"
        f"CLINICAL DIAGNOSIS CASE FILE:\n"
        f"- Crop: {crop_type}\n"
        f"- Disease: {kb['display_name']}\n"
        f"- Pathogen: {kb['pathogen']}\n"
        f"- Assessed Severity: {severity} ({lesion_pct:.1f}% estimated foliar lesion area)\n"
        f"- Model Confidence: {confidence * 100:.1f}%\n"
        f"- Affected Camera Viewpoints: {', '.join(affected_views)}\n\n"
        f"STRICT CONSULTATION GUARDRAILS:\n"
        f"1. You are LOCKED to consulting on this specific diagnosis ({kb['display_name']}) and host crop ({crop_type}).\n"
        f"2. You must thoroughly address questions regarding:\n"
        f"   - What caused this disease and what conditions accelerate its spread.\n"
        f"   - Step-by-step cure, recovery, and disease eradication.\n"
        f"   - Specific cultural sanitization, organic biocontrols, and chemical fungicides/active ingredients.\n"
        f"   - Dosages, application timing, intervals, and resistance management (FRAC rotations).\n"
        f"3. If the user asks about an unrelated non-agricultural topic, decline firmly: 'This clinical consultation is locked to the active diagnosis of {kb['display_name']}. I can only provide guidance on its etiology, causal pathogen, treatment protocols, and preventative management.'\n\n"
        f"MANDATORY OUTPUT FORMATTING RULES:\n"
        f"- Structure your output using clear Markdown headers: '## Primary Section' and '### Specific Protocol'.\n"
        f"- Always use bold lead-in tags for attributes: '- **Etiology:** ...', '- **Recommended Active:** ...', '- **Rate:** ...'.\n"
        f"- When presenting schedules, spray programs, or product comparisons, ALWAYS format them as a clean Markdown table with headers, and ensure EVERY row is separated by a newline:\n"
        f"  | Stage / Week | Product Type | Active Ingredient | Application Rate | Application Timing | Remarks |\n"
        f"  |---|---|---|---|---|---|\n"
        f"  | Stage 1 | Foliar Protectant | Copper Hydroxide | 2 L/ha | Early morning | Preventative barrier |\n"
        f"- Use checklists for immediate tasks: '- [ ] Step to complete'.\n"
        f"- Separate distinct paragraphs with blank lines for clean layout rendering."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-6:]:
        messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
    messages.append({"role": "user", "content": message})

    reply, used_model, err = await _call_groq_with_fallback(
        api_key=api_key,
        messages=messages,
        temperature=0.2,
        max_tokens=1000,
    )
    if reply:
        return {
            "reply": reply,
            "disease": kb["display_name"],
            "engine": "groq_api",
            "model": used_model,
        }
    else:
        cultural_str = "\n".join([f"  • {item}" for item in kb["treatment_cultural"]])
        organic_str = "\n".join([f"  • {item}" for item in kb["treatment_organic"]])
        chemical_str = "\n".join([f"  • {item}" for item in kb["treatment_chemical"]])
        
        reply = (
            f"🔬 **Clinical Pathologist Prescription: {kb['display_name']}**\n\n"
            f"**Diagnostic Baseline:**\n"
            f"- **Host Crop:** {crop_type.title()} | **Assessed Severity:** {severity} ({lesion_pct:.1f}% foliar area)\n"
            f"- **Confidence Interval:** {confidence * 100:.1f}%\n"
            f"- **Observed Viewpoints:** {', '.join(affected_views)}\n\n"
            f"### 🦠 1. Causal Pathogen & Etiology\n"
            f"- **Pathogen:** {kb['pathogen']}\n"
            f"- **What Caused This:** {kb['causes']}\n"
            f"- **Symptom Signs:** {kb['symptoms']}\n\n"
            f"### 📋 2. Actionable Treatment Protocol\n"
            f"**A. Immediate Cultural & Sanitation Actions:**\n{cultural_str}\n\n"
            f"**B. Certified Organic & Biological Controls:**\n{organic_str}\n\n"
            f"**C. Targeted Chemical Interventions (Registered Actives):**\n{chemical_str}\n\n"
            f"### 🛡️ 3. Long-Term Prevention\n"
            f"{kb['prevention']}\n"
        )
        return {
            "reply": reply,
            "disease": kb["display_name"],
            "engine": "clinical_knowledgebase_fallback",
            "model": "agronomic_treatment_catalog",
        }
