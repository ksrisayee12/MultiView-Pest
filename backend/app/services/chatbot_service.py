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
        "display_name": "Healthy Tomato Foliage (Optimum Vitality)",
        "pathogen": "None detected (Normal Botanical Function)",
        "causes": "Balanced nutrition, optimal irrigation, adequate spacing, and absence of pathogenic foliar lesions.",
        "symptoms": "Uniform deep green leaf lamina, turgid petioles, normal leaflet margins without necrotic spotting or viral chlorosis.",
        "treatment_cultural": [
            "Maintain standard drip irrigation schedule (1.5 to 2.0 inches per week depending on soil evapotranspiration).",
            "Ensure regular balanced fertilization (N-P-K 5-10-10 or 10-10-10 plus trace calcium to prevent blossom end rot).",
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
    }
}


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
    "phosphorus", "potassium", "ph", "mulch", "yield", "canopy", "stunt", "wilt"
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
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
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
        "'I am AgriHealth Advisor, specialized exclusively in plant science and crop health. I cannot answer queries outside the botanical and agricultural domain.'\n"
        "3. Provide clear, practical, scientifically accurate recommendations with bullet points and bold headers."
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
    # Normalize key
    matched_key = "healthy"
    for k in _DISEASE_KNOWLEDGEBASE:
        if k in disease_key:
            matched_key = k
            break
            
    kb = _DISEASE_KNOWLEDGEBASE.get(matched_key, _DISEASE_KNOWLEDGEBASE["healthy"])
    crop_type = context.get("crop_type", "Tomato")
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
        f"3. If the user asks about an unrelated non-agricultural topic, decline firmly: 'This clinical consultation is locked to the active diagnosis of {kb['display_name']}. I can only provide guidance on its etiology, causal pathogen, treatment protocols, and preventative management.'\n"
        f"4. Format your advice clearly with structured headings, actionable numbered steps, and safety advisories."
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
