# MultiView-AgriHealth — Dataset Catalog & Governance

**Date of Catalog Review:** 2026-10-18  
**Scope:** Public candidate and active agricultural datasets for leaf disease identification, pest monitoring, and multi-view plant observation.  
**Governance Policy:** Strictly adheres to reproducible licensing, no naive cross-dataset merging, group-aware partitioning, and immutable manifest logging.

---

## 1. Canonical Label Ontology

To prevent confusion between pests, fungal diseases, viral pathogens, and abiotic stresses, MultiView-AgriHealth maintains a canonical taxonomy for tomato leaf observation:

| Canonical Key | Domain | Common Name | Pathogen / Cause |
|---|---|---|---|
| `healthy` | Baseline | Healthy Tomato Foliage | None (Asymptomatic) |
| `early_blight` | Disease (Fungal) | Early Blight | *Alternaria solani* |
| `late_blight` | Disease (Oomycete) | Late Blight | *Phytophthora infestans* |
| `septoria_leaf_spot` | Disease (Fungal) | Septoria Leaf Spot | *Septoria lycopersici* |
| `bacterial_spot` | Disease (Bacterial) | Bacterial Spot | *Xanthomonas spp.* |
| `yellow_leaf_curl` | Disease (Viral) | Tomato Yellow Leaf Curl | Tomato yellow leaf curl virus (TYLCV) |
| `mosaic_virus` | Disease (Viral) | Tomato Mosaic | Tomato mosaic virus (ToMV) |
| `leaf_mold` | Disease (Fungal) | Leaf Mold | *Passalora fulva* |
| `spider_mites` | Pest Damage | Two-Spotted Spider Mite Damage | *Tetranychus urticae* |
| `no_supported_condition_detected` | Decision Support | No Detected Symptom Box | Unclassified / Below Model Threshold |

---

## 2. Dataset Evaluations & Candidate Matrix

### 2.1 PlantVillage (Tomato Subset)
- **Publisher / Source:** Sharada P. Mohanty, David P. Hughes, Marcel Salathé (Penn State University & EPFL)
- **Official URL:** `https://github.com/spMohanty/PlantVillage-Dataset`
- **Citation / DOI:** Mohanty et al., *Frontiers in Plant Science* (2016), 7:1419.
- **License / Terms:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
- **Target Crop:** Tomato (*Solanum lycopersicum*).
- **Modality:** Controlled laboratory/indoor close-up RGB leaf photography (solid gray/black backdrop).
- **Total Samples:** ~18,160 tomato leaf images across 10 condition classes.
- **Acquisition Status:** `APPROVED` & `DERIVATIVE USED IN PRETRAINING`.
- **Known Limitations:** Controlled laboratory backgrounds create a domain gap when applied to in-the-wild field photography with complex soil/weed backgrounds.

### 2.2 PlantDoc
- **Publisher / Source:** Indian Institute of Technology (IIT) Kharagpur (Singh et al.)
- **Official URL:** `https://github.com/pratikkayal/PlantDoc-Dataset`
- **Citation / DOI:** Singh et al., *CoDS-COMAD* (2020), pp. 249–253.
- **License / Terms:** MIT / Academic Research.
- **Target Crop:** Multi-crop including Tomato.
- **Modality:** In-the-wild field imagery with complex backgrounds, partial occlusions, and varied lighting.
- **Total Samples:** 2,598 images across 13 plant species and 27 disease/healthy classes.
- **Acquisition Status:** `REVIEWED & RECOMMENDED FOR FIELD FINE-TUNING`.
- **Known Limitations:** Variable image resolution, noisy label boundaries in dense foliage.

### 2.3 IP102 Agricultural Pest Benchmark
- **Publisher / Source:** Wu et al. (State Key Laboratory of Software Development, Beihang University)
- **Official URL:** `https://github.com/xpwu95/IP102`
- **Citation / DOI:** Wu et al., *CVPR* (2019), pp. 8778–8787.
- **License / Terms:** Non-commercial research use only.
- **Target Crop:** 8 crops (rice, corn, wheat, beet, alfalfa, millet, citrus, tomato).
- **Modality:** Field RGB images with bounding box annotations.
- **Total Samples:** 75,222 images across 102 insect pest classes.
- **Acquisition Status:** `REVIEWED FOR PEST DETECTION STAGE`.
- **Known Limitations:** Substantial class imbalance; license restricts commercial client delivery without explicit agreement.

### 2.4 TomatoMAP (Tomato Multi-Angle Perception Dataset)
- **Publisher / Source:** IPK Gatersleben / Phenomics Initiative
- **Official URL:** `https://doi.org/10.5447/ipk/2025/14`
- **License / Terms:** Open Access Academic Data License.
- **Target Crop:** Tomato canopy.
- **Modality:** Synchronized multi-angle camera observations (horizontal 360°, top-down nadir, and under-canopy).
- **Acquisition Status:** `REVIEWED FOR MULTI-VIEW CALIBRATION`.
- **Known Limitations:** Large raw video footprint; specialized phenotyping gantry setup.

---

## 3. Data Integrity & Partitioning Rules

1. **No Group Leakage:** Splits must strictly partition at the plant/video/capture-session level. Consecutive video frames or multiple views of the same physical plant must never be divided across train and test sets.
2. **Duplicate Suppression:** Perceptual hashing (pHash) and SHA-256 deduplication are enforced prior to split assignment.
3. **No Naive Merging:** External datasets with divergent disease definitions (e.g., broad "blight" vs. distinct "early blight" and "late blight") are never naively collapsed into identical classes without expert re-annotation.
4. **Local Smoke Sample Manifest:** Located at `data/manifests/sample_test_manifest.json`, providing verifiable verification fixtures.
