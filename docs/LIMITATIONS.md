# MultiView-AgriHealth — Scientific, Operational & Model Limitations

**Date:** 2026-10-18  
**Policy:** Strict honesty in agricultural decision support — no overstatement of model capabilities.

---

## 1. Decision Support vs. Guaranteed Diagnosis

1. **Not a Clinical Substitute:** This application provides agricultural **decision support**. It is not a guaranteed diagnosis or substitute for laboratory microbiological isolation, fungal microscopy, ELISA, or PCR confirmation.
2. **Consult Qualified Specialists:** Chemical fungicides, bactericides, or pesticide treatments should never be applied on the sole basis of computer vision outputs without independent agronomist review.
3. **No Fabricated Dosages:** The system deliberately does not output automated chemical dosage recommendations.

---

## 2. Computer Vision & Model Limitations

1. **Supported Crop Domain:** The loaded model (`yolo26_tomato.pt`) is fine-tuned strictly on tomato leaf photography (*Solanum lycopersicum*). Applying it to non-tomato crops (e.g., corn, apple, grapevine) is unsupported.
2. **Supported Conditions:** Restricted to the 9 supported conditions (healthy, early blight, late blight, septoria leaf spot, bacterial spot, yellow leaf curl, mosaic virus, leaf mold, spider mites). Unrecognized diseases or nutrient deficiencies will result in `no_supported_condition_detected`.
3. **Absence of Evidence is Not Health:** If no bounding boxes are detected, the system strictly outputs `no_supported_condition_detected`. It never claims a leaf is healthy merely because no symptom was found.
4. **Domain Shift Risk:** Background lighting variations (direct sunlight vs. cloudy canopy shade) and soil texture can affect detection sensitivity.
5. **Heuristic Visible Lesion Estimation:** The visible symptom percentage metric is calculated via HSV foliage thresholding. It is clearly labeled as an algorithmic visible indicator, NOT a biological whole-plant pathology score.

---

## 3. Viewpoint Coverage & Multi-View Tracking

1. **Unobserved Angles Remain Unknown:** If only 2 of 6 standard views are submitted (`front` and `back`), the unobserved angles (`top`, `underside`, `left`, `right`) are strictly marked `UNKNOWN` and never assumed healthy.
2. **Cross-View Identity Limitations:** The system associates observations by viewpoint consensus, but does not claim physical 3D leaf re-identification across opposite camera angles without active stereo triangulation.
3. **Temporal Tracking Within Video:** Consecutive video frames are tracked via 2D centroid/IoU matching with track expiry to suppress duplicate detections. It operates strictly within individual video clips.

---

## 4. Deployment & Infrastructure Scope

1. **Single-User / Local Operation:** In its default configuration, the system operates as a single-user local deployment. Multi-tenant access controls, RBAC, and client authentication tokens should be handled via an enterprise API gateway or reverse proxy.
2. **In-Memory Fallback:** When MongoDB is not configured or offline, session records are stored in memory for testing and local exploration. Persistent storage requires setting `MONGODB_URI`.
