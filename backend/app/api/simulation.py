from fastapi import APIRouter
from simulation.create_field_image import create_simulated_field

router = APIRouter()

@router.post("/simulation/create")
def trigger_simulation():
    field_path, meta_path = create_simulated_field()
    return {
        "status": "success",
        "message": "Simulated 4x4 Field Composite Image generated.",
        "field_image_path": field_path,
        "metadata_path": meta_path,
        "disclaimer": "Drone view is simulated using a multi-tile composite image because real drone imagery is not available for this prototype."
    }
