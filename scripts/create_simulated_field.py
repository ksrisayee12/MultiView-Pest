import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.create_field_image import create_simulated_field

if __name__ == "__main__":
    print("[Simulation] Creating 4x4 composite field image...")
    f_path, m_path = create_simulated_field()
    print(f"[Success] Field composite image saved: {f_path}")
    print(f"[Success] Metadata saved: {m_path}")
