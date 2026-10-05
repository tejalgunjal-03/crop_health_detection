import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os, tempfile
_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["UPLOADS_DIR"] = f"{_tmp}/uploads"
os.environ["RESULTS_DIR"] = f"{_tmp}/results"
os.environ["WEATHER_API_KEY"] = "test-key"
os.environ["LEAF_MODEL_PATH"] = f"{_tmp}/missing_leaf.pth"     # guaranteed missing
os.environ["UAV_MODEL_PATH"] = f"{_tmp}/missing_uav.pth"
