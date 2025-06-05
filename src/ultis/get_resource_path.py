import os
import sys

def _resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class ResourcePath:
    MODEL_PATH = _resource_path('assets/best.pt')
    LOGO_PATH = _resource_path('assets/favicon.png')
    UI_PATH = _resource_path('assets/PlantDiseasePrediction.ui')


