import xml.etree.ElementTree as ET
from pathlib import Path

class InkmlGroundTruthExtractor:
    def __init__(self, inkml_dir):
        self.inkml_dir = Path(inkml_dir)

    def extract_ground_truth(self, inkml_file):
        try:
            tree = ET.parse(inkml_file)
            root = tree.getroot()

            truth_annotation = root.find(".//{http://www.w3.org/2003/InkML}annotation[@type='truth']")
            if truth_annotation is not None:
                return truth_annotation.text.strip()
            else:
                print(f"Ground Truth fehlt in Datei: {inkml_file}")
                return None
        except Exception as e:
            print(f"Fehler beim Lesen von {inkml_file}: {e}")
            return None
