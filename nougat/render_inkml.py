import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class InkmlProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Eingabeverzeichnis: {self.input_dir}")
        print(f"Ausgabeverzeichnis: {self.output_dir}")

    def render_to_image(self, inkml_path, output_path):

        try:
            tree = ET.parse(inkml_path)
            root = tree.getroot()

            # Extrahiere die Traces aus der INKML-Datei
            strokes = []
            for trace in root.findall(".//{http://www.w3.org/2003/InkML}trace"):
                if trace.text is None:
                    print(f"Warnung: Leerer Trace in Datei {inkml_path}")
                    continue
                points = [list(map(float, coord.split())) for coord in trace.text.strip().split(", ")]
                strokes.append(points)

            if not strokes:
                print(f"Warnung: Keine gültigen Traces in Datei {inkml_path}")
                return


            plt.figure(figsize=(5, 5))
            for stroke in strokes:
                stroke = np.array(stroke)
                plt.plot(stroke[:, 0], -stroke[:, 1], 'k-', linewidth=2)
            plt.axis('off')


            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Bild erfolgreich gespeichert: {output_path}")
        except ET.ParseError as pe:
            print(f"XML-Parsing-Fehler in Datei {inkml_path}: {pe}")
            raise
        except Exception as e:
            print(f"Fehler beim Rendern von {inkml_path}: {e}")
            raise

    def process_all(self):

        print("Beginne die Verarbeitung aller INKML-Dateien.")
        if not self.input_dir.exists():
            print(f"Fehler: Eingabeverzeichnis existiert nicht: {self.input_dir}")
            return

        for inkml_file in self.input_dir.glob("*.inkml"):
            output_image_path = self.output_dir / (inkml_file.stem + ".png")
            print(f"Verarbeite INKML-Datei: {inkml_file}")
            try:
                self.render_to_image(inkml_file, output_image_path)
            except Exception as e:
                print(f"Fehler bei der Verarbeitung von {inkml_file}: {e}")
