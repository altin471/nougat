from PIL import Image
from transformers import pipeline


class NougatOCR:
    def __init__(self, model_name="facebook/nougat-base"):
        self.pipeline = None
        try:
            print(f"Lade Modell {model_name} für die Pipeline...")
            self.pipeline = pipeline("image-to-text", model=model_name)
            print("Pipeline erfolgreich geladen.")
        except Exception as e:
            print(f"Fehler beim Laden der Pipeline: {e}")

    def batch_inference(self, images):

        if not self.pipeline:
            print("Pipeline ist nicht geladen. Batch-Inferenz wird abgebrochen.")
            return [""] * len(images)

        results = []
        for image_path in images:
            try:
                print(f"Verarbeite Bild: {image_path}")
                # Öffne das Bild mit PIL
                image = Image.open(image_path)
                result = self.pipeline(image)[0].get("generated_text", "")
                results.append(result)
                print(f"OCR-Ergebnis für {image_path}: {result}")
            except Exception as e:
                print(f"Fehler bei der Verarbeitung von {image_path}: {e}")
                results.append("")
        return results
