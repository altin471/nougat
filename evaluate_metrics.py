from difflib import SequenceMatcher
import csv
from inkml_groundtruth_extractor import InkmlGroundTruthExtractor


class MetricsEvaluator:
    def __init__(self, inkml_dir):
        self.inkml_extractor = InkmlGroundTruthExtractor(inkml_dir)

    def calculate_cer(self, predicted, actual):
        sm = SequenceMatcher(None, predicted, actual)
        distance = sum(op[2] - op[1] for op in sm.get_opcodes() if op[0] != 'equal')
        return distance / max(len(predicted), len(actual))

    def calculate_wer(self, predicted, actual):
        predicted_words = predicted.split()
        actual_words = actual.split()
        sm = SequenceMatcher(None, predicted_words, actual_words)
        distance = sum(op[2] - op[1] for op in sm.get_opcodes() if op[0] != 'equal')
        return distance / max(len(predicted_words), len(actual_words))

    def evaluate(self, ocr_results, output_file):
        metrics = []

        for result in ocr_results:
            inkml_file = self.inkml_extractor.inkml_dir / result["file"].replace(".png", ".inkml")
            if inkml_file.exists():
                ground_truth = self.inkml_extractor.extract_ground_truth(inkml_file)
                if ground_truth is not None:
                    cer = self.calculate_cer(result["result"], ground_truth)
                    wer = self.calculate_wer(result["result"], ground_truth)
                    metrics.append({"file": result["file"], "cer": cer, "wer": wer})
                else:
                    print(f"Keine Ground Truth in Datei: {inkml_file}")
                    metrics.append({"file": result["file"], "cer": 1.0, "wer": 1.0})
            else:
                print(f"inkml-Datei fehlt für: {result['file']}")
                metrics.append({"file": result["file"], "cer": 1.0, "wer": 1.0})  # Standardwerte

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["File", "CER", "WER"])
            for metric in metrics:
                writer.writerow([metric["file"], metric["cer"], metric["wer"]])

        return metrics
