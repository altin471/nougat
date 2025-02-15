class MetricsEvaluatorLatex:
    def __init__(self, ground_truth_file):
        """Lädt die Ground Truth aus einer JSON-Datei"""
        self.ground_truth_data = self.load_ground_truth(ground_truth_file)

    def load_ground_truth(self, ground_truth_file):
        """Lädt die Ground Truth Daten aus der JSON-Datei"""
        import json
        from pathlib import Path

        ground_truth_path = Path(ground_truth_file)
        if ground_truth_path.exists():
            with open(ground_truth_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            print("❌ Fehler: Ground Truth Datei nicht gefunden!")
            return {}

    def calculate_cer(self, predicted, actual):
        """Berechnet die Character Error Rate (CER)"""
        from difflib import SequenceMatcher

        sm = SequenceMatcher(None, predicted, actual)
        distance = sum(op[2] - op[1] for op in sm.get_opcodes() if op[0] != 'equal')
        return distance / max(len(predicted), len(actual)) if max(len(predicted), len(actual)) > 0 else 1.0

    def calculate_wer(self, predicted, actual):
        """Berechnet die Word Error Rate (WER)"""
        from difflib import SequenceMatcher

        predicted_words = predicted.split()
        actual_words = actual.split()
        sm = SequenceMatcher(None, predicted_words, actual_words)
        distance = sum(op[2] - op[1] for op in sm.get_opcodes() if op[0] != 'equal')
        return distance / max(len(predicted_words), len(actual_words)) if max(len(predicted_words), len(actual_words)) > 0 else 1.0

