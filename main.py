from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel, StoppingCriteria, StoppingCriteriaList
from collections import defaultdict
import fitz
import io
import torch

processor = AutoProcessor.from_pretrained("facebook/nougat-small")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

base_directory = Path.cwd()
data_directory = base_directory / "data"
output_pdfs_directory = Path("/bachelor/nougat_ori/nougat/data/output_pdfs")
output_text_directory = Path("/bachelor/nougat_ori/nougat/data/extracted_text")
output_text_directory.mkdir(parents=True, exist_ok=True)


def rasterize_paper(pdf: Path, dpi: int = 96, return_pil: bool = True, pages: None = None):
    pillow_images = []
    try:
        pdf = fitz.open(pdf)
        if pages is None:
            pages = range(len(pdf))
        for i in pages:
            page_bytes = pdf[i].get_pixmap(dpi=dpi).pil_tobytes(format="PNG")
            if return_pil:
                pillow_images.append(io.BytesIO(page_bytes))
    except Exception as e:
        print(f"Fehler beim Rasterisieren von {pdf}: {e}")
    return pillow_images


class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return None
        return torch.var(self.values, 1) / self.values.shape[1] if self.norm else torch.var(self.values, 1)


class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0


for pdf_path in output_pdfs_directory.glob("*.pdf"):
    print(f"Verarbeite Datei: {pdf_path}")

    if not pdf_path.exists():
        print(f"Fehler: Die Datei {pdf_path} existiert nicht.")
        continue

    images = rasterize_paper(pdf=pdf_path, return_pil=True)

    if not images:
        print(f"Fehler: Keine Bilder aus {pdf_path} extrahiert.")
        continue

    image = Image.open(images[0])
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        min_length=1,
        max_length=3584,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
    )

    generated = processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
    print("\n----- Rohe Ausgabe -----\n")
    print(generated)

    generated = generated.replace("\\", "\\")
    generated = f"$$\n{generated.strip()}\n$$"

    print("\n----- Nachbearbeitete Ausgabe -----\n")
    print(generated)

    output_path = output_text_directory / f"{pdf_path.stem}.md"
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write("# Extrahierter Text aus PDF\n\n")
            file.write("\n")
            file.write(generated)
            file.write("\n")
        print(f"\nText erfolgreich in {output_path} gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern von {output_path}: {e}")
