import argparse

from src.models.load_bert_whisper_multimodal_model import BertWhisperTrainingModule

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", type=str, default="models/bert_whisper_multimodal_model.pt"
)

args = parser.parse_args()

print(f"Loading model from {args.model_path}...")
model = BertWhisperTrainingModule.load_from_checkpoint(args.model_path)

bert = model.text_encoder.to("cpu")
bert.save_pretrained(args.model_path.replace("last.ckpt", "bert"))

whisper = model.audio_encoder.to("cpu")
whisper.save_pretrained(args.model_path.replace("last.ckpt", "whisper"))

print(f"Done!")
