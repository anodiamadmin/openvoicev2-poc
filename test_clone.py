import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 1) Paths
ckpt_converter = "checkpoints_v2/converter"
output_dir = "outputs_v2"
os.makedirs(output_dir, exist_ok=True)

reference_speaker = "resources/example_reference.mp3"  # later this will be user's .wav
text = "This is a test of OpenVoice V2 running on my Vast.ai GPU server."

# 2) Init converter
tone_color_converter = ToneColorConverter(
    f"{ckpt_converter}/config.json",
    device=device
)
tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

# 3) Extract target tone color embedding from reference audio
tgt_se, audio_name = se_extractor.get_se(
    reference_speaker,
    tone_color_converter,
    vad=True  # voice activity detection
)

# 4) Use MeloTTS as base speaker (choose English variant)
language = "EN_NEWEST"  # can try EN, EN_US, EN_AU, etc.
model = TTS(language=language, device=device)
speaker_ids = model.hps.data.spk2id

# Choose one base speaker (e.g. the first)
first_speaker_key = list(speaker_ids.keys())[0]
speaker_id = speaker_ids[first_speaker_key]

tmp_path = f"{output_dir}/tmp_base.wav"
speed = 1.0

# 5) Generate base TTS audio
model.tts_to_file(
    text=text,
    speaker_id=speaker_id,
    output_path=tmp_path,
    speed=speed
)

# 6) Load source speaker embedding from checkpoints_v2
#    Map 'EN_NEWEST' to lower-case file key if needed
speaker_key_file = "en-newest"  # must match filenames like en-newest.pth
src_se = torch.load(
    f"checkpoints_v2/base_speakers/ses/{speaker_key_file}.pth",
    map_location=device
)

# 7) Run tone color converter to get cloned voice
save_path = f"{output_dir}/test_clone_output.wav"
encode_message = "@MyShell"

tone_color_converter.convert(
    audio_src_path=tmp_path,
    src_se=src_se,
    tgt_se=tgt_se,
    output_path=save_path,
    message=encode_message
)

print("Done. Output saved to:", save_path)
