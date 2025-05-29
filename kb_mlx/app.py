import gradio as gr
from pathlib import Path
from kb_mlx.cli import TTS_Handler

# Detect Kokoro model directory
models_root = Path("./models")
kokoro_dirs = []
for d in models_root.iterdir():
    if d.name.startswith("Kokoro") and d.is_dir():
        kokoro_dirs.append(d)
    elif d.name.startswith("kokoro") and d.is_dir():
        kokoro_dirs.append(d)
if not kokoro_dirs:
    raise FileNotFoundError("No Kokoro model directory found under ./models")
model_dir = kokoro_dirs[0]

# Load available voice embeddings from the model directory
voices_dir = model_dir / "voices"
if not voices_dir.is_dir():
    raise FileNotFoundError(f"Voices directory not found: {voices_dir}")
voice_choices = [p.stem for p in sorted(voices_dir.glob("*.pt"))]

# Custom CSS to center Markdown text
css = """
.center-text {
    text-align: center !important;
    display: block;
}
.h1 h1 {
    font-size: 4rem !important;
    line-height: 1.2 !important;
}
"""

def tts_from_txt(files, voice1, voice2, mix_ratio, speed):
    """
    Run TTS on each uploaded .txt file and return paths to the generated .wav files.
    """
    output_paths = []
    for uploaded in files:
        handler = TTS_Handler(
            text=uploaded.name,
            voice1=voice1,
            voice2=voice2 or None,
            mix_ratio=mix_ratio,
            speed=speed,
            model_dir=model_dir,
        )
        handler.run_tts()
        stem = Path(uploaded.name).stem
        blended = handler.blended_voice
        wav_path = handler.output_dir / f"{stem}_{blended}.wav"
        output_paths.append(str(wav_path))
    return output_paths

# Theme
theme = gr.themes.Citrus(
    primary_hue="emerald",
    neutral_hue="slate",
    spacing_size=gr.themes.sizes.spacing_sm,
    text_size="md",
    radius_size="sm",
    font=[
        gr.themes.GoogleFont('Open Sans', 'Roboto'), 
        'ui-sans-serif', 
        'system-ui', 
        'sans-serif'
    ],
    font_mono=['Roboto Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

# Build Gradio Blocks interface
with gr.Blocks(theme=theme, css=css) as app:
    title = "🦜 KokoroMLX Voice Blender"
    gr.HTML(
        f"<h1 style='font-size:2.2rem; text-align:center; margin-bottom:0.5rem;'>{title}</h1>",
        elem_classes="center-text"
    )
    gr.Markdown(
        "Upload one or more .txt files to generate speech with one or \
two blended voices. Make sure that your **Kokoro models folder** exists at \
`./models/Kokoro...` with a `voices` subfolder. If your models folder was \
loaded correctly all voices are available in the dropdown menus of `Voice 1` \
and `Voice 2` below.",
        elem_classes="center-text"
    )

    with gr.Row():
        with gr.Column():
            # Inputs
            file_input = gr.Files(
                file_count="multiple",
                file_types=[".txt"],
                label="Upload Text Files"
            )
            voice1_dropdown = gr.Dropdown(
                choices=voice_choices,
                value=voice_choices[0] if voice_choices else None,
                label="Voice 1"
            )
            voice2_dropdown = gr.Dropdown(
                choices=["", *voice_choices],
                value="",
                label="Voice 2 (optional)"
            )
            mix_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.01,
                label="Voice Mix Ratio"
            )
            speed_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Voice Speed"
            )
            # Generate Button
            generate_button = gr.Button("Generate")
        with gr.Column():
            # Outputs
            output_files = gr.Files(
                file_count="multiple",
                label="Generated audio files"
            )
            generate_button.click(
                fn=tts_from_txt,
                inputs=[file_input, voice1_dropdown, voice2_dropdown, mix_slider, speed_slider],
                outputs=[output_files]
            )

if __name__ == "__main__":
    app.launch()
    