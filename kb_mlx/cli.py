import typer
from pathlib import Path
from typing import Optional

app = typer.Typer(
    no_args_is_help=True,
    help="KokoroMLX Voice Blender CLI for Mac M1-M4",
    )

@app.command("run", no_args_is_help=True)
def main(
    text: str = typer.Option(
        ...,
        "--text",
        "-t",
        help="Input text as .txt file path or direct string"
    ),
    voice1: str = typer.Option(
        "af_heart",
        "--voice1",
        "-v1",
        help="Name of the first voice (without .pt)"
    ),
    voice2: Optional[str] = typer.Option(
        None,
        "--voice2",
        "-v2",
        help="Name of the second voice (without .pt); if omitted, use only voice1"
    ),
    mix_ratio: float = typer.Option(
        0.5,
        "--mix-ratio",
        "-m",
        help="Blend weight for voice1 and voice2 (0.5 = 50% each)"
    ),
    speed: float = typer.Option(
        1,
        "--speed",
        "-s",
        help="Speed multiplier (1.5 = 50% faster, 0.5 = 50% slower)"
    ),
    file_name: str = typer.Option(
        None,
        "--file-name",
        "-fn",
        help="Configure individual output audio file name (without extension)"
    ),
    model_dir: Path = typer.Option(
        "./models/Kokoro-82M-bf16",
        "--model-dir",
        "-md",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to the local Kokoro model directory"
    ),
    output_dir: str = typer.Option(
        "./output",
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory where output audio file will be saved"
    ),
    verbose: bool = typer.Option(
        True,
        is_flag=True,
        show_default=True,
        help="Enable verbose output"
    )
):
    """
    Run TTS with KokoroMLX for M1-M4. Use one voice or blend two voices.
    """     
    import torch
    import mlx.core as mx
    from mlx_audio.tts.generate import generate_audio
    from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline
    
    def blend_voices(
        current_model_path: Path,
        voice_names: list[str],
        weights: Optional[list[float]] = None
        ) -> str:
        """
        Blend multiple voice style embeddings located in model_path/voices.
        current_model_path: Path object to the model directory.
        voice_names: list of voice identifiers (without .pt extension).
        weights: list of float weights; if None, uses equal weights.
        Returns the name of the blended voice embedding file (without extension).
        """
        voices_dir = current_model_path / "voices"
        if not voices_dir.is_dir():
            typer.secho(
                f"Voices directory not found: {voices_dir}",
                fg=typer.colors.RED
                )
            raise typer.Exit(code=1)

        if weights is None:
            weights = [1.0 / len(voice_names)] * len(voice_names)
        
        if len(voice_names) != len(weights):
            typer.secho(
                "Number of voices and weights must match.",
                fg=typer.colors.RED
                )
            raise typer.Exit(code=1)

        blended_style = None
        for name, weight in zip(voice_names, weights):
            pt_file = voices_dir / f"{name}.pt"
            if not pt_file.is_file():
                typer.secho(
                    f"Voice file not found: {pt_file}",
                    fg=typer.colors.RED
                    )
                raise typer.Exit(code=1)
            try:
                style = torch.load(pt_file, map_location="cpu")
                style = style * weight # Apply weight
                if blended_style is None:
                    blended_style = style
                else:
                    blended_style += style
            except Exception as e:
                typer.secho(
                    f"Error loading or processing voice {name}: {e}",
                    fg=typer.colors.RED
                    )
                raise typer.Exit(code=1)
        
        # Create a name for the blended voice
        blended_name_parts = []
        for name, weight in zip(voice_names, weights):
            blended_name_parts.append(f"{name}{int(weight*100)}") # e.g., voiceA50_voiceB50
        
        blended_name = "_".join(blended_name_parts) + "_blend"
        blended_path = voices_dir / f"{blended_name}.pt"
        
        try:
            torch.save(blended_style, blended_path)
        except Exception as e:
            typer.secho(f"Error saving blended voice {blended_name}.pt: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
            
        return blended_name
    
    # Load text from .txt file or use direct string
    if text.endswith(".txt") and Path(text).is_file():
        text_content = Path(text).read_text()
    else:
        text_content = text

    # Monkey-patch KokoroPipeline to load embeddings locally first
    _original_load_single = KokoroPipeline.load_single_voice
    
    def _load_single_voice_local(self, voice):
        voices_dir_local = Path(model_dir) / "voices"
        print(voices_dir_local)
        local_file = voices_dir_local / f"{voice}.pt"
        if local_file.exists():
            style_torch = torch.load(local_file, map_location="cpu")
            style_arr = mx.array(style_torch.cpu().numpy())
            return style_arr
        return _original_load_single(self, voice)

    KokoroPipeline.load_single_voice = _load_single_voice_local

    # Determine voice or blend
    if voice2:
        w1 = mix_ratio
        w2 = 1.0 - mix_ratio
        blend_name = blend_voices(
            model_dir,
            [voice1, voice2],
            [w1, w2]
        )
        typer.echo(f"Blended voice saved as '{blend_name}.pt'")
    else:
        blend_name = voice1
        typer.echo(f"Using single voice: '{blend_name}'")

    # Determine output filename
    output_name = file_name if file_name else blend_name

    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = output_path / output_name

    # Generate audio
    generate_audio(
        text=text_content,
        model_path=str(model_dir),
        voice=blend_name,
        speed=speed,
        file_prefix=str(prefix),
        audio_format="wav",
        sample_rate=24000,
        join_audio=True,
        verbose=verbose
    )

@app.command("list", no_args_is_help=False)
def list_voices(
    model_dir: Path = typer.Option(
        "./models/Kokoro-82M-bf16",
        "--model-dir",
        "-md",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to the local Kokoro model directory"
    ),
):
    """
    List all available voices.
    """
    voices_path = Path(model_dir) / "voices"
    if not voices_path.exists() or not voices_path.is_dir():
        typer.secho(
            f"Voices directory not found: {voices_path.resolve()}. Please \
ensure the model_dir exists.",
            fg=typer.colors.RED
            )
        raise typer.Exit(code=1)
    files = sorted(voices_path.glob("*.pt"))
    if not files:
        typer.secho(
            "No voice embeddings (.pt files) found.",
            fg=typer.colors.RED
            )
        return
    typer.echo("Available voices:")
    for f in files:
        typer.echo(f" - {f.stem}")

if __name__ == "__main__":
    app()
