import typer
import re
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
        help="Input text(s) as string, single .txt or directory path"
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
        help="Name of second voice (without .pt); if omitted, use only voice1"
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
    handler = TTS_Handler(
        text=text,
        voice1=voice1,
        voice2=voice2,
        mix_ratio=mix_ratio,
        speed=speed,
        model_dir=model_dir,
        output_dir=output_dir,
        verbose=verbose
    )
    
    handler.run_tts()

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


class TTS_Handler:
    """
    Class for handling TTS tasks.
    """
    def __init__(
        self,
        text: str,
        voice1: str = "af_heart",
        voice2: Optional[str] = None,
        mix_ratio: float = 0.5,
        speed: float = 1.0,
        model_dir: Path = Path("./models/Kokoro-82M-bf16"),
        output_dir: str = "./output",
        verbose: bool = True,
    ) -> None:
        self.text = text
        self.voice1 = voice1
        self.voice2 = voice2
        self.weight1 = None
        self.weight2 = None
        self.blended_voice = None
        self.mix_ratio = mix_ratio
        self.speed = speed
        self.inputs: list[(str, str)] = None
        self.model_dir = model_dir
        self.output_dir: Path = self._validate_output_dir(output_dir)
        self.verbose = verbose

    def _validate_output_dir(self, output_dir):
        """
        Check if output_dir exists. If not create it.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        return self.output_dir
    
    def blend_voices(
        self,
        voice_names: list[str],
        weights: Optional[list[float]] = None
        ) -> None:
        """
        Blend multiple voice style embeddings located in model_path/voices.
        """
        import torch
        
        voices_dir = self.model_dir / "voices"
        
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
        
        blended_name_parts = []
        for name, weight in zip(voice_names, weights):
            blended_name_parts.append(f"{name}{int(weight*100)}")
        
        blended_name = "_".join(blended_name_parts) + "_blend"
        blended_path = voices_dir / f"{blended_name}.pt"
        
        # Save blended voice
        if not blended_path.exists():
            try:
                torch.save(blended_style, blended_path)
            except Exception as e:
                typer.secho(
                    f"Error saving blended voice {blended_name}.pt: {e}",
                    fg=typer.colors.RED
                )
                raise typer.Exit(code=1)
            
        return blended_name

    def collect_inputs(self) -> None:
        """
        Collect all input strings for TTS generation.
        """
        text_path = Path(self.text)

        try:
            # Single file
            if text_path.is_file() and text_path.suffix == ".txt":
                output_name = f"{text_path.stem}_{self.blended_voice}"
                self.inputs = [(text_path.read_text(), output_name)]

            # Folder
            elif text_path.is_dir():
                txt_files = sorted(text_path.glob("*.txt"))
                if not txt_files:
                    typer.secho(
                        f"No .txt files found in directory: {text_path}",
                        fg=typer.colors.RED
                    )
                    raise typer.Exit(code=1)
                
                self.inputs = (
                    [(f.read_text(), f"{f.stem}_{self.blended_voice}") for f in txt_files]
                )

            # String
            else:
                if not self.text.strip():
                    typer.secho(
                        "The input text string is empty.",
                        fg=typer.colors.RED
                    )
                    raise typer.Exit(code=1)

                # Construct output name
                clean_name = re.sub(r'[\s\W]+', '_', self.text[:20].lower().strip())
                temp_name = f"{clean_name}_{self.blended_voice}"
                output_name = temp_name
                self.inputs = [(self.text, output_name)]
        except FileNotFoundError:
            typer.secho(f"File not found: {self.text}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        except NotADirectoryError:
            typer.secho(
                f"Directory not found: {self.text}",
                fg=typer.colors.RED
                )
            raise typer.Exit(code=1)
        except ValueError as e:
            typer.secho(f"Invalid input: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    
    def run_tts(self):
        """
        Run TTS
        """
        import torch
        import mlx.core as mx
        from mlx_audio.tts.generate import generate_audio
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline
        
        # Monkey-patch KokoroPipeline to load embeddings locally first
        _original = KokoroPipeline.load_single_voice
        pipeline_model_dir = self.model_dir

        def _load_single_voice_local(pipeline_self, voice):
            voice_file = Path(pipeline_model_dir) / "voices" / f"{voice}.pt"
            if voice_file.exists():
                data = torch.load(voice_file, map_location="cpu")
                return mx.array(data.cpu().numpy())
            return _original(pipeline_self, voice)

        KokoroPipeline.load_single_voice = _load_single_voice_local

        # Blend voices if voice2 was provided
        if self.voice2:
            self.weight1 = self.mix_ratio
            self.weight2 = 1.0 - self.mix_ratio
            self.blended_voice = self.blend_voices(
                [self.voice1, self.voice2],
                [self.weight1, self.weight2]
            )
            typer.secho(
                f"Using blended voice: '{self.blended_voice}'",
                fg=typer.colors.GREEN
            )
        else:
            self.blended_voice = self.voice1
            typer.secho(
                f"Using single voice: '{self.blended_voice}'",
                fg=typer.colors.GREEN
            )

        # Collect inputs
        self.collect_inputs()

        # Run TTS
        for content, stem in self.inputs:
            file_prefix = self.output_dir / stem
            generate_audio(
                text=content,
                model_path=str(self.model_dir),
                voice=self.blended_voice,
                speed=self.speed,
                file_prefix=str(file_prefix),
                audio_format="wav",
                sample_rate=24000,
                join_audio=True,
                verbose=self.verbose
            )


if __name__ == "__main__":
    app()
