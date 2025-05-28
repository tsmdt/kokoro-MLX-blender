# Voice Blender CLI for Kokoro MLX

Run Text-To-Speech with the [MLX implementation](https://huggingface.co/models?search=mlx%20kokoro) (Mac M1-M4) of [Kokoro](https://github.com/hexgrad/kokoro). Use one voice or blend two voices by specifying a mixing ratio.

## Prerequisites

* Python >= 3.10
* HuggingFace Access Token

## Installation

### Clone this repo
```shell
git clone https://github.com/tsmdt/kokoro-MLX-blender.git
```

### Change to project folder
```shell
cd kokoro-MLX-blender
```

### Create a python env and activate it
```shell
python3 -m venv venv_kokoro
source venv_kokoro/bin/activate
```

### Install kokoro-MLX-blender
```shell
pip install .
```

### Download MLX Kokoro model using huggingface-cli
Run the following command from the main project folder (`./kokoro-MLX-blender/`)

```shell
huggingface-cli download --local-dir models/Kokoro-82M-bf16 mlx-community/Kokoro-82M-bf16
```
Make sure the folder `Kokoro-82M-bf16` ([HuggingFace](https://huggingface.co/mlx-community/Kokoro-82M-bf16/)) with a `voices` subfolder and different `.pt` files (e.g., `af_heart`, `af_alloy` etc.) now exists in the `models` folder of `kokoro-MLX-blender`. Your directory should look like this:

```markdown
kokoro-MLX-blender
├── kb_mlx/
│   ├── __init__.py
│   └── cli.py
├── models/
│   └── Kokoro-82M-bf16/
│       ├── samples/
│       ├── voices/
│       ├── .gitattributes
│       ├── config.json
│       ├── DONATE.md
│       ├── kokoro-v1_0.safetensors
│       ├── README.md
│       ├── SAMPLES.md
│       └── VOICES.md
├── .gitignore
├── LICENSE
├── README.md
...
```

### Check if everything works correctly
Run the following command in CLI to check if everything works.

```shell
kbx list
```

If you see a list of voice names `kokoro-MLX-blender` should work. If not please make sure that you downloaded the `kokoro` model in the previouse step and placed it correctly in your `models` dir.

## Usage
Options for running TTS with `kokoro-MLX-blender`:

```shell
$ kbx run

 Usage: kbx run [OPTIONS]

 Run TTS with KokoroMLX for M1-M4. Use one voice or blend two voices.

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --text        -t                   TEXT       Input text as .txt file path or direct string [default: None] [required]                                  │
│    --voice1      -v1                  TEXT       Name of the first voice (without .pt) [default: af_heart]                                                 │
│    --voice2      -v2                  TEXT       Name of the second voice (without .pt); if omitted, use only voice1 [default: None]                       │
│    --mix-ratio   -m                   FLOAT      Blend weight for voice1 and voice2 (0.5 = 50% each) [default: 0.5]                                        │
│    --speed       -s                   FLOAT      Speed multiplier (1.5 = 50% faster, 0.5 = 50% slower) [default: 1]                                        │
│    --file-name   -fn                  TEXT       Configure individual output audio file name (without extension) [default: None]                           │
│    --model-dir   -md                  DIRECTORY  Path to the local Kokoro model directory [default: ./models/Kokoro-82M-bf16]                              │
│    --output-dir  -o                   TEXT       Directory where output audio file will be saved [default: ./output]                                       │
│    --verbose          --no-verbose               Enable verbose output [default: verbose]                                                                  │
│    --help                                        Show this message and exit.                                                                               │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```






