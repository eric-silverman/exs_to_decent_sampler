# exs_to_decent_sampler

Convert Logic EXS24 (.exs) instruments into DecentSampler bundles (.dsbundle with .dspreset and Samples).

This script parses EXS instruments (XML or binary plist), copies referenced samples, and generates a DecentSampler preset that preserves mappings such as key/velocity ranges, root note, tuning, volume, pan, and looping (including crossfades when available). It also includes sensible UI/effects options for DecentSampler.

## Features

- Parses EXS24 instruments and extracts zones: keys, velocities, root note, tuning, volume, pan, loops, loop crossfade.
- Copies samples and builds a DecentSampler bundle structure:
  - `<CWD>/Decent Sampler Bundes/<InstrumentName>.dsbundle/` (by default)
    - `<InstrumentName>.dspreset`
    - `Samples/…`
- Generates a `.dspreset` targeting DecentSampler (minVersion 1.6.0) with a default envelope and optional UI.
- Robust sample path resolution: absolute, relative, sibling directories, or name-based search within the input folder.
- Fallbacks when EXS parsing is incomplete:
  - Try an `.sfz` in the same folder.
  - Infer mapping from filenames like `Instrument-C3-V95-…` and AIFF loop markers.
- Optional default loop crossfade if missing in source data.

## Requirements

- Python 3.8+
- A folder containing a `.exs` instrument and its samples
- Optional: Pillow (Python Imaging Library) to generate a background image for the UI. If Pillow is not installed, the conversion still works; the script simply skips background generation.

### Install Pillow (optional)

If you want the generated DecentSampler bundle to include a simple gradient background image, install Pillow:

- macOS/Linux (user install): `python3 -m pip install --user pillow`
- macOS/Linux (virtualenv): `pip install pillow`
- Windows (user install): `py -m pip install --user pillow`

You can verify installation in a Python shell with: `import PIL; import PIL.Image`

## Usage

Basic conversion (writes output under `<CWD>/Decent Sampler Bundes`):

```
python3 exs_to_decent_sampler.py /path/to/EXS_folder
```

Control output location (results go directly under `<out>`):

```
python3 exs_to_decent_sampler.py /path/to/EXS_folder --out /path/to/output
```

Specify a particular `.exs` file (when the folder has multiple or none at top level):

```
python3 exs_to_decent_sampler.py /path/to/EXS_folder --exs /path/to/instrument.exs
```

Batch convert all `.exs` files in a folder:

```
python3 exs_to_decent_sampler.py /path/to/EXS_folder --batch
```

Ensure a minimal or full custom UI is included in the preset:

```
# Minimal UI canvas (helps some hosts display controls)
python3 exs_to_decent_sampler.py /path/to/EXS_folder --force-ui

# Full UI with ADSR, Gain, and Tone controls bound to effects
python3 exs_to_decent_sampler.py /path/to/EXS_folder --full-ui
```

Provide a recursive samples search root if references aren’t found locally:

```
python3 exs_to_decent_sampler.py /path/to/EXS_folder --samples-root /path/to/all/samples
```

Apply a default loop crossfade where zones have loop points but no xfade:

```
# Crossfade in samples (frames)
python3 exs_to_decent_sampler.py /path/to/EXS_folder --xfade-samples 1000

# Crossfade in milliseconds (converted per-sample using its sample rate)
python3 exs_to_decent_sampler.py /path/to/EXS_folder --xfade-ms 20
```

### Command line options

- `input_folder` (positional): Folder containing the `.exs` file and samples.
- `--check-deps`: Check optional dependencies (e.g., Pillow) and exit.
- `--out <path>`: Output folder root. Default is `<CWD>/Decent Sampler Bundes`. Results are placed directly under `<out>`.
- `--exs <path>`: Specific `.exs` file to convert (optional).
- `--batch`: Convert all `.exs` files in the input folder (non-recursive). Ignores `--exs` if provided.
- `--force-ui`: Add a minimal `<ui>` block explicitly.
- `--full-ui`: Add a full UI with knobs bound to ADSR, Gain, and Tone.
- `--samples-root <path>`: Optional root to search recursively for samples when direct resolution fails.
- `--xfade-samples <int>`: Default loop crossfade (in samples) for zones missing one.
- `--xfade-ms <float>`: Default loop crossfade (in milliseconds) for zones missing one.

## Output

The script creates a DecentSampler bundle:

```
<out or CWD>/Decent Sampler Bundes/<InstrumentName>.dsbundle/
  ├─ <InstrumentName>.dspreset
  └─ Samples/
      └─ <copied sample files>
```

If duplicate sample basenames occur, files are de-duplicated and suffixed (`_2`, `_3`, …) to avoid clobbering.

## How mapping is derived

- EXS24: Reads keys/velocities/root/tune/volume/pan and loop info when present.
- SFZ fallback: Parses common `<region>` opcodes (sample, pitch_keycenter, lokey/hikey, lovel/hivel, tune, volume, pan, loop_mode, loop_start/end/crossfade, offset/end) and resolves samples relative to `default_path`, the SFZ’s folder, or by basename.
- Filename/AIFF fallback: If EXS and SFZ both fail, the script infers zones from filenames like `…-C3-V95-…` (Logic-style note numbering with C3=60) and uses AIFF `INST/MARK` chunks for loop points when available.

## Notes & limitations

- DecentSampler version: The preset targets DS `minVersion=1.6.0`.
- UI: Without `--force-ui` or `--full-ui`, DecentSampler’s standard UI is used (no custom artwork).
- Crossfades: If loop crossfades aren’t obtainable (e.g., from filename/AIFF fallback), use `--xfade-samples` or `--xfade-ms` to apply a default.
- Sample resolution: Absolute/relative paths are tried; if unresolved, the script searches by basename within the input folder (and `--samples-root` if provided). Unresolved zones are skipped with warnings.

## Troubleshooting

- “No .exs file found in folder”: Provide `--exs /path/to/file.exs` or point `input_folder` at a directory containing the `.exs`.
- “Zone N: could not resolve sample path”: Ensure samples are present; try `--samples-root` to search more broadly.
- “No zones resolved; nothing to convert.”: The source format may be unusual or filenames don’t follow the expected pattern. Try placing an `.sfz` alongside the `.exs`, or use filename conventions like `Instrument-C3-V95-…`.
- Loops not crossfading as expected: Use `--xfade-samples` or `--xfade-ms`.
- “Pillow not installed; skipping gradient background generation.”: This is harmless. If you want a background image in the UI, install Pillow using one of the commands above and rerun.

## Example

```
python3 exs_to_decent_sampler.py "~/Logic/EXS/GrandPiano" \
  --out "~/Instruments" \
  --full-ui \
  --xfade-ms 15

# Default output (no --out):
#   <CWD>/Decent Sampler Bundes/GrandPiano.dsbundle/
# With --out "~/Instruments":
#   ~/Instruments/GrandPiano.dsbundle/
#         ├─ GrandPiano.dspreset
#         └─ Samples/…
```

### Quick dependency check

To verify optional packages are available (currently checks Pillow):

```
python3 exs_to_decent_sampler.py --check-deps
```

## License

No license specified. If you plan to distribute converted libraries, ensure you have permission to redistribute the original samples.
