#!/usr/bin/env python3
"""
EXS24 (.exs) -> DecentSampler (.dspreset + Samples) converter

What it does
- Parses an EXS24 instrument (.exs; XML or binary plist) and extracts zones
  (key range, velocity range, root note, tuning, volume, pan, looping, crossfade).
- Copies referenced sample files into a DecentSampler bundle folder structure:
  <Output>/<InstrumentName>.dsbundle/
    ├─ <InstrumentName>.dspreset
    └─ Samples/
- Generates a DecentSampler preset that preserves mappings (keys, velocities,
  looping, loop crossfades where present). The preset includes DecentSampler’s
  standard controls via the default UI (no custom UI artwork required).

Notes
- EXS24 keys vary by version; this script tries common names and fails gracefully
  with warnings for anything it can’t map. It’s conservative: it won’t guess if
  a value is missing; it will just fall back to DS defaults.
- EXS24 sample references can be absolute or relative. This script tries to
  resolve paths by: direct path -> sibling folder lookup -> name-based search
  within the provided input folder. If a sample can’t be resolved, it will be
  skipped with a warning.

Usage
  python3 exs_to_decent_sampler.py /path/to/EXS_folder [--out /path/to/output]

Output
  By default (no --out), bundles are written under:
  <CWD>/Decent Sampler Bundes/<instrument_name>.dsbundle/
    - <instrument_name>.dspreset
    - Samples/<copied sample files>
    - Resources/bg.png (generated gradient background)

Limitations
- UI: DecentSampler shows a full standard UI if no custom <ui> is provided.
  This script provides a minimal custom UI with a gradient background and a
  large, centered title label. If you want additional controls or styling, that
  can be added in a follow-up.

- Loop crossfades: EXS loop crossfade values are not available when EXS parsing
  fails and we fall back to filename/AIFF parsing. In those cases, use
  --xfade-samples or --xfade-ms to apply a default crossfade to all loops.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import plistlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -------------------------
# Helpers and data classes
# -------------------------

MIDI_MIN = 0
MIDI_MAX = 127


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def generate_background(bundle_dir: Path, title: str, theme: str, width: int, height: int) -> Optional[str]:
    """Generate a vertical gradient background PNG with a large top-centered title.

    - theme: 'light' or 'dark' to choose palette.
    Returns a relative path (inside the bundle) if successful, else None.
    Requires Pillow. If unavailable, returns None.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
    except Exception:
        warn("Pillow not installed; skipping gradient background generation. Install with: 'python3 -m pip install --user pillow' (or 'pip install pillow').")
        return None

    res_dir = bundle_dir / 'Resources'
    res_dir.mkdir(parents=True, exist_ok=True)
    bg_path = res_dir / 'bg.png'

    # Gradient background
    img = Image.new('RGB', (width, height), '#3b3f44')
    if (theme or '').lower() == 'light':
        top = (235, 238, 242)
        bottom = (206, 211, 219)
    else:
        top = (55, 58, 62)
        bottom = (30, 32, 35)
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / max(1, height - 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Render large title near the top center
    text = title
    # Heuristic font size relative to canvas width
    # Title size reduced by ~50% from previous
    size = max(36, min(90, width // 20))
    try:
        from PIL import ImageFont
        # Try a common font first; fallback to default if unavailable
        try:
            font = ImageFont.truetype("Arial.ttf", size)
        except Exception:
            font = ImageFont.load_default()
    except Exception:
        font = None  # type: ignore
    if font is not None:
        # Measure text size
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                tw, th = draw.textsize(text, font=font)
            except Exception:
                tw, th = (len(text) * max(8, size // 2), size)
        tx = (width - tw) // 2
        ty = max(10, 20)
        shadow = (0, 0, 0, 140)
        fg = (20, 23, 27) if (theme or '').lower() == 'light' else (236, 240, 245)
        # Soft shadow for contrast
        draw.text((tx+2, ty+2), text, font=font, fill=shadow)
        draw.text((tx, ty), text, font=font, fill=fg)

    # Draw knob labels into the background with theme-aware color
    names = ['Gain', 'Attack', 'Decay', 'Sustain', 'Release', 'Tone']
    label_color = (32, 36, 40) if (theme or '').lower() == 'light' else (220, 225, 232)
    # Layout mirrors ds_add_full_ui
    n = 6
    k_w = 160
    margin = 50
    if width <= (n * k_w + 2 * margin):
        spacing = 10
        x0 = margin
    else:
        spacing = (width - 2 * margin - n * k_w) // (n - 1)
        x0 = margin
    # Place labels even closer to the knobs
    # Keep labels aligned with knobs: knob top y is set in ds_add_full_ui
    knob_top_y = 100
    y_label = knob_top_y - 4
    try:
        lbl_font = ImageFont.truetype("Arial.ttf", 22)
    except Exception:
        lbl_font = ImageFont.load_default()
    for i, name in enumerate(names):
        x = x0 + i * (k_w + spacing)
        center_x = x + k_w // 2
        try:
            bbox = draw.textbbox((0, 0), name, font=lbl_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                tw, th = draw.textsize(name, font=lbl_font)
            except Exception:
                tw, th = (len(name) * 8, 18)
        tx = int(center_x - tw / 2)
        # subtle shadow for contrast
        draw.text((tx+1, y_label+1), name, font=lbl_font, fill=(0,0,0))
        draw.text((tx, y_label), name, font=lbl_font, fill=label_color)

    # Save PNG
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGB')
    img.save(bg_path, 'PNG')
    return 'Resources/bg.png'


# External background downloads removed; always generate local gradient.


def guess_instrument_name(exs_path: Path) -> str:
    return exs_path.stem


def safe_int(v, default: int) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def safe_float(v, default: float) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


class Zone:
    def __init__(
        self,
        sample_path: Path,
        root_note: int,
        lo_note: int,
        hi_note: int,
        lo_vel: int,
        hi_vel: int,
        tune_cents: float = 0.0,
        volume_db: float = 0.0,
        pan: float = 0.0,
        start: Optional[int] = None,
        end: Optional[int] = None,
        loop_start: Optional[int] = None,
        loop_end: Optional[int] = None,
        loop_xfade: Optional[int] = None,
        loop_mode: Optional[str] = None,
    ) -> None:
        self.sample_path = sample_path
        self.root_note = root_note
        self.lo_note = lo_note
        self.hi_note = hi_note
        self.lo_vel = lo_vel
        self.hi_vel = hi_vel
        self.tune_cents = tune_cents
        self.volume_db = volume_db
        self.pan = pan
        self.start = start
        self.end = end
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.loop_xfade = loop_xfade
        self.loop_mode = loop_mode


# ---------------------------------
# EXS24 parsing and key extraction
# ---------------------------------

def load_exs(path: Path) -> dict:
    with path.open('rb') as f:
        try:
            return plistlib.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to parse EXS24 plist: {e}")


def find_samples_in_folder(root: Path) -> Dict[str, Path]:
    """Index samples by basename (case-insensitive) within root."""
    exts = {'.wav', '.aif', '.aiff', '.flac', '.mp3'}
    index: Dict[str, Path] = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in exts:
                index[fn.lower()] = Path(dirpath) / fn
    return index


def resolve_sample_path(zone_dict: dict, base_folder: Path, sample_index: Dict[str, Path]) -> Optional[Path]:
    """Resolve sample path referenced by a zone.

    EXS24 typically stores paths in keys like 'sample', 'name', 'file', 'filename',
    or nested inside a 'sample' dict. We try the common patterns, then fallback
    to a name-based index lookup in the provided folder.
    """
    candidates: List[str] = []

    # Common keys that might directly reference file names/paths
    for k in [
        'sample', 'file', 'filename', 'name', 'Sample', 'FileName', 'Path'
    ]:
        v = zone_dict.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())

    # Nested structure: zone['sample'] may be a dict
    s_obj = zone_dict.get('sample') or zone_dict.get('Sample')
    if isinstance(s_obj, dict):
        for k in ['name', 'file', 'filename', 'path', 'Name']:
            v = s_obj.get(k)
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())

    # Try the candidates as-is or relative to base
    for c in candidates:
        p = Path(c)
        if p.is_file():
            return p
        if (base_folder / p).is_file():
            return base_folder / p

    # Try to match by basename via index
    for c in candidates:
        key = Path(c).name.lower()
        if key in sample_index:
            return sample_index[key]

    return None


def get_int_key(d: dict, keys: List[str], default: int) -> int:
    for k in keys:
        if k in d:
            return safe_int(d.get(k), default)
    return default


def get_float_key(d: dict, keys: List[str], default: float) -> float:
    for k in keys:
        if k in d:
            return safe_float(d.get(k), default)
    return default


def parse_zones(exs: dict, base_folder: Path, sample_index: Dict[str, Path]) -> List[Zone]:
    zones_raw = None
    # Common containers for zones in EXS24 formats
    for key in ['Zones', 'zones', 'Samples', 'samples']:
        if key in exs and isinstance(exs[key], list):
            zones_raw = exs[key]
            break
    if zones_raw is None:
        # Sometimes zones live inside 'groups' -> each has 'zones'
        groups = exs.get('Groups') or exs.get('groups')
        if isinstance(groups, list):
            zones_raw = []
            for g in groups:
                z = g.get('zones') or g.get('Zones')
                if isinstance(z, list):
                    zones_raw.extend(z)
    if zones_raw is None:
        raise RuntimeError('Could not find any zones in EXS instrument.')

    parsed: List[Zone] = []
    for i, zr in enumerate(zones_raw):
        if not isinstance(zr, dict):
            continue

        sample_path = resolve_sample_path(zr, base_folder, sample_index)
        if not sample_path:
            warn(f"Zone {i}: could not resolve sample path; skipping zone.")
            continue

        # Root note and key range
        root_note = get_int_key(zr, ['rootKey', 'root', 'rootNote', 'RootNote', 'Pitch'], 60)
        lo_note = get_int_key(zr, ['loKey', 'lowKey', 'minKey', 'KeyRangeLower', 'LoKey'], MIDI_MIN)
        hi_note = get_int_key(zr, ['hiKey', 'highKey', 'maxKey', 'KeyRangeUpper', 'HiKey'], MIDI_MAX)

        # Velocity range
        lo_vel = get_int_key(zr, ['loVel', 'minVel', 'VelocityRangeLower', 'LoVel'], MIDI_MIN)
        hi_vel = get_int_key(zr, ['hiVel', 'maxVel', 'VelocityRangeUpper', 'HiVel'], MIDI_MAX)

        # Tuning in cents
        tune_cents = get_float_key(zr, ['tune', 'fineTune', 'TuneCents', 'Tune', 'FineTune'], 0.0)

        # Volume (dB) and pan (-100..100)
        volume_db = get_float_key(zr, ['volume', 'gain', 'Volume'], 0.0)
        pan = get_float_key(zr, ['pan', 'Pan'], 0.0)

        # Start/end sample points (frames), if present
        start = zr.get('start') or zr.get('SampleStart') or zr.get('Start')
        start = safe_int(start, None) if start is not None else None
        end = zr.get('end') or zr.get('SampleEnd') or zr.get('End')
        end = safe_int(end, None) if end is not None else None

        # Loop points and crossfade
        loop_start = zr.get('loopStart') or zr.get('LoopStart')
        loop_start = safe_int(loop_start, None) if loop_start is not None else None
        loop_end = zr.get('loopEnd') or zr.get('LoopEnd')
        loop_end = safe_int(loop_end, None) if loop_end is not None else None
        loop_xfade = zr.get('loopCrossfade') or zr.get('LoopCrossfade') or zr.get('xfade')
        loop_xfade = safe_int(loop_xfade, None) if loop_xfade is not None else None

        # Loop mode hint
        loop_mode = zr.get('loopMode') or zr.get('LoopMode')
        if isinstance(loop_mode, (int, float)):
            # Rough mapping guess if numeric: 0=off, 1=forward, 2=pingpong
            loop_mode = {
                0: 'no_loop',
                1: 'forward',
                2: 'pingpong',
            }.get(int(loop_mode), 'forward')
        elif isinstance(loop_mode, str):
            loop_mode = loop_mode.lower()

        parsed.append(Zone(
            sample_path=sample_path,
            root_note=root_note,
            lo_note=lo_note,
            hi_note=hi_note,
            lo_vel=lo_vel,
            hi_vel=hi_vel,
            tune_cents=tune_cents,
            volume_db=volume_db,
            pan=pan,
            start=start,
            end=end,
            loop_start=loop_start,
            loop_end=loop_end,
            loop_xfade=loop_xfade,
            loop_mode=loop_mode,
        ))

    return parsed


# ---------------------------------
# SFZ fallback parsing
# ---------------------------------

def parse_sfz(sfz_path: Path, base_folder: Path, sample_index: Dict[str, Path]) -> List[Zone]:
    """Minimal SFZ parser for <region> entries.

    Supports common opcodes: sample, pitch_keycenter, lokey/hikey, lovel/hivel,
    tune, volume, pan, loop_mode, loop_start, loop_end, loop_crossfade, offset, end.
    """
    default_path = ""
    zones: List[Zone] = []

    def strip_comment(line: str) -> str:
        # Remove '//' comments
        if '//' in line:
            return line.split('//', 1)[0]
        return line

    def parse_keyvals(s: str) -> Dict[str, str]:
        # Parse key=value pairs; values may contain spaces. We treat any token containing
        # an '=' as starting a new key; words in-between are part of previous value.
        parts = s.strip().split()
        out: Dict[str, str] = {}
        k: Optional[str] = None
        v_parts: List[str] = []
        for w in parts:
            if '=' in w:
                # flush previous
                if k is not None:
                    out[k] = ' '.join(v_parts)
                k, v = w.split('=', 1)
                v_parts = [v]
            else:
                if k is not None:
                    v_parts.append(w)
        if k is not None:
            out[k] = ' '.join(v_parts)
        return out

    with sfz_path.open('r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = strip_comment(raw).strip()
            if not line:
                continue
            # capture default_path from control section
            if '<control>' in line:
                # continue; default_path may appear on same or following lines
                # handled generically by parse_keyvals below
                pass
            if 'default_path=' in line:
                kv = parse_keyvals(line)
                dp = kv.get('default_path')
                if dp:
                    default_path = dp.strip().lstrip('./')
            if '<region>' in line:
                region_spec = line.split('>', 1)[1] if '>' in line else ''
                kv = parse_keyvals(region_spec)
                sample_val = kv.get('sample')
                if not sample_val:
                    continue

                # Try resolution: absolute/relative -> base -> default_path -> index by basename
                sample_candidate = Path(sample_val)
                resolved: Optional[Path] = None
                if sample_candidate.is_file():
                    resolved = sample_candidate
                else:
                    if default_path:
                        dp_path = base_folder / default_path / sample_candidate
                        if dp_path.is_file():
                            resolved = dp_path
                    if not resolved:
                        if (base_folder / sample_candidate).is_file():
                            resolved = base_folder / sample_candidate
                    if not resolved:
                        key = sample_candidate.name.lower()
                        if key in sample_index:
                            resolved = sample_index[key]
                if not resolved:
                    warn(f"SFZ: could not resolve sample '{sample_val}'")
                    continue

                root_note = safe_int(kv.get('pitch_keycenter'), 60)
                lo_note = safe_int(kv.get('lokey'), MIDI_MIN)
                hi_note = safe_int(kv.get('hikey'), MIDI_MAX)
                lo_vel = safe_int(kv.get('lovel'), MIDI_MIN)
                hi_vel = safe_int(kv.get('hivel'), MIDI_MAX)
                tune_cents = safe_float(kv.get('tune'), 0.0)
                volume_db = safe_float(kv.get('volume'), 0.0)
                pan = safe_float(kv.get('pan'), 0.0)
                start = safe_int(kv.get('offset'), None) if kv.get('offset') is not None else None
                end = safe_int(kv.get('end'), None) if kv.get('end') is not None else None

                loop_mode_raw = kv.get('loop_mode', '')
                loop_mode = None
                if loop_mode_raw:
                    if loop_mode_raw in ('loop_continuous', 'loop_sustain'):
                        loop_mode = 'forward'
                    elif loop_mode_raw in ('loop_pingpong',):
                        loop_mode = 'pingpong'
                    else:
                        loop_mode = 'forward'

                loop_start = safe_int(kv.get('loop_start'), None) if kv.get('loop_start') is not None else None
                loop_end = safe_int(kv.get('loop_end'), None) if kv.get('loop_end') is not None else None
                loop_xfade = safe_int(kv.get('loop_crossfade'), None) if kv.get('loop_crossfade') is not None else None

                zones.append(Zone(
                    sample_path=resolved,
                    root_note=root_note,
                    lo_note=lo_note,
                    hi_note=hi_note,
                    lo_vel=lo_vel,
                    hi_vel=hi_vel,
                    tune_cents=tune_cents,
                    volume_db=volume_db,
                    pan=pan,
                    start=start,
                    end=end,
                    loop_start=loop_start,
                    loop_end=loop_end,
                    loop_xfade=loop_xfade,
                    loop_mode=loop_mode,
                ))

    return zones


# ---------------------------------
# Filename- and AIFF-driven fallback
# ---------------------------------

NOTE_TO_SEMITONE = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}


def note_to_midi_exs(note_name: str) -> Optional[int]:
    """Convert a note token like C1, D#3 to MIDI assuming Logic's C3=60."""
    note_name = note_name.strip()
    if len(note_name) < 2:
        return None
    # Split letter+accidental and octave
    head = note_name[0].upper()
    idx = 1
    if idx < len(note_name) and note_name[idx] in ['#', 'b']:
        head += note_name[idx]
        idx += 1
    try:
        octave = int(note_name[idx:])
    except ValueError:
        return None
    if head not in NOTE_TO_SEMITONE:
        return None
    semitone = NOTE_TO_SEMITONE[head]
    # If C3 = 60, then midi = (octave - 3) * 12 + 60 + semitone - 0
    midi = (octave - 3) * 12 + 60 + semitone
    if midi < MIDI_MIN or midi > MIDI_MAX:
        return None
    return midi


def parse_sample_name_tokens(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Extract (root_midi, vel_marker) from filenames like ...-C1-V95-XXXX.aif.

    Returns (root_note, velocity_value) or (None, None) if not matched.
    """
    name = path.stem
    # Look for -<Note>-V<vel>
    root_midi: Optional[int] = None
    vel: Optional[int] = None
    # Split by '-' and scan
    parts = name.split('-')
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Note token contains a letter and a digit
        if any(n in p for n in ['C', 'D', 'E', 'F', 'G', 'A', 'B']):
            # Try to match like C1 or D#4 etc.
            nm = p
            m = note_to_midi_exs(nm)
            if m is not None:
                root_midi = m
        if p.startswith('V') and p[1:].isdigit():
            vel = int(p[1:])
    return root_midi, vel


def derive_zones_from_folder(folder: Path) -> List[Zone]:
    exts = {'.aif', '.aiff', '.wav', '.flac'}
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    if not files:
        return []

    # Group by root note; within, by velocity marker
    grouped: Dict[int, List[Tuple[int, Path]]] = {}
    for fp in files:
        rn, vel = parse_sample_name_tokens(fp)
        if rn is None or vel is None:
            # Skip files we can't infer from
            continue
        grouped.setdefault(rn, []).append((vel, fp))

    if not grouped:
        warn('No inferable zones from filenames.')
        return []

    zones: List[Zone] = []
    # Sort root notes ascending
    root_notes = sorted(grouped.keys())
    for idx, rn in enumerate(root_notes):
        prev_rn = root_notes[idx - 1] if idx > 0 else None
        lo_key = rn if prev_rn is None else prev_rn + 1
        hi_key = rn
        # For each velocity tier sort ascending and segment [0..v1], [v1+1..v2], ...
        tiers = sorted(grouped[rn], key=lambda t: t[0])
        prev_hi = -1
        for vel_value, fp in tiers:
            lo_vel = max(0, prev_hi + 1)
            hi_vel = min(127, vel_value)
            prev_hi = hi_vel
            # AIFF loop detection
            loop_start, loop_end = read_aiff_loop_points(fp)
            zones.append(Zone(
                sample_path=fp,
                root_note=rn,
                lo_note=lo_key,
                hi_note=hi_key,
                lo_vel=lo_vel,
                hi_vel=hi_vel,
                loop_start=loop_start,
                loop_end=loop_end,
                loop_mode='forward' if (loop_start is not None and loop_end is not None) else None,
            ))

    return zones


def derive_zones_from_samples_root(samples_root: Path, instr_name: str) -> List[Zone]:
    """Search recursively under samples_root for files belonging to instr_name.

    Heuristic: include files whose basename starts with '<instr_name>-'.
    Mapping is inferred from filename tokens just like derive_zones_from_folder.
    """
    exts = {'.aif', '.aiff', '.wav', '.flac'}
    prefix = f"{instr_name}-".lower()
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(samples_root):
        for fn in filenames:
            if Path(fn).suffix.lower() in exts:
                if fn.lower().startswith(prefix):
                    files.append(Path(dirpath) / fn)
    if not files:
        return []

    # Group by root note; within, by velocity marker
    grouped: Dict[int, List[Tuple[int, Path]]] = {}
    for fp in files:
        rn, vel = parse_sample_name_tokens(fp)
        if rn is None or vel is None:
            continue
        grouped.setdefault(rn, []).append((vel, fp))

    if not grouped:
        warn(f"No inferable zones from filenames for instrument '{instr_name}'.")
        return []

    zones: List[Zone] = []
    root_notes = sorted(grouped.keys())
    for idx, rn in enumerate(root_notes):
        prev_rn = root_notes[idx - 1] if idx > 0 else None
        lo_key = rn if prev_rn is None else prev_rn + 1
        hi_key = rn
        tiers = sorted(grouped[rn], key=lambda t: t[0])
        prev_hi = -1
        for vel_value, fp in tiers:
            lo_vel = max(0, prev_hi + 1)
            hi_vel = min(127, vel_value)
            prev_hi = hi_vel
            loop_start, loop_end = read_aiff_loop_points(fp)
            zones.append(Zone(
                sample_path=fp,
                root_note=rn,
                lo_note=lo_key,
                hi_note=hi_key,
                lo_vel=lo_vel,
                hi_vel=hi_vel,
                loop_start=loop_start,
                loop_end=loop_end,
                loop_mode='forward' if loop_start is not None and loop_end is not None else None,
            ))

    return zones


def get_sample_rate(path: Path) -> int:
    """Return sample rate for AIFF/WAV; fallback to 44100 if unknown."""
    try:
        ext = path.suffix.lower()
        if ext == '.wav':
            import wave
            with wave.open(str(path), 'rb') as w:
                return int(w.getframerate())
        if ext in ('.aif', '.aiff'):
            import aifc
            with aifc.open(str(path), 'rb') as f:
                return int(f.getframerate())
    except Exception:
        pass
    return 44100


def read_aiff_loop_points(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Return (loop_start, loop_end) in frames if present via AIFF INST/MARK chunks."""
    try:
        with path.open('rb') as f:
            data = f.read()
        if len(data) < 12 or data[0:4] != b'FORM' or data[8:12] not in (b'AIFF', b'AIFC'):
            return None, None

        import struct
        pos = 12
        markers: Dict[int, int] = {}
        sustain_loop = None  # (start_marker_id, end_marker_id)
        # Walk chunks
        while pos + 8 <= len(data):
            chunk_id = data[pos:pos+4]
            chunk_size = struct.unpack('>I', data[pos+4:pos+8])[0]
            pos += 8
            chunk_data = data[pos:pos+chunk_size]
            # Chunks are padded to even sizes
            pos += chunk_size + (chunk_size & 1)

            if chunk_id == b'MARK':
                # >H numMarkers
                if len(chunk_data) < 2:
                    continue
                num = struct.unpack('>H', chunk_data[0:2])[0]
                cpos = 2
                for _ in range(num):
                    if cpos + 6 > len(chunk_data):
                        break
                    marker_id, = struct.unpack('>H', chunk_data[cpos:cpos+2])
                    pos48 = struct.unpack('>I', chunk_data[cpos+2:cpos+6])[0]
                    cpos += 6
                    # Skip marker name (pstring)
                    if cpos >= len(chunk_data):
                        break
                    name_len = chunk_data[cpos]
                    cpos += 1 + name_len
                    if (name_len & 1) == 0:
                        cpos += 1  # pad to even
                    markers[marker_id] = pos48
            elif chunk_id == b'INST':
                # Skip 8 bytes of base pitch etc; then sustain loop: 6 bytes
                if len(chunk_data) >= 20:
                    # sustain loop starts at offset 8
                    # struct SustainLoop { UINT16 playMode; UINT16 beginLoop; UINT16 endLoop; }
                    play_mode = struct.unpack('>H', chunk_data[8:10])[0]
                    start_marker = struct.unpack('>H', chunk_data[10:12])[0]
                    end_marker = struct.unpack('>H', chunk_data[12:14])[0]
                    if play_mode != 0:
                        sustain_loop = (start_marker, end_marker)
        if sustain_loop:
            sm, em = sustain_loop
            if sm in markers and em in markers:
                return markers[sm], markers[em]
        return None, None
    except Exception:
        return None, None


# ---------------------------------
# DecentSampler preset generation
# ---------------------------------

def ds_root() -> ET.Element:
    # Use capitalized root name for broad compatibility
    root = ET.Element('DecentSampler')
    root.set('minVersion', '1.6.0')  # safe baseline
    return root


def ds_add_metadata(root: ET.Element, title: str) -> None:
    meta = ET.SubElement(root, 'metadata')
    ET.SubElement(meta, 'name').text = title
    # Let DS render its standard UI if no custom UI provided


def ds_add_basic_ui(root: ET.Element) -> None:
    """Add a minimal UI declaration to encourage hosts to render controls.

    Note: DecentSampler shows a default UI when no <ui> is present. Some hosts
    may hide it; this lightweight <ui> block can help ensure something is shown.
    This does not wire up custom artwork; it simply declares a canvas.
    """
    ui = ET.SubElement(root, 'ui')
    ui.set('width', '700')
    ui.set('height', '400')
    # Intentionally minimal; rely on DecentSampler’s built-in panels.


def ds_add_all_effects(root: ET.Element) -> None:
    """Global Tone filter per docs: <effects><effect type="lowpass" frequency="22000"/></effects>.

    Using a global effect ensures bindings work with level="instrument" and effectIndex=0.
    """
    effs = ET.SubElement(root, 'effects')
    # Effect 0: Global gain (in dB)
    ET.SubElement(effs, 'effect', {
        'type': 'gain',
        'level': '0',
    })
    # Effect 1: Global lowpass filter
    ET.SubElement(effs, 'effect', {
        'type': 'lowpass',
        'frequency': '22000',
        'resonance': '0.7',
    })


def ds_add_full_ui(root: ET.Element, title: str, theme: str = 'dark', bg_rel_path: Optional[str] = None, bg_w: int = 1100, bg_h: int = 420) -> None:
    """Minimal UI: Background gradient + Title label + ADSR + Tone.

    - Attack/Decay/Sustain/Release are bound to group 0's envelope.
    - Tone controls global lowpass cutoff (svf1 cutoff).
    - A large label is pinned to the top center with the instrument name.
    """
    ui = ET.SubElement(root, 'ui')
    ui.set('width', '1100')
    ui.set('height', '420')

    tab_main = ET.SubElement(ui, 'tab', {'name': 'Main'})
    # Optional background image (gradient with embedded title)
    if bg_rel_path:
        ET.SubElement(tab_main, 'image', {
            'x': '0', 'y': '0', 'width': str(bg_w), 'height': str(bg_h),
            'path': bg_rel_path,
        })
    canvas_w = int(ui.get('width'))

    def knob(x: int, y: int, label: str, min_v: str, max_v: str, value: str) -> ET.Element:
        # Use labeled-knob for consistent rendering; hide internal label (drawn in background)
        return ET.SubElement(tab_main, 'labeled-knob', {
            'x': str(x), 'y': str(y), 'width': '160', 'height': '160',
            'label': '', 'type': 'float',
            'minValue': min_v, 'maxValue': max_v, 'value': value,
        })

    # Center six knobs evenly across the canvas
    n = 6
    k_w = 160
    margin = 50
    if canvas_w <= (n * k_w + 2 * margin):
        spacing = 10
        x0 = margin
    else:
        spacing = (canvas_w - 2 * margin - n * k_w) // (n - 1)
        x0 = margin
    # Move knobs slightly down
    y = 100
    x_positions = [x0 + i * (k_w + spacing) for i in range(n)]

    # Gain first as dB knob (-18..+18 dB), centered at 0 dB
    gain_ctl = ET.SubElement(tab_main, 'labeled-knob', {
        'x': str(x_positions[0]), 'y': str(y),
        'width': '160', 'height': '160',
        'label': '', 'type': 'float',
        'minValue': '-18', 'maxValue': '18', 'value': '0'
    })
    # Wider ADSR sweeps for more range
    atk = knob(x_positions[1], y, 'Attack', '0.000', '10.000', '0.005')
    dec = knob(x_positions[2], y, 'Decay',  '0.000', '10.000', '0.200')
    sus = knob(x_positions[3], y, 'Sustain','0.000', '1.000',  '1.000')
    rel = knob(x_positions[4], y, 'Release','0.000', '10.000', '0.500')
    ton = knob(x_positions[5], y, 'Tone',   '0.0',   '1.0',   '1.0')

    # Gain per docs: bind to global gain effect level (0..8)
    # Gain: two-stage mapping for reliable cut/boost across DS versions
    # 1) Cuts: map -18..0 dB (normalized 0..0.5) to ~0.13..1.0 linear gain; stay 1.0 above mid
    # Single-path gain that both cuts and boosts without muting
    # Map 0..1 to 0.25..8.0, with unity near center
    ET.SubElement(gain_ctl,  'binding', {
        'type': 'effect', 'level': 'instrument', 'effectIndex': '0', 'position': '0', 'parameter': 'LEVEL',
        'translation': 'table', 'translationTable': '0.0,0.25;0.25,0.5;0.5,1.0;0.75,2.83;1.0,8.0'
    })

    # ADSR per docs: binding type="amp" level="group" parameter=ENV_ATTACK/...
    ET.SubElement(atk, 'binding', {'type': 'amp', 'level': 'group', 'groupIndex': '0', 'parameter': 'ENV_ATTACK'})
    ET.SubElement(dec, 'binding', {'type': 'amp', 'level': 'group', 'groupIndex': '0', 'parameter': 'ENV_DECAY'})
    ET.SubElement(sus, 'binding', {'type': 'amp', 'level': 'group', 'groupIndex': '0', 'parameter': 'ENV_SUSTAIN'})
    ET.SubElement(rel, 'binding', {'type': 'amp', 'level': 'group', 'groupIndex': '0', 'parameter': 'ENV_RELEASE'})

    # Tone per docs: bind to global lowpass (now effect #1) frequency with gradual response.
    # Make response more gradual using a table (log-like mapping):
    ET.SubElement(ton, 'binding', {
        'type': 'effect', 'level': 'instrument', 'effectIndex': '1', 'position': '1',
        'parameter': 'FX_FILTER_FREQUENCY', 'translation': 'table',
        'translationTable': '0,200;0.25,1000;0.5,3000;0.75,8000;1,20000'
    })


def ds_add_groups(root: ET.Element, zones: List[Zone], samples_subdir: str = 'Samples') -> None:
    groups_el = ET.SubElement(root, 'groups')
    group_el = ET.SubElement(groups_el, 'group')
    group_el.set('id', 'g1')
    # Default ADSR via group attributes per docs
    group_el.set('ampEnvEnabled', 'true')
    group_el.set('attack', '0.005')
    group_el.set('decay', '0.200')
    group_el.set('sustain', '1.000')
    group_el.set('release', '0.500')
    # Set envelope curves for a more natural swell and tail
    group_el.set('attackCurve', '80')   # 0 linear, 100 exponential
    group_el.set('decayCurve', '0')
    group_el.set('releaseCurve', '0')

    for z in zones:
        sample_el = ET.SubElement(group_el, 'sample')
        rel_path = f"{samples_subdir}/{z.sample_path.name}"
        sample_el.set('path', rel_path)

        # Mapping
        sample_el.set('rootNote', str(z.root_note))
        sample_el.set('loNote', str(max(MIDI_MIN, z.lo_note)))
        sample_el.set('hiNote', str(min(MIDI_MAX, z.hi_note)))
        sample_el.set('loVel', str(max(MIDI_MIN, z.lo_vel)))
        sample_el.set('hiVel', str(min(MIDI_MAX, z.hi_vel)))

        # Optional attributes
        if abs(z.tune_cents) > 0.0001:
            sample_el.set('tune', f"{z.tune_cents:.3f}")  # cents
        if abs(z.volume_db) > 0.0001:
            sample_el.set('volume', f"{z.volume_db:.3f}")  # dB
        if abs(z.pan) > 0.0001:
            sample_el.set('pan', f"{z.pan:.3f}")  # -100..100 or -1..1, DS accepts floats

        # Start/end if defined
        if z.start is not None:
            sample_el.set('start', str(z.start))
        if z.end is not None:
            sample_el.set('end', str(z.end))

        # Looping
        if z.loop_start is not None and z.loop_end is not None:
            sample_el.set('loopStart', str(z.loop_start))
            sample_el.set('loopEnd', str(z.loop_end))
            if z.loop_xfade is not None:
                sample_el.set('loopCrossfade', str(z.loop_xfade))
            # Mode: default to forward if present
            if z.loop_mode in {'forward', 'pingpong'}:
                sample_el.set('loopMode', z.loop_mode)
            else:
                sample_el.set('loopMode', 'forward')


def write_preset(root: ET.Element, out_path: Path) -> None:
    tree = ET.ElementTree(root)
    # Ensure pretty printing
    indent_xml(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding='utf-8', xml_declaration=True)


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent_xml(e, level + 1)
        if not e.tail or not e.tail.strip():  # type: ignore[name-defined]
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


# -------------------------
# Copying files and driver
# -------------------------

def copy_samples(zones: List[Zone], out_samples_dir: Path) -> None:
    out_samples_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for z in zones:
        dest = out_samples_dir / z.sample_path.name
        key = dest.name.lower()
        if key in seen:
            # If duplicate name with different source path, rename to avoid clobber
            if not dest.exists() or not file_content_equal(z.sample_path, dest):
                base = dest.stem
                ext = dest.suffix
                n = 2
                while True:
                    candidate = out_samples_dir / f"{base}_{n}{ext}"
                    if not candidate.exists():
                        dest = candidate
                        break
                    n += 1
        shutil.copy2(z.sample_path, dest)
        seen.add(dest.name.lower())


def file_content_equal(a: Path, b: Path) -> bool:
    try:
        if a.stat().st_size != b.stat().st_size:
            return False
        with a.open('rb') as fa, b.open('rb') as fb:
            chunk = 8192
            while True:
                ba = fa.read(chunk)
                bb = fb.read(chunk)
                if ba != bb:
                    return False
                if not ba:
                    return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description='Convert EXS24 instrument to DecentSampler bundle')
    parser.add_argument('--check-deps', action='store_true', help='Check optional dependencies (Pillow) and exit')
    parser.add_argument('input_folder', type=str, nargs='?', help='Folder containing .exs instrument and samples')
    parser.add_argument('--out', type=str, default=None, help='Output folder root (default: <CWD>/Decent Sampler Bundes). Bundles are written directly under this folder.')
    parser.add_argument('--exs', type=str, default=None, help='Specific .exs file to convert (optional)')
    parser.add_argument('--batch', action='store_true', help='Convert all .exs files in the input folder (non-recursive)')
    parser.add_argument('--force-ui', action='store_true', help='Add a minimal <ui> section explicitly')
    parser.add_argument('--full-ui', action='store_true', help='Add a full <ui> with knobs bound to effects and a title label')
    parser.add_argument('--samples-root', type=str, default=None,
                        help='Optional folder to search recursively for samples when EXS/SFZ parsing fails. '
                             'When provided, the script will look for files whose names begin with '
                             '<InstrumentName>- and infer mapping from filenames.')
    parser.add_argument('--xfade-samples', type=int, default=None,
                        help='Default loop crossfade length (in samples) applied to zones that have loopStart/loopEnd but no crossfade.')
    parser.add_argument('--xfade-ms', type=float, default=None,
                        help='Default loop crossfade length (in milliseconds) applied to zones without a crossfade. Converted per-sample using its sample rate (defaults to 44100 Hz if unknown).')
    parser.add_argument('--theme', type=str, choices=['light', 'dark'], default=None,
                        help='UI theme for gradient background and label color. Default: dark.')
    parser.add_argument('--bg-width', type=int, default=1100, help='Background width in pixels (default: 1100).')
    parser.add_argument('--bg-height', type=int, default=420, help='Background height in pixels (default: 420).')
    args = parser.parse_args()

    # Dependency check mode: no input folder required
    if args.check_deps:
        rc = 0
        try:
            import PIL  # type: ignore
            try:
                import PIL.Image  # type: ignore
            except Exception:
                pass
            print("Pillow: OK")
        except Exception:
            print("Pillow: MISSING — install with 'python3 -m pip install --user pillow' or 'pip install pillow'")
            rc = 1
        return rc

    if not args.input_folder:
        print("input_folder is required (or pass --check-deps)", file=sys.stderr)
        return 2

    in_dir = Path(args.input_folder).expanduser().resolve()
    if not in_dir.is_dir():
        print(f"Input folder not found: {in_dir}", file=sys.stderr)
        return 2

    # Determine base output root. Default to current working directory
    # inside a folder named "Decent Sampler Bundes" unless --out is provided.
    base_out = Path(args.out).expanduser().resolve() if args.out else (Path.cwd() / 'Decent Sampler Bundes')

    def convert_one(exs_path: Path) -> int:
        out_root = base_out
        instr_name = guess_instrument_name(exs_path)
        bundle_dir = out_root / f"{instr_name}.dsbundle"
        samples_dir = bundle_dir / 'Samples'
        preset_path = bundle_dir / f"{instr_name}.dspreset"

        info("Indexing samples in input folder...")
        sample_index = find_samples_in_folder(in_dir)
        zones: List[Zone] = []
        try:
            info(f"Parsing EXS: {exs_path}")
            exs = load_exs(exs_path)
            info("Extracting zones from EXS...")
            zones = parse_zones(exs, in_dir, sample_index)
        except Exception as ex:
            warn(f"EXS parse failed ({ex}). Trying SFZ fallback...")
            # Try to find SFZ in same folder
            sfz_path = None
            for p in exs_path.parent.glob('*.sfz'):
                sfz_path = p
                break
            if sfz_path:
                info(f"Parsing SFZ: {sfz_path}")
                zones = parse_sfz(sfz_path, exs_path.parent, sample_index)
            else:
                warn("No .sfz found; deriving mapping from filenames and AIFF loop markers…")
                # First try local folder inference
                zones = derive_zones_from_folder(in_dir)
                # If nothing resolved and a samples-root was provided, try a recursive search
                if not zones and args.samples_root:
                    sr = Path(args.samples_root).expanduser().resolve()
                    if sr.is_dir():
                        info(f"Searching samples recursively under: {sr}")
                        zones = derive_zones_from_samples_root(sr, guess_instrument_name(exs_path))
                    else:
                        warn(f"samples-root not found: {sr}")
        if not zones:
            print("No zones resolved; nothing to convert.", file=sys.stderr)
            return 3

        info(f"Preparing DecentSampler bundle at: {bundle_dir}")
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Optional: apply default crossfade to zones missing one
        if (args.xfade_samples is not None or args.xfade_ms is not None) and zones:
            applied = 0
            for z in zones:
                if z.loop_start is not None and z.loop_end is not None and (z.loop_xfade is None or z.loop_xfade <= 0):
                    if args.xfade_samples is not None:
                        z.loop_xfade = max(0, int(args.xfade_samples))
                    else:
                        sr = get_sample_rate(z.sample_path)
                        z.loop_xfade = max(0, int(round((args.xfade_ms or 0.0) * sr / 1000.0)))
                    applied += 1
            if applied:
                info(f"Applied default loop crossfade to {applied} zone(s)")

        info("Copying samples...")
        copy_samples(zones, samples_dir)

        info("Building .dspreset...")
        root = ds_root()
        ds_add_metadata(root, instr_name)
        ds_add_groups(root, zones, samples_subdir='Samples')
        # Add effects with IDs so UI knobs can bind reliably.
        ds_add_all_effects(root)

        # Generate a simple gradient background (always), themed if requested
        theme = (args.theme or 'dark')
        bg_rel: Optional[str] = None
        try:
            bg_rel = generate_background(bundle_dir, instr_name, theme, args.bg_width, args.bg_height)
            if bg_rel:
                info(f"Background generated: {bg_rel}")
        except Exception as e:
            warn(f"Background generation failed: {e}")

        # Add UI
        if args.full_ui:
            ds_add_full_ui(root, title=instr_name, theme=theme, bg_rel_path=bg_rel, bg_w=args.bg_width, bg_h=args.bg_height)
        elif args.force_ui:
            ds_add_basic_ui(root)
        write_preset(root, preset_path)

        info("Done.")
        info(f"Bundle: {bundle_dir}")
        info(f"Preset: {preset_path}")
        return 0

    # Batch mode: convert all .exs in folder
    if args.batch:
        if args.exs:
            warn("--batch specified; ignoring --exs and converting all .exs in folder.")
        exs_files = sorted(in_dir.glob('*.exs'))
        if not exs_files:
            print("No .exs files found in folder for batch conversion.", file=sys.stderr)
            return 2
        total = len(exs_files)
        ok = 0
        for idx, exs_path in enumerate(exs_files, start=1):
            info(f"=== [{idx}/{total}] Converting: {exs_path.name} ===")
            rc = convert_one(exs_path.resolve())
            if rc == 0:
                ok += 1
            else:
                warn(f"Conversion failed for: {exs_path.name} (exit {rc})")
        info(f"Batch completed: {ok}/{total} succeeded")
        return 0 if ok == total else 1

    # Single conversion path (original behavior)
    exs_path: Optional[Path] = None
    if args.exs:
        exs_path = Path(args.exs).expanduser().resolve()
        if not exs_path.is_file():
            print(f"Specified .exs not found: {exs_path}", file=sys.stderr)
            return 2
    else:
        # Find first .exs file in folder
        for p in in_dir.glob('*.exs'):
            exs_path = p.resolve()
            break
        if exs_path is None:
            print("No .exs file found in folder. Use --exs to specify one.", file=sys.stderr)
            return 2

    return convert_one(exs_path)


if __name__ == '__main__':
    sys.exit(main())
