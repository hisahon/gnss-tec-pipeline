# -*- coding: utf-8 -*-
"""
dtec_videos.py -- one video (MP4, H.264) per day, made from the frames of dtec_frames.py.

Input : IN_DIR / year / doy / dtec_YYYYmmdd_HHMMSS.jpg        (the folder layout of dtec_frames.py)
Output: VIDEO_DIR / year / dtec_YYYY_DOY.mp4

The frames of a day, in time order, are sent to ffmpeg through a pipe: no temporary images, no
renaming, and a missing frame does not break anything (it is only reported).

Speed: ffmpeg decodes JPEG with ONE thread (about 24 frames/s for 3141 x 2126), and that, not the
H.264 encoder, is what limits a single ffmpeg. So each day is cut into WORKERS consecutive slices,
each slice is encoded by its own ffmpeg at the same time, and the slices are glued together at the end
without re-encoding (a few seconds). Each slice starts with a key frame, so the cut cannot be seen.

The video is written under a temporary name and renamed when everything is done, so an interrupted run
never leaves a half-written video that the next run would skip as "already done". (An interrupted run
may leave files called dtec_YYYY_DOY.sliceN.mp4: they are overwritten by the next run, delete them freely.)

Needs ffmpeg: on the PATH, or `pip install imageio-ffmpeg` (it brings its own copy), or set FFMPEG below.
"""
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------- parameters
IN_DIR      = Path(r"D:\database\ionosphere\dtecframes")    # ROOT of the frames (the OUT_DIR of dtec_frames.py)
VIDEO_DIR   = Path(r"D:\database\ionosphere\dtecvideos")    # ROOT of the videos
IMG_EXT     = "jpg"          # which frames to read: "jpg" or "png" (the IMG_FORMAT used by dtec_frames.py)
CADENCE     = 30             # seconds between two frames (only used to count the missing ones)
FPS         = 30             # frames per second of the video: 2880 frames (24 h) -> 3 min 12 s
VIDEO_WIDTH = None           # None = original size. Or a width in pixels, e.g. 1920 (height follows):
                             # smaller file, faster to encode, easier to play
CRF         = 20             # H.264 quality: lower = better and larger (18 ~ visually lossless, 23 = ffmpeg default)
PRESET      = "veryfast"     # "medium" / "slow": smaller file for the same quality, but much slower
PIX_FMT     = "yuv420p"      # plays everywhere (browsers, PowerPoint, Windows). The colour is stored at half the
                             # resolution, so thin coloured lines and text get a faint colour blur.
                             # "yuv444p" keeps it sharp, but many players cannot open it
WORKERS     = max(1, (os.cpu_count() or 2) // 4)     # ffmpeg processes working at the same time on the SAME day.
                             # Each one takes ~1 core to decode plus a few for the encoder: 1/4 of the cores is a
                             # safe start, raise it while the processor is not full (1 = a single ffmpeg)
FFMPEG      = None           # None = automatic (PATH, then imageio-ffmpeg). Or the full path: r"C:\ffmpeg\bin\ffmpeg.exe"
OVERWRITE   = False          # False: days whose video already exists are skipped (set True to redo a day
                             # whose frames were completed afterwards)


# ---------------------------------------------------------------- ffmpeg
def ffmpeg_exe():
    """The ffmpeg to use: FFMPEG, else the one on the PATH, else the copy that imageio-ffmpeg brings."""
    if FFMPEG:
        return str(FFMPEG)
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise FileNotFoundError("ffmpeg not found. Install it and put it on the PATH "
                                "(https://www.ffmpeg.org/download.html), or run `pip install imageio-ffmpeg`, "
                                "or set FFMPEG = r'C:\\...\\ffmpeg.exe' in this file.") from None


def run_ffmpeg(args, frames=(), what=""):
    """Run ffmpeg; `frames` (image files) are sent to its stdin, in order. Raises RuntimeError with ffmpeg's own
    message if it fails."""
    with tempfile.TemporaryFile() as err:                            # ffmpeg messages (a file: a full pipe would block)
        proc = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error"] + args, stdin=subprocess.PIPE, stderr=err)
        try:
            for f in frames:
                proc.stdin.write(Path(f).read_bytes())
        except OSError:                                              # ffmpeg quit early: its message is shown below
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
        code = proc.wait()
        err.seek(0)
        message = err.read().decode(errors="replace").strip()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed (exit code {code}) on {what}:\n{message}")


def encode(frames, dest, threads, vf):
    """These frames -> one MP4 file (dest)."""
    run_ffmpeg(["-framerate", str(FPS), "-f", "image2pipe", "-i", "-",       # the frames arrive on stdin
                "-vf", vf, "-c:v", "libx264", "-threads", str(threads), "-preset", PRESET, "-crf", str(CRF),
                "-pix_fmt", PIX_FMT, "-movflags", "+faststart", "-f", "mp4", str(dest)],
               frames, what=Path(dest).name)
    if not Path(dest).exists() or Path(dest).stat().st_size == 0:
        raise RuntimeError(f"ffmpeg wrote no video: {dest}")


def join(slices, dest):
    """Glue MP4 files with the same settings end to end, without re-encoding."""
    with tempfile.TemporaryDirectory() as folder:
        listing = Path(folder) / "slices.txt"
        listing.write_text("".join("file '" + s.as_posix().replace("'", "'\\''") + "'\n" for s in slices),
                           encoding="utf-8")
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart",
                    "-f", "mp4", str(dest)], what=f"joining the slices of {dest.name}")


def make_video(day_dir, out_file):
    """Frames of one day -> one MP4. Returns (frames used, frames missing)."""
    frames = sorted(Path(day_dir).glob(f"dtec_*.{IMG_EXT}"))       # the names carry the time: sorted = chronological
    if not frames:
        raise FileNotFoundError(f"no dtec_*.{IMG_EXT} frames in {day_dir}")
    stamps = [datetime.strptime(f.stem[5:], "%Y%m%d_%H%M%S") for f in frames]       # dtec_YYYYmmdd_HHMMSS
    missing = int(round((stamps[-1] - stamps[0]).total_seconds() / CADENCE)) + 1 - len(frames)

    # H.264 with 4:2:0 colour needs even sides. Original size: drop the last column / row if odd
    # (it is white margin). With VIDEO_WIDTH: scale, the height is chosen by ffmpeg (-2 = even).
    vf = f"scale={VIDEO_WIDTH // 2 * 2}:-2:flags=lanczos" if VIDEO_WIDTH else "crop=trunc(iw/2)*2:trunc(ih/2)*2"
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_name(out_file.name + ".part")                # renamed to out_file when finished

    n = max(1, min(WORKERS, len(frames) // 60))                      # slices: at least 60 frames (4 s) each
    if n == 1:
        encode(frames, tmp, 0, vf)                                   # 0 = the encoder chooses its threads
    else:
        cuts = [len(frames) * i // n for i in range(n + 1)]          # consecutive, equal slices
        slices = [out_file.with_name(f"{out_file.stem}.slice{i}.mp4") for i in range(n)]
        threads = max(2, (os.cpu_count() or 2) // n)                 # the cores are shared among the slices
        try:
            with ThreadPoolExecutor(n) as pool:                      # these threads only wait: the work is in the ffmpegs
                list(pool.map(lambda i: encode(frames[cuts[i]:cuts[i + 1]], slices[i], threads, vf), range(n)))
            join(slices, tmp)
        finally:
            for s in slices:
                s.unlink(missing_ok=True)
    tmp.replace(out_file)
    return len(frames), missing


# ---------------------------------------------------------------- run
def main(year=None, doys=None):
    """One video per day folder found under IN_DIR/year/doy.
    year: None = all the years found, or one year (int).
    doys: None = all the days found, or a list of day-of-year numbers, e.g. [296, 297]."""
    root = Path(IN_DIR).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"IN_DIR does not exist: {root}")
    ffmpeg_exe()                                                     # fail now, not after listing the days
    days = [(int(y.name), int(d.name), d)
            for y in sorted(root.iterdir()) if y.is_dir() and y.name.isdigit() and (year is None or int(y.name) == year)
            for d in sorted(y.iterdir()) if d.is_dir() and d.name.isdigit() and (doys is None or int(d.name) in doys)]
    if not days:
        raise FileNotFoundError(f"no year/doy folders under {root} for year={year}, doys={doys}")

    print(f"{len(days)} days | {IMG_EXT} frames at {FPS} fps | CRF {CRF}, preset {PRESET}, "
          f"{'original size' if not VIDEO_WIDTH else f'width {VIDEO_WIDTH}'} | {WORKERS} ffmpeg at a time | "
          f"videos in {Path(VIDEO_DIR).expanduser()}")
    done, skipped, failed = 0, 0, []
    for y, d, day_dir in days:
        out_file = Path(VIDEO_DIR).expanduser() / str(y) / f"dtec_{y}_{d:03d}.mp4"
        if out_file.exists() and not OVERWRITE:
            skipped += 1
            continue
        t0 = time.perf_counter()
        try:
            n, missing = make_video(day_dir, out_file)
        except (FileNotFoundError, RuntimeError) as e:               # one bad day does not stop the others
            print(f"{y} {d:03d}: FAILED -- {e}")
            failed.append((y, d))
            continue
        done += 1
        seconds = time.perf_counter() - t0
        print(f"{y} {d:03d}: {n} frames" + (f" ({missing} MISSING, the video skips them)" if missing else "")
              + f" -> {out_file.name}, {out_file.stat().st_size / 1e6:.0f} MB, {n / FPS:.0f} s of video, "
              f"{seconds:.0f} s ({n / seconds:.1f} frames/s)")
    print(f"{done} videos made, {skipped} skipped (already exist), {len(failed)} failed"
          + (f": {failed}" if failed else ""))


if __name__ == "__main__":
    main()                                                           # all days found. Or: main(year=2021, doys=[296, 297])