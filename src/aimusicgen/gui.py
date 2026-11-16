"""Tkinter GUI (tkinter ships with the python.org Windows installer - no extra install)."""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import __version__
from .genres import GENRE_NAMES
from .pipeline import GenerateOptions, GenerateResult, generate
from .theory import NOTE_NAMES, SCALES


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"AI Music Generator {__version__}")
        self.geometry("720x640")
        self.minsize(680, 600)
        self.q: queue.Queue = queue.Queue()
        self.result: GenerateResult | None = None
        self.worker: threading.Thread | None = None

        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)

        self.v_genre = tk.StringVar(value="all")
        self.v_mode = tk.StringVar(value="audio_midi")
        self.v_vocal = tk.StringVar()
        self.v_out = tk.StringVar(value=str(Path.home() / "Music" / "AIMusicGenerator"))
        self.v_name = tk.StringVar(value="song")
        self.v_bars = tk.IntVar(value=32)
        self.v_bpm = tk.StringVar()
        self.v_key = tk.StringVar(value="auto")
        self.v_scale = tk.StringVar(value="auto")
        self.v_seed = tk.StringVar()
        self.v_lead = tk.BooleanVar(value=True)
        self.v_stems = tk.BooleanVar(value=False)
        self.v_duck = tk.DoubleVar(value=4.0)

        r = 0
        ttk.Label(frm, text="Genre").grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(frm, textvariable=self.v_genre, values=["all"] + GENRE_NAMES, state="readonly").grid(
            row=r, column=1, sticky="ew", **pad)
        r += 1
        ttk.Label(frm, text="Output").grid(row=r, column=0, sticky="w", **pad)
        mf = ttk.Frame(frm)
        mf.grid(row=r, column=1, sticky="w", **pad)
        for text, val in (("Audio + MIDI", "audio_midi"), ("Audio only", "audio"), ("MIDI only", "midi")):
            ttk.Radiobutton(mf, text=text, value=val, variable=self.v_mode).pack(side="left", padx=6)
        r += 1
        ttk.Label(frm, text="Vocal to preserve").grid(row=r, column=0, sticky="w", **pad)
        vf = ttk.Frame(frm)
        vf.grid(row=r, column=1, sticky="ew", **pad)
        vf.columnconfigure(0, weight=1)
        ttk.Entry(vf, textvariable=self.v_vocal).grid(row=0, column=0, sticky="ew")
        ttk.Button(vf, text="Browse...", command=self.pick_vocal).grid(row=0, column=1, padx=4)
        ttk.Button(vf, text="Clear", command=lambda: self.v_vocal.set("")).grid(row=0, column=2)
        r += 1
        ttk.Label(frm, text="(vocal stays untouched; music is generated around it, in its tempo & key)",
                  foreground="#666").grid(row=r, column=1, sticky="w", padx=8)
        r += 1

        sf = ttk.LabelFrame(frm, text="Settings (leave blank / auto for automatic)", padding=8)
        sf.grid(row=r, column=0, columnspan=2, sticky="ew", **pad)
        for i in range(6):
            sf.columnconfigure(i, weight=1)
        ttk.Label(sf, text="Bars").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(sf, from_=8, to=256, textvariable=self.v_bars, width=6).grid(row=0, column=1, sticky="w")
        ttk.Label(sf, text="BPM").grid(row=0, column=2, sticky="w")
        ttk.Entry(sf, textvariable=self.v_bpm, width=6).grid(row=0, column=3, sticky="w")
        ttk.Label(sf, text="Seed").grid(row=0, column=4, sticky="w")
        ttk.Entry(sf, textvariable=self.v_seed, width=10).grid(row=0, column=5, sticky="w")
        ttk.Label(sf, text="Key").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Combobox(sf, textvariable=self.v_key, values=["auto"] + NOTE_NAMES, width=6, state="readonly").grid(
            row=1, column=1, sticky="w")
        ttk.Label(sf, text="Scale").grid(row=1, column=2, sticky="w")
        ttk.Combobox(sf, textvariable=self.v_scale, values=["auto"] + sorted(SCALES), width=14, state="readonly").grid(
            row=1, column=3, columnspan=2, sticky="w")
        ttk.Label(sf, text="Duck vocal (dB)").grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Spinbox(sf, from_=0, to=12, increment=0.5, textvariable=self.v_duck, width=6).grid(
            row=2, column=2, sticky="w")
        ttk.Checkbutton(sf, text="Lead melody (no vocal)", variable=self.v_lead).grid(row=2, column=3, columnspan=2, sticky="w")
        ttk.Checkbutton(sf, text="Export stems", variable=self.v_stems).grid(row=2, column=5, sticky="w")
        r += 1

        ttk.Label(frm, text="Output folder").grid(row=r, column=0, sticky="w", **pad)
        of = ttk.Frame(frm)
        of.grid(row=r, column=1, sticky="ew", **pad)
        of.columnconfigure(0, weight=1)
        ttk.Entry(of, textvariable=self.v_out).grid(row=0, column=0, sticky="ew")
        ttk.Button(of, text="Browse...", command=self.pick_out).grid(row=0, column=1, padx=4)
        r += 1
        ttk.Label(frm, text="File name").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.v_name).grid(row=r, column=1, sticky="ew", **pad)
        r += 1

        bf = ttk.Frame(frm)
        bf.grid(row=r, column=0, columnspan=2, sticky="ew", pady=8)
        self.btn_gen = ttk.Button(bf, text="Generate", command=self.on_generate)
        self.btn_gen.pack(side="left", padx=4)
        self.btn_play = ttk.Button(bf, text="Play", command=self.on_play, state="disabled")
        self.btn_play.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(bf, text="Stop", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        self.btn_open = ttk.Button(bf, text="Open folder", command=self.on_open, state="disabled")
        self.btn_open.pack(side="left", padx=4)
        r += 1

        self.progress = ttk.Progressbar(frm, maximum=100)
        self.progress.grid(row=r, column=0, columnspan=2, sticky="ew", **pad)
        r += 1
        self.log = tk.Text(frm, height=10, state="disabled", wrap="word")
        self.log.grid(row=r, column=0, columnspan=2, sticky="nsew", **pad)
        frm.rowconfigure(r, weight=1)

        self.after(100, self.poll)

    # ---- helpers ----
    def write(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def pick_vocal(self) -> None:
        p = filedialog.askopenfilename(
            title="Choose vocal file",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg"), ("All files", "*.*")])
        if p:
            self.v_vocal.set(p)

    def pick_out(self) -> None:
        p = filedialog.askdirectory(title="Choose output folder")
        if p:
            self.v_out.set(p)

    def options(self) -> GenerateOptions:
        bpm = self.v_bpm.get().strip()
        seed = self.v_seed.get().strip()
        vocal = self.v_vocal.get().strip() or None
        return GenerateOptions(
            genre=self.v_genre.get(), output_dir=self.v_out.get(), name=self.v_name.get().strip() or "song",
            bars=int(self.v_bars.get()), bpm=float(bpm) if bpm else None,
            key=None if self.v_key.get() == "auto" else self.v_key.get(),
            scale=None if self.v_scale.get() == "auto" else self.v_scale.get(),
            seed=int(seed) if seed else None, mode=self.v_mode.get(), vocal_path=vocal,
            lead=(self.v_lead.get() if vocal is None else False), export_stems=self.v_stems.get(),
            duck_db=float(self.v_duck.get()),
        )

    # ---- actions ----
    def on_generate(self) -> None:
        try:
            opts = self.options()
        except ValueError as e:
            messagebox.showerror("Invalid setting", str(e))
            return
        self.btn_gen.configure(state="disabled")
        self.btn_play.configure(state="disabled")
        self.progress["value"] = 0
        self.write("--- Generating ---")

        def work() -> None:
            try:
                res = generate(opts, lambda p, m: self.q.put(("progress", p, m)))
                self.q.put(("done", res))
            except Exception:
                self.q.put(("error", traceback.format_exc()))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def poll(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "progress":
                    self.progress["value"] = item[1] * 100
                    self.write(item[2])
                elif item[0] == "done":
                    self.finish(item[1])
                elif item[0] == "error":
                    self.btn_gen.configure(state="normal")
                    self.write(item[1])
                    messagebox.showerror("Error", item[1].strip().splitlines()[-1])
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def finish(self, res: GenerateResult) -> None:
        self.result = res
        s = res.song
        self.write(f"Genre {s.genre} | {s.bpm:g} BPM | {s.key_name} | {s.bars} bars | seed {s.seed}")
        if res.vocal:
            self.write(f"Vocal preserved (detected {res.vocal.bpm or '?'} BPM, {res.vocal.key_name})")
        for p in (res.midi_path, res.audio_path, res.accompaniment_path):
            if p:
                self.write(f"Saved: {p}")
        self.write(f"Done in {res.seconds_taken:.1f}s")
        self.btn_gen.configure(state="normal")
        self.btn_open.configure(state="normal")
        if res.audio_path:
            self.btn_play.configure(state="normal")

    def on_play(self) -> None:
        if not (self.result and self.result.audio_path):
            return
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(str(self.result.audio_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.btn_stop.configure(state="normal")
        else:
            self.write("Preview playback is available on Windows only; open the WAV in your player.")

    def on_stop(self) -> None:
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)

    def on_open(self) -> None:
        folder = self.v_out.get()
        if sys.platform == "win32":
            os.startfile(folder)  # type: ignore[attr-defined]
        else:
            self.write(f"Output folder: {folder}")


def run_gui() -> None:
    App().mainloop()


if __name__ == "__main__":
    run_gui()
