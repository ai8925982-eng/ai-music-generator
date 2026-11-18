"""Write a short synthetic 'vocal' (A minor melody, 96 BPM) for smoke tests: python tools/make_test_vocal.py out.wav|out.flac"""
import sys
import numpy as np

sr, bpm = 44100, 96
spb = 60 / bpm
notes = [69, 72, 71, 69, 67, 64, 67, 69, 72, 74, 72, 69, 71, 69, 67, 64] * 2
sig = np.zeros(int(sr * (1.3 + len(notes) * spb / 2 + 5 * spb)))
t0 = 1.3
for n in notes:
    d = int(spb / 2 * sr)
    t = np.arange(d) / sr
    f = 440 * 2 ** ((n - 69) / 12)
    s = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.minimum(1, t * 60) * np.minimum(1, (d / sr - t) * 60) * 0.4
    a = int(t0 * sr)
    sig[a:a + d] += s[:len(sig) - a]
    t0 += spb / 2
out = sys.argv[1] if len(sys.argv) > 1 else "test_vocal.wav"
if out.lower().endswith(".wav"):
    import wave
    pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm)
else:
    import soundfile as sf
    sf.write(out, sig, sr)
print("wrote", out)
