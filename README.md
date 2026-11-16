# 🎵 AI Music Generator (Windows)

โปรแกรมสร้างเพลงอัตโนมัติ ทำงานออฟไลน์บน Windows

| ตัวเลือก | ค่าที่ใช้ |
|---|---|
| แพลตฟอร์ม | Windows 10/11 (รันด้วย Python หรือ build เป็น `.exe`) |
| การสร้างดนตรี | `audio_midi` – ได้ทั้งไฟล์เสียง `.wav` และไฟล์ `.mid` |
| เสียงร้อง | `preserve` – เสียงร้องของคุณ **ไม่ถูกแก้ไข** ดนตรีถูกสร้างให้เข้ากับเสียงร้อง |
| แนวเพลง | `all` – 15 แนว (pop, rock, hiphop, edm, house, lofi, jazz, blues, rnb, country, reggae, latin, classical, ambient, metal) หรือสุ่มอัตโนมัติ |

> **หมายเหตุสำคัญ:** นี่คือ *ต้นแบบ (prototype)* ที่ใช้ทฤษฎีดนตรี + กฎของแต่ละแนว + ซินธ์ที่เขียนเอง
> ไม่ได้ใช้โมเดล AI สร้างเสียงสำเร็จรูป ผลลัพธ์เป็นเสียงสังเคราะห์พื้นฐาน เหมาะเป็นสเก็ตช์/แบ็คกิ้งแทร็ก/ไฟล์ MIDI
> เพื่อนำไปโหลดซาวด์ที่ดีกว่าใน DAW ต่อ ส่วนการต่อยอดไปโมเดล AI จริง ดูหัวข้อ [แผนต่อยอด](#-แผนต่อยอด)

---

## 📁 โครงสร้างโปรเจกต์

```
ai-music-generator/
├─ run_app.py              # จุดเริ่มโปรแกรม (ไม่มีอาร์กิวเมนต์ = เปิด GUI, มี = CLI)
├─ setup_windows.bat       # ติดตั้งครั้งแรก (สร้าง .venv + ติดตั้งแพ็กเกจ)
├─ run_gui.bat             # เปิดหน้าต่างโปรแกรม
├─ run_cli.bat             # รันแบบคำสั่ง
├─ build_exe.bat           # build เป็น .exe บนเครื่องตัวเอง
├─ requirements.txt        # numpy (จำเป็น) + soundfile (ไม่บังคับ)
├─ requirements-dev.txt    # สำหรับทดสอบ / build
├─ pyproject.toml
├─ src/aimusicgen/
│  ├─ theory.py            # สเกล คอร์ด ลำดับคอร์ด
│  ├─ genres.py            # โปรไฟล์ 15 แนวเพลง (BPM, สเกล, กลอง, เบส, คอร์ด, เมโลดี้, เสียง)
│  ├─ composer.py          # สร้างเพลง: ท่อน intro/verse/chorus/outro + กลอง/เบส/คอร์ด/ลีด
│  ├─ midi_io.py           # เขียน/อ่านไฟล์ MIDI (ไม่ต้องใช้ไลบรารีเสริม)
│  ├─ synth.py             # ซินธ์ numpy เรนเดอร์โน้ต → เสียงสเตอริโอ
│  ├─ vocal.py             # วิเคราะห์เสียงร้อง (BPM, คีย์) โดยไม่แตะต้องเสียงร้อง
│  ├─ mixer.py             # มิกซ์ ducking ลิมิตเตอร์ (ประมวลผลเฉพาะดนตรี ไม่ใช่เสียงร้อง)
│  ├─ audio_io.py          # อ่าน/เขียน WAV (16/24-bit)
│  ├─ pipeline.py          # ท่อประมวลผลครบวงจร
│  ├─ cli.py               # คำสั่งในเทอร์มินัล
│  └─ gui.py               # หน้าต่าง Tkinter
├─ tests/test_pipeline.py  # เทสต์อัตโนมัติ (21 เคส)
└─ .github/workflows/
   ├─ ci.yml               # รันเทสต์บน Windows ทุกครั้งที่ push
   └─ build-windows.yml    # build AIMusicGenerator.exe + แนบ Release
```

## ⚙️ ทำงานอย่างไร

```
(เสียงร้อง .wav) ──► vocal.py: หา BPM + คีย์ ──┐
                                               ▼
ตั้งค่า (แนว/BPM/คีย์/ความยาว) ──► composer.py ──► โน้ต (Drums/Bass/Chords/Lead)
                                               │
                              ┌────────────────┴───────────────┐
                              ▼                                ▼
                        midi_io.py → song.mid            synth.py → ดนตรีสเตอริโอ
                                                               │
                  เสียงร้องต้นฉบับ (ไม่แก้ไข) ─────────────────► mixer.py (ducking + มิกซ์)
                                                               ▼
                                                    song.wav + song_accompaniment.wav
```

**เสียงร้องแบบ preserve:** โปรแกรมโหลดไฟล์เสียงร้อง วิเคราะห์ BPM/คีย์เพื่อสร้างดนตรีให้เข้ากัน
แล้วนำเสียงร้องเดิมมาวางทับ *ตามต้นฉบับ* (ปรับได้แค่ระดับความดังด้วย `--vocal-gain` ค่าเริ่มต้น 0 dB = ไม่แตะ)
ดนตรีจะลดเสียงลงเล็กน้อยตอนมีเสียงร้อง (`--duck`) ได้ไฟล์ `song_accompaniment.wav` แยกให้ด้วย

ข้อจำกัดที่ควรรู้:
- BPM ที่ตรวจจับได้เป็นค่าประมาณ ถ้าเสียงร้องไม่มีจังหวะชัด (เช่น ร้องอิสระ) ให้กำหนด `--bpm` เอง
- คีย์ที่ตรวจจับอาจได้คีย์ขนาน/สัมพันธ์ (เช่น C major ↔ A minor) ซึ่งใช้โน้ตชุดเดียวกัน จึงยังเข้ากันได้
- รองรับ `.wav` ได้เลย ส่วน mp3/flac/m4a ต้องติดตั้ง `soundfile` หรือ `ffmpeg` เพิ่ม
- เสียงร้องที่จะ preserve ควรเป็นเสียงร้องเปล่า (acapella) ถ้าใส่เพลงเต็ม โปรแกรมจะไม่แยกดนตรีเดิมออกให้

---

## 🚀 วิธีรันบน Windows (ไม่ผ่าน GitHub)

1. ติดตั้ง **Python 3.10 ขึ้นไป** จาก https://www.python.org/downloads/ (ตอนติดตั้งติ๊ก **Add python.exe to PATH**)
2. ดาวน์โหลดโปรเจกต์ (ปุ่ม **Code → Download ZIP**) แล้วแตกไฟล์
3. ดับเบิลคลิก `setup_windows.bat` (ทำครั้งเดียว)
4. ดับเบิลคลิก `run_gui.bat` → เปิดหน้าต่างโปรแกรม
5. เลือกแนวเพลง (`all` = สุ่ม) → เลือกโหมด (Audio + MIDI) → (ถ้ามี) เลือกไฟล์เสียงร้อง → กด **Generate**
6. ไฟล์จะอยู่ที่โฟลเดอร์ `Music\AIMusicGenerator` (ปุ่ม **Open folder**)

### แบบคำสั่ง (CLI)

```bat
:: สร้างเพลงสุ่มแนว ได้ทั้ง .wav และ .mid
run_cli.bat

:: เลือกแนว ความยาว และ seed (seed เดิม = เพลงเดิม)
run_cli.bat -g lofi --bars 32 --seed 123 -o output -n my_song

:: คงเสียงร้องเดิม + สร้างดนตรีรอบเสียงร้อง (ดึง BPM/คีย์จากเสียงร้องอัตโนมัติ)
run_cli.bat --vocal "C:\Users\me\Music\my_vocal.wav" -g pop -o output -n with_vocal

:: ได้เฉพาะ MIDI / เฉพาะเสียง / แยกสเต็ม
run_cli.bat -g jazz --mode midi
run_cli.bat -g edm --mode audio --stems --bits 24
```

| ตัวเลือก | ความหมาย |
|---|---|
| `-g, --genre` | `all` หรือชื่อแนว (`--list-genres` เพื่อดูทั้งหมด) |
| `--mode` | `audio_midi` (ค่าเริ่มต้น) / `audio` / `midi` |
| `--vocal FILE` | ไฟล์เสียงร้องที่ต้องการคงไว้ |
| `--bars N` | ความยาวเป็นห้อง (ค่าเริ่มต้น 32) |
| `--bpm`, `--key`, `--scale` | กำหนดเอง (ไม่ใส่ = สุ่มตามแนว / ตามเสียงร้อง) |
| `--seed N` | ใช้ซ้ำเพื่อได้เพลงเดิม |
| `--lead on/off` | เมโลดี้ลีดที่โปรแกรมสร้าง (ปิดอัตโนมัติเมื่อมีเสียงร้อง) |
| `--duck DB` | ลดเสียงดนตรีตอนมีเสียงร้อง (ค่าเริ่มต้น 4 dB, 0 = ปิด) |
| `--vocal-gain DB` | ปรับความดังเสียงร้อง (0 = ไม่แตะ) |
| `--stems` | ส่งออกไฟล์แยกเครื่องดนตรี |
| `--bits 16/24` | ความละเอียด WAV |

ไฟล์ที่ได้: `song.mid`, `song.wav`, `song_accompaniment.wav`, `song_info.json` (และ `song_stem_*.wav`)

---

## 🐙 วิธีนำขึ้น GitHub แบบละเอียด

### ขั้นที่ 1 – สร้าง Repository
1. เข้า https://github.com แล้วล็อกอิน
2. กด **New repository** → ตั้งชื่อ `ai-music-generator` → เลือก Public หรือ Private → **อย่าติ๊ก** "Add a README" → **Create repository**

### ขั้นที่ 2 – อัปโหลดโค้ด
**ทางที่ 1: ใช้ Git (แนะนำ)** ติดตั้ง Git for Windows จาก https://git-scm.com แล้วเปิดโฟลเดอร์โปรเจกต์ → คลิกขวา → *Open Git Bash here*

```bash
git init
git add .
git commit -m "Initial commit: AI Music Generator"
git branch -M main
git remote add origin https://github.com/<ชื่อผู้ใช้>/ai-music-generator.git
git push -u origin main
```

**ทางที่ 2: ลากไฟล์ผ่านเว็บ** ในหน้า repo กด **Add file → Upload files** แล้วลากโฟลเดอร์/ไฟล์ทั้งหมดลงไป
(ต้องให้โฟลเดอร์ `.github/workflows` ขึ้นไปด้วย ถ้าโฟลเดอร์ซ่อนไม่ขึ้น ให้ใช้ทางที่ 1)

### ขั้นที่ 3 – ตรวจว่าเทสต์ผ่าน
เปิดแท็บ **Actions** → workflow **CI** จะรันเองหลัง push (เครื่อง Windows จริงของ GitHub, Python 3.10 และ 3.12)
เครื่องหมายถูกสีเขียว = โค้ดทำงานบน Windows ได้

### ขั้นที่ 4 – Build ไฟล์ .exe บน GitHub (ไม่ต้องลง Python เอง)
1. แท็บ **Actions** → เลือก **Build Windows EXE** ทางซ้าย → กด **Run workflow** → **Run workflow**
2. รอประมาณ 3–6 นาที เมื่อเสร็จ (สีเขียว) คลิกที่รอบที่รัน เลื่อนลงไปหัวข้อ **Artifacts**
3. ดาวน์โหลด **AIMusicGenerator-windows** (ไฟล์ zip) แตกไฟล์
   - `AIMusicGenerator.exe` – หน้าต่างโปรแกรม (ดับเบิลคลิก)
   - `AIMusicGenerator-cli.exe` – แบบคำสั่ง เช่น `AIMusicGenerator-cli.exe -g rock --bars 16`

> Windows SmartScreen อาจเตือนว่า "Windows protected your PC" เพราะไฟล์ไม่ได้ลงนามดิจิทัล → กด **More info → Run anyway**

### ขั้นที่ 5 (ไม่บังคับ) – ออก Release ให้คนอื่นดาวน์โหลด
```bash
git tag v0.1.0
git push origin v0.1.0
```
workflow จะ build แล้วแนบ `AIMusicGenerator-windows.zip` ไว้ในหน้า **Releases** อัตโนมัติ

### Build บนเครื่องตัวเอง
ดับเบิลคลิก `build_exe.bat` → ได้ไฟล์ในโฟลเดอร์ `dist`

---

## 🧪 ทดสอบ

```bat
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest -q
```

เทสต์ครอบคลุม: ทุกแนวสร้างเพลงได้, seed เดิมได้เพลงเดิม, MIDI เขียน/อ่านกลับตรงกัน, ไฟล์เสียงออกถูกต้อง,
และ **เสียงร้องถูกคงไว้จริง** (วัดจากการเทียบมิกซ์กับไฟล์เสียงร้องต้นฉบับ)

## 🛠 แก้ปัญหาเบื้องต้น

| อาการ | วิธีแก้ |
|---|---|
| `python` is not recognized | ติดตั้ง Python ใหม่แล้วติ๊ก *Add to PATH* หรือใช้ `py -m venv .venv` |
| GUI ไม่เปิด / ไม่มี tkinter | ติดตั้ง Python จาก python.org (ตัวติดตั้งมี tkinter มาให้ ต้องไม่ถอดติ๊ก *tcl/tk*) |
| อ่านไฟล์ mp3 ไม่ได้ | `pip install soundfile` หรือติดตั้ง ffmpeg หรือแปลงเป็น WAV ก่อน |
| ดนตรีไม่เข้ากับเสียงร้อง | กำหนด `--bpm` / `--key` / `--scale` เองให้ตรงกับเพลง |
| exe โดน antivirus บล็อก | พบบ่อยกับ PyInstaller ให้ใช้วิธีรันด้วย Python แทน หรือ build เองบนเครื่อง |

## 🔭 แผนต่อยอด

- ใช้โมเดล AI จริงสร้างเสียง: **MusicGen / Stable Audio** (ต้องมี GPU + PyTorch) แทนซินธ์ในโมดูล `synth.py`
- แยกเสียงร้องจากเพลงเต็มด้วย **Demucs** ก่อนเข้าโหมด preserve
- เมโลดี้จากเสียงร้อง: ใช้ **basic-pitch / CREPE** ถอดโน้ตแล้วสร้างคอร์ดให้ตรงกับทำนอง
- โหลด SoundFont (`.sf2`) ผ่าน **FluidSynth** เพื่อให้เสียงเครื่องดนตรีสมจริงขึ้น
- ปรับจังหวะเสียงร้องให้ตรงกริด (time-align / beat-sync) และส่งออกเป็น VST3 (โครงสร้างเดียวกับโปรเจกต์ JUCE)

## 📄 License
MIT – ดู [LICENSE](LICENSE)
