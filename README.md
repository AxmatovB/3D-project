# 🖐 Gesture 3D Creative Environment

Webcam orqali qo'l harekatları bilan boshqariladigan real-vaqt 3D ijodiy muhit.

---

## 📋 Talablar

| Dastur | Versiya |
|--------|---------|
| Python | 3.10+   |
| Webcam | Har qanday USB yoki built-in |
| OpenGL | GPU haydovchisi (odatda o'rnatilgan) |

---

## 🪟 WINDOWS — O'rnatish

### Birinchi marta (bir marta bajarish)

```bat
cd C:\
mkdir gesture3d
cd gesture3d
```

> `gesture_3d_env.py` faylini `C:\gesture3d\` papkasiga ko'chiring

```bat
python -m venv gesture3d-venv
gesture3d-venv\Scripts\activate
pip install opencv-python mediapipe==0.10.14 pygame PyOpenGL numpy
python gesture_3d_env.py
```

> ⚠️ Python o'rnatishda **"Add Python to PATH"** checkboxni albatta belgilang!

### Keyingi safar ishga tushirish

```bat
cd C:\gesture3d
gesture3d-venv\Scripts\activate
python gesture_3d_env.py
```

### Windows — Keng tarqalgan muammolar

| Muammo | Yechim |
|--------|--------|
| `'python' topilmadi` | Python qayta o'rnating, "Add to PATH" belgilang |
| `Scripts\activate` ishlamaydi | PowerShell da: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `mediapipe` o'rnatilmaydi | `pip install --upgrade pip` keyin qaytadan urining |
| OpenGL xatosi | GPU haydovchisini yangilang |
| Kamera ko'rinmaydi | Skype/Teams/Zoom ni yoping, ular kamerani tutib turishi mumkin |

---

## 🍎 macOS — O'rnatish

### Birinchi marta (bir marta bajarish)

```bash
# Homebrew yo'q bo'lsa o'rnating
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python o'rnatish
brew install python@3.11

# Papka va fayl
mkdir -p ~/gesture3d
cp ~/Downloads/gesture_3d_env.py ~/gesture3d/
cd ~/gesture3d

# venv yaratish va yoqish
python3 -m venv gesture3d-venv
source gesture3d-venv/bin/activate

# Kutubxonalar o'rnatish
pip install opencv-python mediapipe==0.10.14 pygame PyOpenGL numpy

# Ishga tushirish
python gesture_3d_env.py
```

> 📷 Birinchi ishga tushirganda macOS kamera ruxsatini so'raydi — **OK** bosing.
> Agar bloklangan bo'lsa: `System Settings > Privacy & Security > Camera > Terminal` ni yoqing.

### Keyingi safar ishga tushirish

```bash
cd ~/gesture3d
source gesture3d-venv/bin/activate
python gesture_3d_env.py
```

### macOS — Keng tarqalgan muammolar

| Muammo | Yechim |
|--------|--------|
| `python3` topilmadi | `brew install python@3.11` |
| mediapipe M1/M2/M3 da ishlamaydi | `pip install mediapipe==0.10.14` (ARM uchun ishlaydi) |
| OpenGL deprecated xabari | E'tiborsiz qoldiring — dastur ishlaydi, bu macOS 14+ da odatiy |
| Kamera ruxsati yo'q | `System Settings > Privacy > Camera > Terminal` ni yoqing |

---

## 🐧 LINUX — O'rnatish (Ubuntu/Debian)

### Birinchi marta (bir marta bajarish)

```bash
# Tizim kutubxonalari
sudo apt update
sudo apt install python3 python3-pip python3-venv python3-full \
                 libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 -y

# Papka va fayl
mkdir -p ~/gesture3d
cp ~/Downloads/gesture_3d_env.py ~/gesture3d/
cd ~/gesture3d

# venv yaratish va yoqish
python3 -m venv gesture3d-venv
source gesture3d-venv/bin/activate

# Kutubxonalar o'rnatish
pip install opencv-python mediapipe==0.10.14 pygame PyOpenGL numpy

# Ishga tushirish
python gesture_3d_env.py
```

> ℹ️ Kamera ko'rinmasa, foydalanuvchini `video` guruhiga qo'shing:
> ```bash
> sudo usermod -a -G video $USER
> newgrp video
> ```

### Keyingi safar ishga tushirish

```bash
cd ~/gesture3d
source gesture3d-venv/bin/activate
python gesture_3d_env.py
```

### Linux — Keng tarqalgan muammolar

| Muammo | Yechim |
|--------|--------|
| `libGL.so` topilmadi | `sudo apt install libgl1-mesa-glx` |
| Kamera `/dev/video0` yo'q | `sudo usermod -a -G video $USER` keyin qayta login |
| `externally-managed-environment` | venv ichida ekanliningizni tekshiring — `(gesture3d-venv)` ko'rinishi kerak |
| pygame display xatosi | `sudo apt install libsdl2-dev libsdl2-2.0-0` |
| `mediapipe.solutions` yo'q | `pip uninstall mediapipe -y && pip install mediapipe==0.10.14` |

---

## 🕹️ Gesturelar

### DRAW MODE

| Gesture | Natija |
|---------|--------|
| ☝️ Ko'rsatkich barmoq | Trail chizish — barmoqni ko'tarsang shakl aniqlanadi |
| ✊ Mushtlangan qo'l drag | Kamerani aylantirish |
| 🤏 Pinch (ob'ekt yaqin) | Ob'ektni ushlab siljitish |
| 🤏 Pinch (ob'ekt uzoq) | Kamerani zoom in/out |
| 👍 Bosh barmoq yuqori | Tanlangan ob'ekt rangini almashtirish |
| ✌️ Peace + silkitish | Tanlangan ob'ektni o'chirish |
| 🖐 Ochiq qo'l **3 soniya** | CREATE MODE ga o'tish |
| Ikki qo'l yoyish/yaqinlash | Butun sahnani kattalashtirish/kichraytirish |
| Ikki qo'l aylantirish | Butun sahnani aylantirish |
| Ikki qo'l pinch | Tanlangan ob'ektni scale qilish |

### CREATE MODE

| Gesture | Natija |
|---------|--------|
| ✌️ 2 barmoq | Kub (cube) paydo bo'ladi |
| 🤟 3 barmoq | Silindr (cylinder) paydo bo'ladi |
| 🖖 4 barmoq | Sfera (sphere) paydo bo'ladi |
| 🤏 Pinch | Ob'ektni ushlab siljitish |
| Ikki qo'l yoyish | Butun sahnani scale qilish |
| Ikki qo'l aylantirish | Butun sahnani aylantirish |
| Ikki qo'l ochiq **3 soniya** | DRAW MODE ga qaytish |

### Klaviatura

| Tugma | Natija |
|-------|--------|
| `← → ↑ ↓` | Kamerani aylantirish |
| Scroll wheel | Zoom in / out |
| Mouse drag (chap tugma) | Kamerani aylantirish |
| `Z` | Oxirgi ob'ektni bekor qilish (undo) |
| `R` | Sahnani to'liq tozalash (reset) |
| `Delete` / `X` | Tanlangan ob'ektni o'chirish |
| `ESC` / `Q` | Dasturdan chiqish |

---

## 📦 Chizilgan shakllar → 3D ob'ektlar

| Chizilgan shakl | 3D ob'ekt |
|-----------------|-----------|
| Doira | Sfera (Sphere) |
| Kvadrat / To'rtburchak | Kub (Cube) |
| Uchburchak | Piramida (Pyramid) |
| To'g'ri chiziq | Silindr (Cylinder) |

---

## ⚡ Tezkor buyruqlar — nusxa oling va ishlating

### 🪟 Windows — Birinchi o'rnatish
```bat
cd C:\ && mkdir gesture3d && cd gesture3d
python -m venv gesture3d-venv
gesture3d-venv\Scripts\activate
pip install opencv-python mediapipe==0.10.14 pygame PyOpenGL numpy
python gesture_3d_env.py
```

### 🪟 Windows — Keyingi safar
```bat
cd C:\gesture3d && gesture3d-venv\Scripts\activate && python gesture_3d_env.py
```

### 🍎 macOS — Birinchi o'rnatish
```bash
mkdir -p ~/gesture3d && cp ~/Downloads/gesture_3d_env.py ~/gesture3d/ && cd ~/gesture3d
python3 -m venv gesture3d-venv && source gesture3d-venv/bin/activate
pip install opencv-python mediapipe==0.10.14 pygame PyOpenGL numpy
python gesture_3d_env.py
```

### 🍎 macOS — Keyingi safar
```bash
cd ~/gesture3d && source gesture3d-venv/bin/activate && python gesture_3d_env.py
```

### 🐧 Linux — Birinchi o'rnatish
```bash
sudo apt install python3 python3-venv python3-full libgl1-mesa-glx libglib2.0-0 -y
mkdir -p ~/gesture3d && cp ~/Downloads/gesture_3d_env.py ~/gesture3d/ && cd ~/gesture3d
python3 -m venv gesture3d-venv && source gesture3d-venv/bin/activate
pip install opencv-python mediapipe==0.10.14 pygame PyOpenGL numpy
python gesture_3d_env.py
```

### 🐧 Linux — Keyingi safar
```bash
cd ~/gesture3d && source gesture3d-venv/bin/activate && python gesture_3d_env.py
```

### 🔧 mediapipe muammosi bo'lsa (barcha platformalar)
```bash
pip uninstall mediapipe -y
pip install mediapipe==0.10.14
python3 -c "import mediapipe as mp; print(mp.solutions.hands); print('OK')"
```
