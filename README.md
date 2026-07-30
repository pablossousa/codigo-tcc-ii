# Real-Time Face Recognition System

A Windows desktop application for real-time facial recognition using a webcam, local SQLite storage, encrypted embeddings, and automatic card-code typing.

This project was developed as a practical academic/prototype solution for identity recognition, multi-pose registration, and secure local processing without relying on external servers.

## Overview

The system allows users to:

- register people by collecting face samples from multiple poses;
- recognize faces in real time using the webcam;
- compare detected faces against stored embeddings;
- delete registered identities;
- automatically type a recognized card code into the active field using the keyboard.

The application is built in Python and uses:

- OpenCV for camera capture and image processing;
- InsightFace for face detection and feature extraction;
- SQLite for persistent local storage;
- Fernet encryption for secure embedding storage;
- Tkinter for the graphical interface;
- pynput for automatic keyboard input.

## Features

- Real-time face detection and recognition
- Multi-pose enrollment for better robustness
- Temporal voting logic to stabilize recognition decisions
- Local database storage with encrypted embeddings
- Face quality checks based on size, sharpness, and confidence
- Automatic recognition of card IDs with cooldown and repetition control
- Windows-friendly setup scripts
- Browser-free desktop operation on the local machine

## Tech Stack

- Python 3.10
- OpenCV
- InsightFace
- ONNX Runtime
- NumPy
- Pillow
- SQLite
- cryptography
- pynput
- Tkinter

## Project Structure

```text
TCC_teste_robusto/
├── face_recognition_app.py
├── requirements.txt
├── setup.ps1
├── run.ps1
├── build_windows_app.ps1
├── SistemaReconhecimentoFacial.spec
├── data/
│   ├── faces.db
│   └── fernet.key
├── build/
│   └── SistemaReconhecimentoFacial/
└── README.md
```

## Requirements

Before running the project, make sure the following are available:

- Windows 10 or newer
- Python 3.10
- Webcam connected to the machine
- Internet access for installing Python dependencies (first setup only)

## Installation

### Option 1: PowerShell setup script

From the project root, run:

```powershell
./setup.ps1
```

This script:

- creates a virtual environment in `.venv`;
- installs dependencies from `requirements.txt`;
- validates the main imports.

### Option 2: Manual setup

```powershell
python -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Running the Application

After setup, run:

```powershell
./run.ps1
```

You can also start it directly with:

```powershell
.\.venv\Scripts\python.exe .\face_recognition_app.py
```

## Main Usage

The app provides a desktop interface with:

- a live camera preview;
- registration controls;
- recognition status;
- person deletion options;
- automatic typing settings.

Typical workflow:

1. Start the application.
2. Register a user with face samples from different poses.
3. Keep the face in the camera frame.
4. The system compares it with stored embeddings.
5. If a match is found, it displays the recognized identity and can type the card code automatically.

## Configuration and CLI Options

The application supports several command-line arguments, including:

```powershell
python .\face_recognition_app.py --camera-index 0 --window-size 20 --min-votes 6 --disable-auto-type
```

Some available options:

- `--camera-index`: webcam index
- `--window-size`: temporal recognition window size
- `--min-valid-frames`: minimum valid frames before a decision
- `--min-votes`: minimum votes needed to confirm a match
- `--final-threshold`: recognition threshold for final decision
- `--unknown-threshold`: threshold for identifying an unknown face
- `--insightface-det-threshold`: detection confidence threshold
- `--samples-per-pose`: number of samples captured per pose
- `--disable-auto-type`: disable automatic card-code entry
- `--press-enter-after-typing`: send Enter after typing the code
- `--db-path`: custom SQLite database path
- `--key-path`: custom Fernet key path

## Data and Security

The application stores face-related data locally in the `data` directory:

- `faces.db`: SQLite database with user and embedding records
- `fernet.key`: encryption key used to protect stored embeddings

Important:

- keep the `fernet.key` file safe;
- if the key is lost or replaced, previously stored embeddings cannot be decrypted;
- the system is designed for local use and does not send biometric data externally.

## Notes

This project is intended for academic use, research, prototyping, and local biometric demonstrations. It is not a production-grade surveillance or enterprise access-control system.

A few characteristics of the current implementation:

- recognition is performed locally on the machine;
- it uses multiple frames and a voting mechanism to reduce false matches;
- face registration requires multiple valid samples for better stability;
- automatic typing depends on system focus and the availability of the keyboard input library.

## Troubleshooting

### Camera not detected

- verify that the webcam is connected;
- try a different camera index with `--camera-index`;
- ensure the camera is not being used by another application.

### InsightFace model issues

- confirm that dependencies were installed correctly;
- verify that the environment matches the required Python version;
- reinstall the requirements if the model import fails.

### Automatic typing does not work

- make sure the target field is focused in the active application;
- check whether `pynput` is installed successfully;
- confirm that auto typing is enabled in the application.

## License

This project does not include a separate license file. Please check with the project owner before commercial use or redistribution.

## Contact

For questions or project collaboration, use the repository or project maintainer contact details provided in your academic or organizational context.
