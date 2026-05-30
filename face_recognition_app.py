from __future__ import annotations

import argparse
from importlib import metadata as importlib_metadata
import logging
import sqlite3
import time
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import tkinter as tk
from cryptography.fernet import Fernet
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, ImageTk
from tkinter import messagebox, simpledialog, ttk


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


POSE_LABELS = {
    "front": "frente",
    "left": "direita",
    "right": "esquerda",
}

POSE_POPUP_MESSAGES = {
    "front": "Olhe para frente para iniciar a captura.",
    "left": "Agora vire o rosto levemente para a direita.",
    "right": "Agora vire o rosto levemente para a esquerda.",
}

CARD_CODE_MAX_LENGTH = 8
HEX_DIGITS = set("0123456789ABCDEF")


def normalize_card_code(value: str) -> str:
    return value.strip().upper()


def validate_card_code(value: str) -> Tuple[bool, str, str]:
    card_code = normalize_card_code(value)
    if not card_code:
        return False, card_code, "O código do cartão não pode ficar vazio."
    if len(card_code) > CARD_CODE_MAX_LENGTH:
        return False, card_code, f"O código do cartão deve ter no máximo {CARD_CODE_MAX_LENGTH} dígitos."
    if any(character not in HEX_DIGITS for character in card_code):
        return False, card_code, "O código do cartão deve conter apenas dígitos hexadecimais: 0-9 e A-F."
    return True, card_code, ""


@dataclass(slots=True)
class AppConfig:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_warmup_frames: int = 12
    frame_candidate_threshold: float = 1.02
    final_distance_threshold: float = 0.90
    unknown_distance_threshold: float = 1.10
    distance_margin: float = 0.10
    temporal_window: int = 20
    min_valid_frames: int = 8
    min_confirm_votes: int = 6
    min_vote_ratio: float = 0.30
    samples_per_pose: int = 15
    min_samples_per_pose: int = 8
    sample_capture_interval: float = 0.25
    pose_capture_timeout: float = 35.0
    mtcnn_confidence: float = 0.90
    soft_min_face_ratio: float = 0.06
    hard_min_face_ratio: float = 0.04
    soft_max_face_ratio: float = 0.76
    hard_max_face_ratio: float = 0.88
    soft_min_sharpness: float = 18.0
    hard_min_sharpness: float = 10.0
    pose_threshold: float = 0.12
    alignment_size: int = 160
    cache_ttl_seconds: float = 10.0
    type_recognized_id: bool = True
    type_cooldown_seconds: float = 5.0
    max_consecutive_typings_per_id: int = 1
    press_enter_after_typing: bool = False
    require_stable_recognition_before_typing: bool = True
    min_window_width: int = 420
    min_window_height: int = 300
    sidebar_width: int = 280
    sidebar_hide_width: int = 760
    sidebar_show_width: int = 860
    header_color: str = "#17202a"
    sidebar_color: str = "#17202a"
    button_text_color: str = "#1F2933"
    register_button_color: str = "#A8D5BA"
    delete_button_color: str = "#F4A6A6"
    close_button_color: str = "#EFA1A1"
    typing_control_color: str = "#A7C7E7"
    window_name: str = "Reconhecimento Facial em Tempo Real"
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    db_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "faces.db")
    key_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "fernet.key")


@dataclass(slots=True)
class FaceSample:
    sample_id: int
    user_id: int
    registration_id: str
    user_name: str
    pose: str
    embedding: np.ndarray
    created_at: str


@dataclass(slots=True)
class FaceDetection:
    found: bool
    box: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    pose: str = "front"
    feedback: str = "Procurando rosto"
    embedding: Optional[np.ndarray] = None
    face_ratio: float = 0.0
    sharpness: float = 0.0
    landmarks: Optional[np.ndarray] = None


@dataclass(slots=True)
class MatchCandidate:
    user_id: int
    registration_id: str
    user_name: str
    distance: float
    effective_distance: float
    matched_pose: str
    is_within_candidate_threshold: bool


@dataclass(slots=True)
class FrameVote:
    user_id: Optional[int]
    registration_id: Optional[str]
    user_name: Optional[str]
    distance: float
    best_distance: float
    detected_pose: str
    matched_pose: Optional[str]
    timestamp: float


@dataclass(slots=True)
class TemporalDecision:
    status: str
    user_id: Optional[int] = None
    registration_id: Optional[str] = None
    user_name: Optional[str] = None
    votes: int = 0
    total_frames: int = 0
    vote_ratio: float = 0.0
    mean_distance: float = float("inf")
    best_distance: float = float("inf")
    message: str = ""


def ensure_float32(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=np.float32)


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    vector = ensure_float32(vector).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return ensure_float32(vector / norm)


def average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    stacked = np.vstack([ensure_float32(embedding) for embedding in embeddings])
    return normalize_embedding(np.mean(stacked, axis=0))


def text_for_opencv(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")


def clamp_box(box: np.ndarray, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box.astype(np.int32)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


def estimate_pose(landmarks: np.ndarray, pose_threshold: float = 0.12) -> str:
    if landmarks is None or landmarks.shape[0] < 5:
        return "front"

    left_eye, right_eye, nose, mouth_left, mouth_right = landmarks.astype(np.float32)
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_distance = max(abs(right_eye[0] - left_eye[0]), 1.0)

    nose_offset = (nose[0] - eye_mid_x) / eye_distance
    left_nose_gap = abs(nose[0] - left_eye[0])
    right_nose_gap = abs(right_eye[0] - nose[0])
    gap_asymmetry = (right_nose_gap - left_nose_gap) / eye_distance

    mouth_mid_x = (mouth_left[0] + mouth_right[0]) / 2.0
    mouth_offset = (nose[0] - mouth_mid_x) / eye_distance

    score = (0.5 * nose_offset) + (0.3 * gap_asymmetry) + (0.2 * mouth_offset)
    if score <= -pose_threshold:
        return "left"
    if score >= pose_threshold:
        return "right"
    return "front"


def opencv_highgui_available() -> bool:
    try:
        build_info = cv2.getBuildInformation()
        for line in build_info.splitlines():
            stripped = line.strip()
            if stripped.startswith("GUI:"):
                return "NONE" not in stripped.upper()
    except Exception:
        pass

    try:
        test_name = "__codex_highgui_test__"
        cv2.namedWindow(test_name, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(test_name)
        return True
    except cv2.error:
        return False


def collect_environment_warnings() -> List[str]:
    warnings: List[str] = []

    try:
        numpy_major = int(str(np.__version__).split(".", maxsplit=1)[0])
        if numpy_major >= 2:
            warnings.append(
                f"NumPy {np.__version__} detectado. Para melhor compatibilidade com torch/facenet-pytorch neste projeto, prefira numpy<2."
            )
    except Exception:
        pass

    try:
        headless_version = importlib_metadata.version("opencv-python-headless")
        warnings.append(
            f"opencv-python-headless {headless_version} está instalado. Isso costuma desabilitar janelas do cv2 no Windows."
        )
    except importlib_metadata.PackageNotFoundError:
        pass

    if not opencv_highgui_available():
        warnings.append(
            "OpenCV atual está sem HighGUI. O aplicativo vai usar fallback de vídeo em Tkinter no lugar do cv2.imshow."
        )

    return warnings


def open_camera(config: AppConfig) -> cv2.VideoCapture:
    attempts = [
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("CAP_MSMF", cv2.CAP_MSMF),
        ("DEFAULT", None),
    ]
    last_error: Optional[str] = None

    for backend_name, backend in attempts:
        logging.info("Tentando abrir webcam com backend %s", backend_name)
        cap = cv2.VideoCapture(config.camera_index, backend) if backend is not None else cv2.VideoCapture(config.camera_index)

        if not cap or not cap.isOpened():
            last_error = f"Falha ao abrir câmera com backend {backend_name}"
            if cap:
                cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        ok, frame = cap.read()
        if not ok or frame is None:
            last_error = f"Câmera abriu com {backend_name}, mas não entregou frames válidos"
            cap.release()
            continue

        for _ in range(config.camera_warmup_frames):
            cap.read()

        logging.info("Webcam aberta com sucesso usando backend %s", backend_name)
        return cap

    raise RuntimeError(last_error or "Não foi possível abrir a webcam.")


class CryptoManager:
    def __init__(self, key_path: Path) -> None:
        self.key_path = key_path
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            return self.key_path.read_bytes().strip()

        key = Fernet.generate_key()
        self.key_path.write_bytes(key)
        logging.info("Nova chave Fernet criada em %s", self.key_path)
        return key

    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        payload = ensure_float32(embedding).tobytes()
        return self._fernet.encrypt(payload)

    def decrypt_embedding(self, payload: bytes) -> np.ndarray:
        raw = self._fernet.decrypt(payload)
        return np.frombuffer(raw, dtype=np.float32).copy()


class SecureFaceDB:
    def __init__(self, db_path: Path, crypto: CryptoManager) -> None:
        self.db_path = db_path
        self.crypto = crypto
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def _table_exists(self, table_name: str) -> bool:
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _columns(self, table_name: str) -> Dict[str, sqlite3.Row]:
        rows = self.conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {row["name"]: row for row in rows}

    def _create_users_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                registration_id TEXT UNIQUE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_registration_id ON users (registration_id) WHERE registration_id IS NOT NULL"
        )

    def _create_face_embeddings_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS face_embeddings (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                pose TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_face_embeddings_user_pose ON face_embeddings (user_id, pose)"
        )

    def _migrate_users_table(self) -> None:
        if not self._table_exists("users"):
            self._create_users_table()
            return

        columns = self._columns("users")
        required = {"user_id", "registration_id", "name", "created_at"}
        if required.issubset(columns):
            self.conn.execute(
                "UPDATE users SET registration_id = CAST(user_id AS TEXT) WHERE registration_id IS NULL OR TRIM(registration_id) = ''"
            )
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_registration_id ON users (registration_id) WHERE registration_id IS NOT NULL"
            )
            return

        if "registration_id" not in columns:
            self.conn.execute("ALTER TABLE users ADD COLUMN registration_id TEXT")
            self.conn.execute(
                "UPDATE users SET registration_id = CAST(user_id AS TEXT) WHERE registration_id IS NULL OR TRIM(registration_id) = ''"
            )
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_registration_id ON users (registration_id) WHERE registration_id IS NOT NULL"
            )
            columns = self._columns("users")

        if {"user_id", "name", "created_at", "registration_id"}.issubset(columns):
            return

        logging.info("Migrando tabela users para o formato atual")
        self.conn.execute("ALTER TABLE users RENAME TO users_legacy")
        self._create_users_table()

        legacy_columns = self._columns("users_legacy")
        id_expr = "user_id" if "user_id" in legacy_columns else ("id" if "id" in legacy_columns else "rowid")
        registration_expr = (
            "registration_id"
            if "registration_id" in legacy_columns
            else ("matricula" if "matricula" in legacy_columns else f"CAST({id_expr} AS TEXT)")
        )
        name_expr = "name" if "name" in legacy_columns else "'user_' || rowid"
        created_expr = "created_at" if "created_at" in legacy_columns else "CURRENT_TIMESTAMP"

        self.conn.execute(
            f"""
            INSERT OR IGNORE INTO users (user_id, registration_id, name, created_at)
            SELECT {id_expr}, {registration_expr}, {name_expr}, {created_expr}
            FROM users_legacy
            """
        )
        self.conn.execute("DROP TABLE users_legacy")
        self.conn.execute(
            "UPDATE users SET registration_id = CAST(user_id AS TEXT) WHERE registration_id IS NULL OR TRIM(registration_id) = ''"
        )
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_registration_id ON users (registration_id) WHERE registration_id IS NOT NULL"
        )

    def _migrate_face_embeddings_table(self) -> None:
        if not self._table_exists("face_embeddings"):
            self._create_face_embeddings_table()
            return

        columns = self._columns("face_embeddings")
        required = {"sample_id", "user_id", "pose", "embedding", "created_at"}
        if required.issubset(columns):
            return

        logging.info("Migrando tabela face_embeddings para o formato atual")
        self.conn.execute("ALTER TABLE face_embeddings RENAME TO face_embeddings_legacy")
        self._create_face_embeddings_table()

        legacy_columns = self._columns("face_embeddings_legacy")
        sample_expr = "sample_id" if "sample_id" in legacy_columns else ("id" if "id" in legacy_columns else "rowid")
        user_expr = "user_id" if "user_id" in legacy_columns else ("person_id" if "person_id" in legacy_columns else "NULL")
        pose_expr = "pose" if "pose" in legacy_columns else "'front'"
        embedding_expr = "embedding"
        created_expr = "created_at" if "created_at" in legacy_columns else "CURRENT_TIMESTAMP"

        self.conn.execute(
            f"""
            INSERT INTO face_embeddings (sample_id, user_id, pose, embedding, created_at)
            SELECT {sample_expr}, {user_expr}, {pose_expr}, {embedding_expr}, {created_expr}
            FROM face_embeddings_legacy
            WHERE {user_expr} IS NOT NULL
            """
        )
        self.conn.execute("DROP TABLE face_embeddings_legacy")

    def _migrate(self) -> None:
        with self.conn:
            self._migrate_users_table()
            self._migrate_face_embeddings_table()

    def get_user_by_registration(self, registration_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM users WHERE registration_id = ? COLLATE NOCASE",
            (registration_id.strip(),),
        ).fetchone()

    def get_user_by_name(self, name: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM users WHERE name = ?", (name.strip(),)).fetchone()

    def get_user_by_identifier(self, identifier: str) -> Optional[sqlite3.Row]:
        clean_identifier = identifier.strip()
        if not clean_identifier:
            return None

        row = self.get_user_by_registration(clean_identifier)
        if row:
            return row

        if clean_identifier.isdigit():
            row = self.conn.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (int(clean_identifier),),
            ).fetchone()
            if row:
                return row

        return self.conn.execute(
            "SELECT * FROM users WHERE name = ?",
            (clean_identifier,),
        ).fetchone()

    def list_users(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT
                user_id,
                COALESCE(registration_id, CAST(user_id AS TEXT)) AS registration_id,
                name,
                created_at
            FROM users
            ORDER BY name COLLATE NOCASE ASC, registration_id ASC
            """
        ).fetchall()

    def upsert_user(self, registration_id: str, name: str) -> int:
        clean_registration = registration_id.strip()
        clean_name = name.strip()
        existing = self.get_user_by_registration(clean_registration)
        if existing:
            if clean_name and clean_name != str(existing["name"]):
                with self.conn:
                    self.conn.execute(
                        "UPDATE users SET name = ? WHERE user_id = ?",
                        (clean_name, int(existing["user_id"])),
                    )
            return int(existing["user_id"])

        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO users (registration_id, name, created_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (clean_registration, clean_name),
            )
            return int(cursor.lastrowid)

    def delete_user(self, identifier: str) -> Optional[sqlite3.Row]:
        return self.delete_user_by_registration_id(identifier)

    def delete_user_by_registration_id(self, registration_id: str) -> Optional[sqlite3.Row]:
        user = self.get_user_by_registration(registration_id)
        if user is None:
            return None

        with self.conn:
            self.conn.execute("DELETE FROM face_embeddings WHERE user_id = ?", (int(user["user_id"]),))
            self.conn.execute("DELETE FROM users WHERE user_id = ?", (int(user["user_id"]),))
        return user

    def add_embedding(self, user_id: int, pose: str, embedding: np.ndarray) -> int:
        encrypted = self.crypto.encrypt_embedding(normalize_embedding(embedding))
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO face_embeddings (user_id, pose, embedding, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (user_id, pose, encrypted),
            )
            return int(cursor.lastrowid)

    def load_all_samples(self) -> List[FaceSample]:
        rows = self.conn.execute(
            """
            SELECT
                fe.sample_id,
                fe.user_id,
                COALESCE(u.registration_id, CAST(u.user_id AS TEXT)) AS registration_id,
                u.name AS user_name,
                fe.pose,
                fe.embedding,
                fe.created_at
            FROM face_embeddings fe
            INNER JOIN users u ON u.user_id = fe.user_id
            ORDER BY fe.created_at ASC
            """
        ).fetchall()

        samples: List[FaceSample] = []
        for row in rows:
            try:
                embedding = normalize_embedding(self.crypto.decrypt_embedding(row["embedding"]))
                if embedding.shape[0] != 512:
                    logging.warning("Ignorando sample_id=%s com tamanho de embedding invalido: %s", row["sample_id"], embedding.shape)
                    continue
                samples.append(
                    FaceSample(
                        sample_id=int(row["sample_id"]),
                        user_id=int(row["user_id"]),
                        registration_id=str(row["registration_id"]),
                        user_name=str(row["user_name"]),
                        pose=str(row["pose"]),
                        embedding=embedding,
                        created_at=str(row["created_at"]),
                    )
                )
            except Exception as exc:  # pragma: no cover - continua carregando as demais amostras
                logging.exception("Falha ao descriptografar sample_id=%s: %s", row["sample_id"], exc)

        return samples

    def close(self) -> None:
        self.conn.close()


class EmbeddingCache:
    def __init__(self, db: SecureFaceDB, ttl_seconds: float) -> None:
        self.db = db
        self.ttl_seconds = ttl_seconds
        self.last_refresh = 0.0
        self.by_user: Dict[int, List[FaceSample]] = {}

    def refresh(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self.last_refresh) < self.ttl_seconds:
            return

        grouped: Dict[int, List[FaceSample]] = defaultdict(list)
        for sample in self.db.load_all_samples():
            grouped[sample.user_id].append(sample)

        self.by_user = dict(grouped)
        self.last_refresh = now
        logging.info("Cache atualizado: %s usuário(s), %s amostra(s)", len(self.by_user), sum(len(v) for v in self.by_user.values()))

    @property
    def user_count(self) -> int:
        return len(self.by_user)

    @property
    def sample_count(self) -> int:
        return sum(len(samples) for samples in self.by_user.values())


class FaceEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            post_process=False,
            min_face_size=40,
            thresholds=[0.55, 0.70, 0.80],
        )
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.reference_landmarks = np.array(
            [
                [54.71, 73.85],
                [105.05, 73.57],
                [80.04, 102.48],
                [59.36, 131.95],
                [101.04, 131.72],
            ],
            dtype=np.float32,
        )
        logging.info("Modelos carregados em %s", self.device)

    def _select_best_face(
        self,
        boxes: np.ndarray,
        probs: np.ndarray,
        landmarks: np.ndarray,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        frame_area = float(frame_shape[0] * frame_shape[1])
        best_index = 0
        best_score = -1.0

        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = clamp_box(box, frame_shape)
            area_ratio = max((x2 - x1) * (y2 - y1), 1.0) / max(frame_area, 1.0)
            score = float(probs[index]) + (0.25 * area_ratio)
            if score > best_score:
                best_index = index
                best_score = score

        return boxes[best_index], float(probs[best_index]), landmarks[best_index]

    def _evaluate_quality(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        confidence: float,
    ) -> Tuple[bool, str, float, float]:
        x1, y1, x2, y2 = box
        frame_area = float(frame.shape[0] * frame.shape[1])
        box_area = max((x2 - x1) * (y2 - y1), 1.0)
        face_ratio = box_area / max(frame_area, 1.0)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, "Rosto inválido", face_ratio, 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if confidence < self.config.mtcnn_confidence:
            return False, "Centralize melhor o rosto", face_ratio, sharpness

        if face_ratio < self.config.hard_min_face_ratio:
            return False, "Aproxime o rosto", face_ratio, sharpness
        if face_ratio > self.config.hard_max_face_ratio:
            return False, "Afaste um pouco", face_ratio, sharpness
        if sharpness < self.config.hard_min_sharpness:
            return False, "Evite movimento brusco", face_ratio, sharpness

        warnings: List[str] = []
        if face_ratio < self.config.soft_min_face_ratio:
            warnings.append("aproxime um pouco")
        if face_ratio > self.config.soft_max_face_ratio:
            warnings.append("afaste um pouco")
        if sharpness < self.config.soft_min_sharpness:
            warnings.append("segure mais firme")

        feedback = " | ".join(warnings) if warnings else "Rosto válido"
        return True, feedback, face_ratio, sharpness

    def _align_face(self, rgb_frame: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
        matrix, _ = cv2.estimateAffinePartial2D(
            landmarks.astype(np.float32),
            self.reference_landmarks,
            method=cv2.LMEDS,
        )
        if matrix is None:
            return None

        aligned = cv2.warpAffine(
            rgb_frame,
            matrix,
            (self.config.alignment_size, self.config.alignment_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        if aligned.size == 0:
            return None
        return aligned

    def _embedding_from_face(self, aligned_rgb: np.ndarray) -> np.ndarray:
        face_tensor = torch.from_numpy(aligned_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        face_tensor = (face_tensor - 0.5) / 0.5
        face_tensor = face_tensor.to(self.device)

        with torch.no_grad():
            embedding = self.resnet(face_tensor).cpu().numpy()[0]
        return normalize_embedding(embedding)

    def analyze_frame(self, frame: np.ndarray) -> FaceDetection:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self.mtcnn.detect(rgb_frame, landmarks=True)

        if boxes is None or probs is None or landmarks is None or len(boxes) == 0:
            return FaceDetection(found=False, feedback="Procurando rosto")

        best_box, best_prob, best_landmarks = self._select_best_face(boxes, probs, landmarks, frame.shape)
        clamped_box = clamp_box(best_box, frame.shape)
        pose = estimate_pose(best_landmarks, self.config.pose_threshold)

        is_valid, feedback, face_ratio, sharpness = self._evaluate_quality(frame, clamped_box, best_prob)
        if not is_valid:
            return FaceDetection(
                found=True,
                box=clamped_box,
                confidence=best_prob,
                pose=pose,
                feedback=feedback,
                face_ratio=face_ratio,
                sharpness=sharpness,
                landmarks=best_landmarks,
            )

        aligned_face = self._align_face(rgb_frame, best_landmarks)
        if aligned_face is None:
            return FaceDetection(
                found=True,
                box=clamped_box,
                confidence=best_prob,
                pose=pose,
                feedback="Não foi possível alinhar o rosto",
                face_ratio=face_ratio,
                sharpness=sharpness,
                landmarks=best_landmarks,
            )

        embedding = self._embedding_from_face(aligned_face)
        return FaceDetection(
            found=True,
            box=clamped_box,
            confidence=best_prob,
            pose=pose,
            feedback=feedback,
            embedding=embedding,
            face_ratio=face_ratio,
            sharpness=sharpness,
            landmarks=best_landmarks,
        )

    def _pose_weight(self, detected_pose: str, sample_pose: str) -> float:
        if detected_pose == sample_pose:
            return 0.95
        if detected_pose == "front" or sample_pose == "front":
            return 1.00
        return 1.05

    def match_embedding(
        self,
        embedding: np.ndarray,
        detected_pose: str,
        grouped_samples: Dict[int, List[FaceSample]],
    ) -> Optional[MatchCandidate]:
        if not grouped_samples:
            return None

        best_candidate: Optional[MatchCandidate] = None
        for user_id, samples in grouped_samples.items():
            best_raw_distance = float("inf")
            best_effective_distance = float("inf")
            best_pose = "front"
            best_name = samples[0].user_name
            best_registration = samples[0].registration_id

            for sample in samples:
                raw_distance = float(np.linalg.norm(embedding - sample.embedding))
                effective_distance = raw_distance * self._pose_weight(detected_pose, sample.pose)
                if effective_distance < best_effective_distance:
                    best_raw_distance = raw_distance
                    best_effective_distance = effective_distance
                    best_pose = sample.pose

            candidate = MatchCandidate(
                user_id=user_id,
                registration_id=best_registration,
                user_name=best_name,
                distance=best_raw_distance,
                effective_distance=best_effective_distance,
                matched_pose=best_pose,
                is_within_candidate_threshold=best_effective_distance <= self.config.frame_candidate_threshold,
            )

            if best_candidate is None or candidate.effective_distance < best_candidate.effective_distance:
                best_candidate = candidate

        return best_candidate


class TemporalVoteWindow:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.votes: Deque[FrameVote] = deque(maxlen=config.temporal_window)

    def reset(self) -> None:
        self.votes.clear()

    def add_vote(self, vote: FrameVote) -> None:
        self.votes.append(vote)

    def progress(self) -> Tuple[int, int]:
        return len(self.votes), self.config.temporal_window

    def ready(self) -> bool:
        return len(self.votes) >= self.config.temporal_window

    def decide(self) -> TemporalDecision:
        total_frames = len(self.votes)
        if total_frames < self.config.min_valid_frames:
            return TemporalDecision(status="collecting", total_frames=total_frames, message="Coletando frames validos")

        matched_votes = [vote for vote in self.votes if vote.user_id is not None]
        all_best_distances = [vote.best_distance for vote in self.votes if np.isfinite(vote.best_distance)]

        if not matched_votes:
            mean_distance = float(np.mean(all_best_distances)) if all_best_distances else float("inf")
            best_distance = float(np.min(all_best_distances)) if all_best_distances else float("inf")
            if mean_distance >= self.config.unknown_distance_threshold or not all_best_distances:
                return TemporalDecision(
                    status="unknown",
                    total_frames=total_frames,
                    mean_distance=mean_distance,
                    best_distance=best_distance,
                    message="Nova face detectada",
                )
            return TemporalDecision(
                status="uncertain",
                total_frames=total_frames,
                mean_distance=mean_distance,
                best_distance=best_distance,
                message="Baixa confianca",
            )

        counter = Counter(vote.user_id for vote in matched_votes if vote.user_id is not None)
        winner_id, winner_votes = counter.most_common(1)[0]
        winner_frames = [vote for vote in matched_votes if vote.user_id == winner_id]
        mean_distance = float(np.mean([vote.distance for vote in winner_frames]))
        best_distance = float(np.min([vote.distance for vote in winner_frames]))
        vote_ratio = winner_votes / max(total_frames, 1)
        user_name = winner_frames[0].user_name
        registration_id = winner_frames[0].registration_id

        if (
            total_frames >= self.config.min_valid_frames
            and winner_votes >= self.config.min_confirm_votes
            and vote_ratio >= self.config.min_vote_ratio
            and mean_distance <= self.config.final_distance_threshold
            and best_distance <= (self.config.final_distance_threshold + self.config.distance_margin)
        ):
            return TemporalDecision(
                status="confirmed",
                user_id=winner_id,
                registration_id=registration_id,
                user_name=user_name,
                votes=winner_votes,
                total_frames=total_frames,
                vote_ratio=vote_ratio,
                mean_distance=mean_distance,
                best_distance=best_distance,
                message=f"Encontrado: {user_name}",
            )

        mean_best_distance = float(np.mean(all_best_distances)) if all_best_distances else float("inf")
        if mean_best_distance >= self.config.unknown_distance_threshold:
            return TemporalDecision(
                status="unknown",
                total_frames=total_frames,
                mean_distance=mean_best_distance,
                best_distance=float(np.min(all_best_distances)) if all_best_distances else float("inf"),
                message="Nova face detectada",
            )

        return TemporalDecision(
            status="uncertain",
            user_id=winner_id,
            registration_id=registration_id,
            user_name=user_name,
            votes=winner_votes,
            total_frames=total_frames,
            vote_ratio=vote_ratio,
            mean_distance=mean_distance,
            best_distance=best_distance,
            message="Ainda sem consenso suficiente",
        )


class BotInterface:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.attributes("-topmost", True)

    def ask_user_name(self) -> Optional[str]:
        self.root.update()
        return simpledialog.askstring(
            "Novo cadastro",
            "Digite o nome do usuário:",
            parent=self.root,
        )

    def confirm_extend_registration(self, name: str) -> bool:
        self.root.update()
        return messagebox.askyesno(
            "Usuário já existe",
            f'O usuário "{name}" já existe. Deseja adicionar novas amostras?',
            parent=self.root,
        )

    def show_pose_instruction(self, pose: str, samples_per_pose: int) -> None:
        self.root.update()
        messagebox.showinfo(
            "Cadastro guiado",
            f"Pose atual: {POSE_LABELS[pose]}.\n"
            f"Capture {samples_per_pose} frames válidos mantendo essa pose.\n"
            "Use ESC durante a captura para cancelar.",
            parent=self.root,
        )

    def show_info(self, title: str, message: str) -> None:
        self.root.update()
        messagebox.showinfo(title, message, parent=self.root)

    def show_error(self, title: str, message: str) -> None:
        self.root.update()
        messagebox.showerror(title, message, parent=self.root)

    def close(self) -> None:
        try:
            self.root.update()
            self.root.destroy()
        except tk.TclError:
            pass


class BaseVideoDisplay:
    def show(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    def poll_key(self, delay_ms: int = 1) -> Optional[int]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class OpenCVVideoDisplay(BaseVideoDisplay):
    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self.window_name, frame)

    def poll_key(self, delay_ms: int = 1) -> Optional[int]:
        key = cv2.waitKey(delay_ms) & 0xFF
        return None if key == 255 else key

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass


class TkinterVideoDisplay(BaseVideoDisplay):
    def __init__(self, root: tk.Tk, window_name: str) -> None:
        self.root = root
        self.window_name = window_name
        self.window = tk.Toplevel(self.root)
        self.window.title(window_name)
        self.window.geometry("1280x760")
        self.window.configure(bg="black")
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self.window.bind("<KeyPress>", self._on_key_press)
        self.window.bind("<Escape>", self._on_escape)
        self.window.bind("<FocusIn>", self._on_focus)
        self.image_label = tk.Label(self.window, bg="black")
        self.image_label.pack(fill="both", expand=True)
        self.pending_keys: Deque[int] = deque()
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.closed = False
        self.window.after(100, self._focus_window)

    def _focus_window(self) -> None:
        if self.closed:
            return
        try:
            self.window.lift()
            self.window.focus_force()
        except tk.TclError:
            self.closed = True

    def _on_focus(self, _event: tk.Event) -> None:
        self._focus_window()

    def _on_key_press(self, event: tk.Event) -> None:
        if event.char:
            self.pending_keys.append(ord(event.char.lower()))

    def _on_escape(self, _event: tk.Event) -> None:
        self.pending_keys.append(27)

    def _on_close(self) -> None:
        self.closed = True

    def show(self, frame: np.ndarray) -> None:
        if self.closed:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        self.photo_image = ImageTk.PhotoImage(image=image)
        self.image_label.configure(image=self.photo_image)
        self.image_label.image = self.photo_image
        self._pump()

    def _pump(self) -> None:
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def poll_key(self, delay_ms: int = 1) -> Optional[int]:
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        self._pump()

        if self.closed:
            return ord("q")
        if self.pending_keys:
            return self.pending_keys.popleft()
        return None

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.window.destroy()
        except tk.TclError:
            pass


class FaceRecognitionApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        self.crypto = CryptoManager(config.key_path)
        self.db = SecureFaceDB(config.db_path, self.crypto)
        self.cache = EmbeddingCache(self.db, ttl_seconds=config.cache_ttl_seconds)
        self.cache.refresh(force=True)
        self.engine = FaceEngine(config)
        self.ui = BotInterface()
        self.temporal_window = TemporalVoteWindow(config)
        self.display: Optional[BaseVideoDisplay] = None

        self.status_message = "Procurando rosto"
        self.status_color = (0, 255, 255)
        self.status_until = 0.0
        self.last_face_seen_at = 0.0

    def _create_display(self) -> BaseVideoDisplay:
        if opencv_highgui_available():
            try:
                logging.info("Usando janela OpenCV para exibição do vídeo")
                return OpenCVVideoDisplay(self.config.window_name)
            except cv2.error as exc:
                logging.warning("Falha ao abrir janela OpenCV, usando fallback Tkinter: %s", exc)

        logging.warning("Fallback de vídeo ativado: janela Tkinter")
        return TkinterVideoDisplay(self.ui.root, self.config.window_name)

    def _show_frame(self, frame: np.ndarray) -> None:
        if self.display is None:
            raise RuntimeError("Display ainda não foi inicializado.")
        self.display.show(frame)

    def _poll_key(self, delay_ms: int = 1) -> Optional[int]:
        if self.display is None:
            return None
        return self.display.poll_key(delay_ms=delay_ms)

    def set_status(self, message: str, color: Tuple[int, int, int], duration: float = 1.5) -> None:
        self.status_message = message
        self.status_color = color
        self.status_until = time.time() + duration

    def _active_status(self) -> Tuple[str, Tuple[int, int, int]]:
        if time.time() <= self.status_until:
            return self.status_message, self.status_color
        return "Procurando rosto", (0, 255, 255)

    def _draw_text_block(
        self,
        frame: np.ndarray,
        lines: List[str],
        color: Tuple[int, int, int],
        start_y: int = 24,
    ) -> None:
        if not lines:
            return

        y = start_y
        for line in lines:
            safe_line = text_for_opencv(line)
            (text_width, text_height), _ = cv2.getTextSize(safe_line, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
            cv2.rectangle(frame, (10, y - 18), (20 + text_width, y + 8), (0, 0, 0), -1)
            cv2.putText(frame, safe_line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
            y += 28

    def _draw_detection_overlay(self, frame: np.ndarray, detection: FaceDetection, extra_lines: List[str]) -> None:
        if detection.found and detection.box is not None:
            x1, y1, x2, y2 = detection.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                frame,
                text_for_opencv(f"pose: {POSE_LABELS[detection.pose]}"),
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )

        status_message, status_color = self._active_status()
        lines = [
            status_message,
            f"Pose detectada: {POSE_LABELS[detection.pose]}" if detection.found else "Pose detectada: aguardando",
        ]
        lines.extend(extra_lines)
        lines.append("Atalhos: N=cadastrar nova face | C=limpar janela | Q/ESC=sair")
        self._draw_text_block(frame, lines, status_color)

    def _make_vote(self, candidate: Optional[MatchCandidate], detected_pose: str) -> FrameVote:
        if candidate is None:
            return FrameVote(
                user_id=None,
                registration_id=None,
                user_name=None,
                distance=float("inf"),
                best_distance=float("inf"),
                detected_pose=detected_pose,
                matched_pose=None,
                timestamp=time.time(),
            )

        return FrameVote(
            user_id=candidate.user_id if candidate.is_within_candidate_threshold else None,
            registration_id=candidate.registration_id if candidate.is_within_candidate_threshold else None,
            user_name=candidate.user_name if candidate.is_within_candidate_threshold else None,
            distance=candidate.distance,
            best_distance=candidate.effective_distance,
            detected_pose=detected_pose,
            matched_pose=candidate.matched_pose,
            timestamp=time.time(),
        )

    def _handle_decision(self, decision: TemporalDecision) -> None:
        if decision.status == "confirmed" and decision.user_name:
            self.set_status(
                f"Encontrado: {decision.user_name} | votos={decision.votes}/{decision.total_frames} | dist={decision.mean_distance:.3f}",
                (0, 255, 0),
                duration=3.0,
            )
            return

        if decision.status == "unknown":
            self.set_status("Nova face detectada. Pressione N para cadastrar.", (0, 165, 255), duration=3.0)
            return

        self.set_status(
            "Reconhecimento inconclusivo. Continue olhando para a câmera.",
            (0, 255, 255),
            duration=2.0,
        )

    def _capture_pose_samples(self, cap: cv2.VideoCapture, target_pose: str) -> Optional[List[np.ndarray]]:
        self.ui.show_pose_instruction(target_pose, self.config.samples_per_pose)
        collected: List[np.ndarray] = []
        started_at = time.time()
        last_capture_at = 0.0

        while len(collected) < self.config.samples_per_pose:
            ok, frame = cap.read()
            if not ok or frame is None:
                self.set_status("Falha ao ler frame da webcam", (0, 0, 255), duration=2.0)
                continue

            detection = self.engine.analyze_frame(frame)
            extra_lines = [
                f"Cadastro: {POSE_LABELS[target_pose]}",
                f"Amostras: {len(collected)}/{self.config.samples_per_pose}",
            ]

            if not detection.found:
                extra_lines.append("Procure ficar centralizado na câmera")
            elif detection.pose != target_pose:
                extra_lines.append(f"Pose atual: {POSE_LABELS[detection.pose]}")
                extra_lines.append(f"Ajuste para: {POSE_LABELS[target_pose]}")
            elif detection.embedding is None:
                extra_lines.append(detection.feedback)
            else:
                now = time.time()
                if now - last_capture_at >= self.config.sample_capture_interval:
                    collected.append(detection.embedding)
                    last_capture_at = now
                    extra_lines.append(f"Frame válido capturado: {len(collected)}")
                else:
                    extra_lines.append("Mantendo pose para a próxima captura")

            remaining = max(self.config.pose_capture_timeout - (time.time() - started_at), 0.0)
            extra_lines.append(f"Tempo restante: {remaining:0.1f}s")
            self._draw_detection_overlay(frame, detection, extra_lines)
            self._show_frame(frame)

            key = self._poll_key(1)
            if key in (27, ord("q")):
                return None

            if (time.time() - started_at) >= self.config.pose_capture_timeout:
                self.set_status("Tempo de captura esgotado para esta pose", (0, 165, 255), duration=2.5)
                return None

        return collected

    def register_new_user(self, cap: cv2.VideoCapture) -> None:
        name = self.ui.ask_user_name()
        if name is None:
            self.set_status("Cadastro cancelado", (0, 165, 255), duration=1.5)
            return

        name = name.strip()
        if not name:
            self.ui.show_error("Nome inválido", "O nome do usuário não pode ficar vazio.")
            return

        existing_user = self.db.get_user_by_name(name)
        if existing_user and not self.ui.confirm_extend_registration(name):
            self.set_status("Cadastro cancelado", (0, 165, 255), duration=1.5)
            return

        captured_by_pose: Dict[str, np.ndarray] = {}
        for pose in ("front", "left", "right"):
            samples = self._capture_pose_samples(cap, pose)
            if not samples:
                self.ui.show_error("Cadastro interrompido", f"Não foi possível concluir a pose {POSE_LABELS[pose]}.")
                return
            captured_by_pose[pose] = average_embeddings(samples)

        user_id = self.db.upsert_user(name, name)
        for pose, aggregated_embedding in captured_by_pose.items():
            self.db.add_embedding(user_id, pose, aggregated_embedding)

        self.cache.refresh(force=True)
        self.temporal_window.reset()
        self.set_status(f"Cadastro salvo para {name}", (0, 255, 0), duration=3.0)
        self.ui.show_info("Cadastro concluído", f"Usuário {name} cadastrado com 3 amostras multi-pose.")

    def run(self) -> None:
        cap = open_camera(self.config)
        self.display = self._create_display()

        try:
            while True:
                self.cache.refresh()
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.set_status("Erro ao ler webcam", (0, 0, 255), duration=2.0)
                    blank = np.zeros((self.config.camera_height, self.config.camera_width, 3), dtype=np.uint8)
                    self._draw_text_block(blank, ["Erro ao ler webcam"], (0, 0, 255))
                    self._show_frame(blank)
                    if self._poll_key(30) in (27, ord("q")):
                        break
                    continue

                detection = self.engine.analyze_frame(frame)
                extra_lines: List[str] = []

                if detection.found:
                    self.last_face_seen_at = time.time()
                    extra_lines.append(f"Confiança MTCNN: {detection.confidence:.3f}")
                    extra_lines.append(f"Tamanho do rosto: {detection.face_ratio:.3f}")
                    extra_lines.append(f"Nitidez: {detection.sharpness:.1f}")

                    if detection.embedding is None:
                        extra_lines.append(detection.feedback)
                    else:
                        candidate = self.engine.match_embedding(detection.embedding, detection.pose, self.cache.by_user)
                        vote = self._make_vote(candidate, detection.pose)
                        self.temporal_window.add_vote(vote)
                        current_votes, target_votes = self.temporal_window.progress()
                        extra_lines.append(f"Reconhecendo: {current_votes}/{target_votes} frames")

                        if candidate is not None:
                            extra_lines.append(
                                f"Melhor candidato: {candidate.user_name} | dist={candidate.distance:.3f} | pose={POSE_LABELS[candidate.matched_pose]}"
                            )
                        else:
                            extra_lines.append("Base vazia. Pressione N para cadastrar o primeiro usuário.")

                        if self.temporal_window.ready():
                            decision = self.temporal_window.decide()
                            self._handle_decision(decision)
                            self.temporal_window.reset()
                else:
                    extra_lines.append(detection.feedback)
                    if (time.time() - self.last_face_seen_at) > 1.2:
                        self.temporal_window.reset()

                self._draw_detection_overlay(frame, detection, extra_lines)
                self._show_frame(frame)

                key = self._poll_key(1)
                if key in (27, ord("q")):
                    break
                if key == ord("n"):
                    self.register_new_user(cap)
                if key == ord("c"):
                    self.temporal_window.reset()
                    self.set_status("Janela temporal reiniciada", (0, 255, 255), duration=1.5)

        finally:
            cap.release()
            if self.display is not None:
                self.display.close()
            cv2.destroyAllWindows()
            self.db.close()
            self.ui.close()


@dataclass(slots=True)
class RegistrationSession:
    registration_id: str
    name: str
    poses: Tuple[str, ...] = ("front", "left", "right")
    pose_index: int = 0
    current_samples: List[np.ndarray] = field(default_factory=list)
    captured_by_pose: Dict[str, np.ndarray] = field(default_factory=dict)
    pose_started_at: float = field(default_factory=time.time)
    last_capture_at: float = 0.0

    @property
    def current_pose(self) -> str:
        return self.poses[self.pose_index]

    def reset_current_pose(self) -> None:
        self.current_samples.clear()
        self.pose_started_at = time.time()
        self.last_capture_at = 0.0


class CameraManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.backend_name = "desconhecido"

    def open(self) -> None:
        self.cap = open_camera(self.config)

    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_open or self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        return ok, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VirtualKeyboardTyper:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.last_typed_identifier: Optional[str] = None
        self.last_typed_at = 0.0
        self.consecutive_identifier: Optional[str] = None
        self.consecutive_count = 0
        self.available = False
        self.error_message = ""
        self.controller = None
        self.enter_key = None

        try:
            from pynput.keyboard import Controller, Key

            self.controller = Controller()
            self.enter_key = Key.enter
            self.available = True
        except Exception as exc:  # pragma: no cover - depende do SO/sessao grafica
            self.error_message = str(exc)
            logging.exception("Falha ao inicializar pynput: %s", exc)

    def type_identifier(self, identifier: str, enabled: bool) -> Tuple[bool, str]:
        clean_identifier = identifier.strip()
        allowed, blocked_message = self.can_type_identifier(clean_identifier, enabled)
        if not allowed:
            return False, blocked_message

        if clean_identifier != self.consecutive_identifier:
            self.consecutive_identifier = clean_identifier
            self.consecutive_count = 0

        self.controller.type(clean_identifier)
        if self.config.press_enter_after_typing and self.enter_key is not None:
            self.controller.press(self.enter_key)
            self.controller.release(self.enter_key)

        self.last_typed_identifier = clean_identifier
        self.last_typed_at = time.time()
        self.consecutive_count += 1
        return True, f"Código do cartão digitado: {clean_identifier}"

    def can_type_identifier(self, identifier: str, enabled: bool) -> Tuple[bool, str]:
        clean_identifier = identifier.strip()
        if not enabled:
            return False, "digitacao desativada"
        if not self.config.type_recognized_id:
            return False, "digitacao desativada por configuracao"
        if not self.available or self.controller is None:
            return False, f"pynput indisponivel: {self.error_message}"
        if not clean_identifier:
            return False, "código do cartão vazio"

        consecutive_count = self.consecutive_count if clean_identifier == self.consecutive_identifier else 0
        if consecutive_count >= self.config.max_consecutive_typings_per_id:
            return False, f"limite de digitação atingido para {clean_identifier}"

        now = time.time()
        if (
            clean_identifier == self.last_typed_identifier
            and (now - self.last_typed_at) < self.config.type_cooldown_seconds
        ):
            return False, "aguardando cooldown"

        return True, "pronto para digitar"

    def reset_consecutive_state(self) -> None:
        self.consecutive_identifier = None
        self.consecutive_count = 0

    def reset_typing_session(self) -> None:
        self.reset_consecutive_state()
        self.last_typed_identifier = None
        self.last_typed_at = 0.0


class MainApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        self.crypto = CryptoManager(config.key_path)
        self.db = SecureFaceDB(config.db_path, self.crypto)
        self.cache = EmbeddingCache(self.db, ttl_seconds=config.cache_ttl_seconds)
        self.cache.refresh(force=True)
        self.engine = FaceEngine(config)
        self.camera = CameraManager(config)
        self.temporal_window = TemporalVoteWindow(config)
        self.keyboard_typer = VirtualKeyboardTyper(config)

        self.root = tk.Tk()
        self.root.title("Sistema de Reconhecimento Facial")
        self.root.geometry("1120x720")
        self.root.minsize(self.config.min_window_width, self.config.min_window_height)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.running = False
        self.mode = "recognition"
        self.registration_session: Optional[RegistrationSession] = None
        self.video_image: Optional[ImageTk.PhotoImage] = None
        self.sidebar_visible = True
        self.delete_window: Optional[tk.Toplevel] = None
        self.typing_prompt_window: Optional[tk.Toplevel] = None
        self.pending_typing_code: Optional[str] = None
        self.pending_mouse_listener = None
        self.pending_typing_due_at = 0.0
        self.last_face_seen_at = 0.0
        self.status_message = "Inicializando"
        self.status_color = (0, 220, 255)
        self.status_until = 0.0

        self.typing_enabled_var = tk.BooleanVar(value=config.type_recognized_id and self.keyboard_typer.available)
        self.enter_typing_enabled_var = tk.BooleanVar(value=config.press_enter_after_typing and self.typing_enabled_var.get())
        self.status_vars: Dict[str, tk.StringVar] = {}
        self._build_layout()
        self._refresh_database_status()
        self._update_typing_status()

    def _build_layout(self) -> None:
        self._configure_styles()
        self.root.configure(bg="#f3f6f8")
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="Header.TFrame", padding=(14, 10))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Sistema de Reconhecimento Facial", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Modulo unico: reconhecimento facial local com webcam", style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.header_status_var = tk.StringVar(value="Status: inicializando")
        ttk.Label(header, textvariable=self.header_status_var, style="HeaderStatus.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        self.camera_frame = tk.Frame(self.root, bg="#05070a", highlightthickness=0)
        self.camera_frame.grid(row=1, column=0, sticky="nsew")
        self.camera_frame.rowconfigure(0, weight=1)
        self.camera_frame.columnconfigure(0, weight=1)

        self.video_label = tk.Label(self.camera_frame, bg="#05070a", bd=0)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        self.sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", padding=(10, 10), width=self.config.sidebar_width)
        self.sidebar.grid(row=1, column=1, sticky="ns")
        self.sidebar.grid_propagate(False)
        self.sidebar.columnconfigure(0, weight=1)

        ttk.Label(self.sidebar, text="Ações do operador", style="Section.TLabel").grid(row=0, column=0, sticky="ew")
        self.register_button = self._make_sidebar_button(
            "Cadastrar pessoa",
            self.start_registration,
            self.config.register_button_color,
        )
        self.register_button.grid(row=1, column=0, sticky="ew", pady=(10, 6))
        self.delete_button = self._make_sidebar_button(
            "Excluir pessoa",
            self.delete_user,
            self.config.delete_button_color,
        )
        self.delete_button.grid(row=2, column=0, sticky="ew", pady=6)
        self.typing_check = tk.Checkbutton(
            self.sidebar,
            text="Perguntar antes de digitar o código do cartão",
            variable=self.typing_enabled_var,
            command=self._update_typing_status,
            bg=self.config.sidebar_color,
            fg="#ffffff",
            activebackground=self.config.sidebar_color,
            activeforeground="#ffffff",
            selectcolor=self.config.sidebar_color,
            font=("Segoe UI", 9),
            anchor="w",
            relief="flat",
            highlightthickness=0,
        )
        self.typing_check.grid(row=3, column=0, sticky="w", pady=6)
        self.enter_typing_check = tk.Checkbutton(
            self.sidebar,
            text="Digitar Enter automaticamente",
            variable=self.enter_typing_enabled_var,
            command=self._update_typing_status,
            bg=self.config.sidebar_color,
            fg="#ffffff",
            activebackground=self.config.sidebar_color,
            activeforeground="#ffffff",
            selectcolor=self.config.sidebar_color,
            font=("Segoe UI", 9),
            anchor="w",
            relief="flat",
            highlightthickness=0,
        )
        self.enter_typing_check.grid(row=4, column=0, sticky="w", pady=(0, 6))
        self.close_button = self._make_sidebar_button("Fechar app", self.close, self.config.close_button_color)
        self.close_button.grid(row=5, column=0, sticky="ew", pady=(6, 14))

        tk.Frame(self.sidebar, bg="#415061", height=1).grid(row=6, column=0, sticky="ew", pady=(2, 12))
        ttk.Label(self.sidebar, text="Status", style="Section.TLabel").grid(row=7, column=0, sticky="ew")

        status_frame = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        status_frame.grid(row=8, column=0, sticky="nsew", pady=(8, 0))
        status_frame.columnconfigure(1, weight=1)
        self.sidebar.rowconfigure(8, weight=1)

        rows = [
            ("camera", "Câmera"),
            ("database", "Banco"),
            ("recognized_user", "Usuário"),
            ("recognized_id", "Código do cartão"),
            ("pose", "Pose"),
            ("distance", "Distância média"),
            ("frames", "Frames válidos"),
            ("typing", "Digitação"),
            ("operator", "Operação"),
        ]
        for index, (key, label) in enumerate(rows):
            self._add_status_row(status_frame, index, key, label)

        self.root.bind("<Configure>", self._on_root_resize)
        self.root.bind("<Escape>", lambda _event: self.cancel_registration() if self.mode == "registering" else self.close())

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Header.TFrame", background=self.config.header_color)
        style.configure("Sidebar.TFrame", background=self.config.sidebar_color)
        style.configure("Title.TLabel", background=self.config.header_color, foreground="#ffffff", font=("Segoe UI", 17, "bold"))
        style.configure("Subtitle.TLabel", background=self.config.header_color, foreground="#d9e2ec", font=("Segoe UI", 9))
        style.configure("HeaderStatus.TLabel", background=self.config.header_color, foreground="#b6f0c2", font=("Segoe UI", 9, "bold"))
        style.configure("Section.TLabel", background=self.config.sidebar_color, foreground="#ffffff", font=("Segoe UI", 11, "bold"))
        style.configure("StatusName.TLabel", background=self.config.sidebar_color, foreground="#d9e2ec", font=("Segoe UI", 9))
        style.configure("StatusValue.TLabel", background=self.config.sidebar_color, foreground="#ffffff", font=("Segoe UI", 9, "bold"), wraplength=135)
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7))
        style.configure("TCheckbutton", background=self.config.sidebar_color, foreground="#ffffff", font=("Segoe UI", 9))

    def _make_sidebar_button(self, text: str, command, background: str) -> tk.Button:
        return tk.Button(
            self.sidebar,
            text=text,
            command=command,
            bg=background,
            fg=self.config.button_text_color,
            activebackground=background,
            activeforeground=self.config.button_text_color,
            relief="flat",
            bd=0,
            padx=8,
            pady=8,
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            highlightthickness=0,
        )

    def _add_status_row(self, parent: ttk.Frame, row: int, key: str, label: str) -> None:
        self.status_vars[key] = tk.StringVar(value="-")
        ttk.Label(parent, text=label, style="StatusName.TLabel").grid(row=row, column=0, sticky="nw", padx=(0, 6), pady=4)
        ttk.Label(parent, textvariable=self.status_vars[key], style="StatusValue.TLabel").grid(row=row, column=1, sticky="ew", pady=4)

    def _on_root_resize(self, event: tk.Event) -> None:
        if event.widget is not self.root:
            return
        if event.width < self.config.sidebar_hide_width and self.sidebar_visible:
            self.sidebar.grid_remove()
            self.sidebar_visible = False
        elif event.width >= self.config.sidebar_show_width and not self.sidebar_visible:
            self.sidebar.grid()
            self.sidebar_visible = True

    def start(self) -> None:
        self.running = True
        try:
            self.camera.open()
            self.status_vars["camera"].set("ativa")
            self._set_status("Camera ativa / banco conectado / reconhecendo", (0, 220, 0), duration=2.0)
        except Exception as exc:
            self.status_vars["camera"].set("indisponivel")
            self._set_status("Camera indisponivel", (0, 0, 255), duration=5.0)
            messagebox.showerror("Erro de camera", f"Nao foi possivel abrir a webcam.\n\n{exc}", parent=self.root)

        self.root.after(10, self._video_loop)
        self.root.mainloop()

    def _video_loop(self) -> None:
        if not self.running:
            return

        self._process_due_pending_typing()

        if not self.camera.is_open:
            frame = self._blank_frame("Camera indisponivel")
            detection = FaceDetection(found=False, feedback="Camera indisponivel")
            self._draw_detection_overlay(frame, detection, ["Verifique se a webcam esta conectada"])
            self._show_frame(frame)
            self.root.after(250, self._video_loop)
            return

        self.cache.refresh()
        ok, frame = self.camera.read()
        if not ok or frame is None:
            self.status_vars["camera"].set("erro de leitura")
            frame = self._blank_frame("Erro ao ler webcam")
            detection = FaceDetection(found=False, feedback="Erro ao ler webcam")
            self._draw_detection_overlay(frame, detection, ["Tentando ler o proximo frame"])
            self._show_frame(frame)
            self.root.after(80, self._video_loop)
            return

        detection = self.engine.analyze_frame(frame)
        extra_lines: List[str] = []
        self._update_detection_status(detection)

        if self.mode == "registering":
            self._process_registration_frame(detection, extra_lines)
        else:
            self._process_recognition_frame(detection, extra_lines)

        self._draw_detection_overlay(frame, detection, extra_lines)
        self._show_frame(frame)
        self.root.after(10, self._video_loop)

    def _process_recognition_frame(self, detection: FaceDetection, extra_lines: List[str]) -> None:
        self.status_vars["operator"].set("reconhecendo")
        if detection.found:
            self.last_face_seen_at = time.time()
            extra_lines.append(detection.feedback)

            if detection.embedding is None:
                self._reset_typing_when_waiting()
                extra_lines.append("Aguardando frame melhor")
                return

            candidate = self.engine.match_embedding(detection.embedding, detection.pose, self.cache.by_user)
            vote = self._make_vote(candidate, detection.pose)
            self.temporal_window.add_vote(vote)
            current_votes, target_votes = self.temporal_window.progress()
            self.status_vars["frames"].set(f"{current_votes}/{target_votes}")
            extra_lines.append(f"Reconhecendo: {current_votes}/{target_votes} frames")

            if candidate is None:
                extra_lines.append("Base vazia. Cadastre a primeira pessoa.")
            else:
                label = "candidato" if candidate.is_within_candidate_threshold else "baixo consenso"
                extra_lines.append(
                    f"{label}: código {candidate.registration_id} | dist={candidate.distance:.3f} | pose={POSE_LABELS[candidate.matched_pose]}"
                )

            if self.temporal_window.ready():
                decision = self.temporal_window.decide()
                self._handle_decision(decision)
                self.temporal_window.reset()
            return

        extra_lines.append(detection.feedback)
        self._reset_typing_when_waiting()
        if (time.time() - self.last_face_seen_at) > 1.2:
            self.temporal_window.reset()
            self.status_vars["frames"].set("0/{0}".format(self.config.temporal_window))

    def _reset_typing_when_waiting(self) -> None:
        self.keyboard_typer.reset_typing_session()

    def _make_vote(self, candidate: Optional[MatchCandidate], detected_pose: str) -> FrameVote:
        if candidate is None:
            return FrameVote(
                user_id=None,
                registration_id=None,
                user_name=None,
                distance=float("inf"),
                best_distance=float("inf"),
                detected_pose=detected_pose,
                matched_pose=None,
                timestamp=time.time(),
            )

        accepted = candidate.is_within_candidate_threshold
        return FrameVote(
            user_id=candidate.user_id if accepted else None,
            registration_id=candidate.registration_id if accepted else None,
            user_name=candidate.user_name if accepted else None,
            distance=candidate.distance,
            best_distance=candidate.effective_distance,
            detected_pose=detected_pose,
            matched_pose=candidate.matched_pose,
            timestamp=time.time(),
        )

    def _handle_decision(self, decision: TemporalDecision) -> None:
        if decision.status == "confirmed" and decision.registration_id:
            self.status_vars["recognized_user"].set(decision.user_name or "-")
            self.status_vars["recognized_id"].set(decision.registration_id)
            self.status_vars["distance"].set(f"{decision.mean_distance:.3f}")
            self.status_vars["frames"].set(f"{decision.votes}/{decision.total_frames}")
            self._set_status(
                f"Encontrado: {decision.user_name} ({decision.registration_id})",
                (0, 220, 0),
                duration=3.0,
            )
            self._handle_card_code_typing_request(decision.registration_id, decision.user_name or "")
            return

        if decision.status == "unknown":
            self.status_vars["recognized_user"].set("nova face")
            self.status_vars["recognized_id"].set("-")
            self.status_vars["distance"].set(self._format_distance(decision.mean_distance))
            self._set_status("Nova face detectada", (0, 165, 255), duration=3.0)
            return

        self.status_vars["distance"].set(self._format_distance(decision.mean_distance))
        self._set_status("Baixa confianca. Continue olhando para a camera.", (0, 220, 255), duration=2.0)

    def _handle_card_code_typing_request(self, card_code: str, user_name: str) -> None:
        enabled = self.config.type_recognized_id and self.mode == "recognition"
        allowed, typing_message = self.keyboard_typer.can_type_identifier(card_code, enabled=enabled)
        if not allowed:
            self.status_vars["typing"].set(typing_message)
            return

        if not self.typing_enabled_var.get():
            typed, typing_message = self.keyboard_typer.type_identifier(card_code, enabled=enabled)
            self.status_vars["typing"].set(typing_message)
            if typed:
                logging.info("Código do cartão digitado automaticamente via pynput: %s", card_code)
            return

        if self.pending_typing_code == card_code and (
            self.typing_prompt_window is not None or self.pending_mouse_listener is not None
        ):
            return

        self._open_typing_confirmation_popup(card_code, user_name)

    def _open_typing_confirmation_popup(self, card_code: str, user_name: str) -> None:
        self._cancel_pending_typing_prompt()
        self.pending_typing_code = card_code

        self.typing_prompt_window = tk.Toplevel(self.root)
        self.typing_prompt_window.title("Confirmar código do cartão")
        self.typing_prompt_window.geometry("430x210")
        self.typing_prompt_window.resizable(False, False)
        self.typing_prompt_window.configure(bg=self.config.sidebar_color)
        self.typing_prompt_window.attributes("-topmost", True)
        self.typing_prompt_window.columnconfigure(0, weight=1)

        tk.Label(
            self.typing_prompt_window,
            text="Código do cartão reconhecido",
            bg=self.config.sidebar_color,
            fg="#ffffff",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 6))

        tk.Label(
            self.typing_prompt_window,
            text=f"{user_name}\nCódigo: {card_code}",
            bg=self.config.sidebar_color,
            fg="#d9e2ec",
            font=("Segoe UI", 10),
            justify="center",
        ).grid(row=1, column=0, sticky="ew", padx=16)

        tk.Label(
            self.typing_prompt_window,
            text="Confirme e depois clique no campo onde o código deve ser digitado.",
            bg=self.config.sidebar_color,
            fg="#ffffff",
            font=("Segoe UI", 9),
            wraplength=370,
            justify="center",
        ).grid(row=2, column=0, sticky="ew", padx=16, pady=(10, 12))

        button_frame = tk.Frame(self.typing_prompt_window, bg=self.config.sidebar_color)
        button_frame.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 16))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        tk.Button(
            button_frame,
            text="Confirmar",
            command=lambda: self._arm_typing_on_next_click(card_code),
            bg=self.config.register_button_color,
            fg=self.config.button_text_color,
            activebackground=self.config.register_button_color,
            activeforeground=self.config.button_text_color,
            relief="flat",
            bd=0,
            padx=8,
            pady=8,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))

        tk.Button(
            button_frame,
            text="Cancelar",
            command=self._cancel_pending_typing_prompt,
            bg=self.config.close_button_color,
            fg=self.config.button_text_color,
            activebackground=self.config.close_button_color,
            activeforeground=self.config.button_text_color,
            relief="flat",
            bd=0,
            padx=8,
            pady=8,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.typing_prompt_window.protocol("WM_DELETE_WINDOW", self._cancel_pending_typing_prompt)
        self.status_vars["typing"].set("aguardando confirmação do operador")

    def _arm_typing_on_next_click(self, card_code: str) -> None:
        allowed, typing_message = self.keyboard_typer.can_type_identifier(
            card_code,
            enabled=self.config.type_recognized_id and self.mode == "recognition",
        )
        if not allowed:
            self.status_vars["typing"].set(typing_message)
            self._cancel_pending_typing_prompt()
            return

        if self.typing_prompt_window is not None:
            try:
                self.typing_prompt_window.destroy()
            except tk.TclError:
                pass
            self.typing_prompt_window = None

        self.status_vars["typing"].set("clique no campo de destino")
        try:
            from pynput import mouse

            self.pending_mouse_listener = mouse.Listener(
                on_click=lambda _x, _y, _button, pressed: self._on_destination_mouse_click(card_code, pressed)
            )
            self.pending_mouse_listener.start()
            self.status_vars["typing"].set("aguardando clique no campo de destino")
        except Exception as exc:
            logging.exception("Falha ao aguardar clique do mouse: %s", exc)
            self.pending_mouse_listener = None
            self.pending_typing_code = None
            self.pending_typing_due_at = 0.0
            self.status_vars["typing"].set(f"mouse indisponivel: {exc}")

    def _on_destination_mouse_click(self, card_code: str, pressed: bool) -> bool:
        if not pressed:
            return True

        listener = self.pending_mouse_listener
        self.pending_mouse_listener = None
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass

        self.pending_typing_due_at = time.time() + 0.55
        self.status_vars["typing"].set("campo selecionado; digitando em instantes")
        return False

    def _process_due_pending_typing(self) -> None:
        if not self.pending_typing_code or self.pending_typing_due_at <= 0.0:
            return
        if time.time() < self.pending_typing_due_at:
            return

        card_code = self.pending_typing_code
        self.pending_typing_due_at = 0.0
        self._type_pending_card_code(card_code)

    def _type_pending_card_code(self, card_code: str) -> None:
        if self.pending_typing_code != card_code:
            return

        typed, typing_message = self.keyboard_typer.type_identifier(
            card_code,
            enabled=self.config.type_recognized_id and self.mode == "recognition",
        )
        self.pending_typing_code = None
        self.pending_typing_due_at = 0.0
        self.status_vars["typing"].set(typing_message)
        if typed:
            logging.info("Código do cartão digitado via pynput: %s", card_code)

    def _cancel_pending_typing_prompt(self) -> None:
        if self.pending_mouse_listener is not None:
            try:
                self.pending_mouse_listener.stop()
            except Exception:
                pass
            self.pending_mouse_listener = None

        if self.typing_prompt_window is not None:
            try:
                self.typing_prompt_window.destroy()
            except tk.TclError:
                pass
            self.typing_prompt_window = None

        self.pending_typing_code = None
        self.pending_typing_due_at = 0.0

    def start_registration(self) -> None:
        if self.mode != "recognition":
            messagebox.showinfo("Operacao em andamento", "Finalize ou cancele o cadastro atual antes de iniciar outro.", parent=self.root)
            return
        if not self.camera.is_open:
            messagebox.showerror("Camera indisponivel", "A webcam precisa estar ativa para cadastrar uma pessoa.", parent=self.root)
            return

        registration_id = simpledialog.askstring(
            "Cadastrar pessoa",
            f"Digite o código do cartão (hexadecimal, até {CARD_CODE_MAX_LENGTH} dígitos):",
            parent=self.root,
        )
        if registration_id is None:
            return
        is_valid_card_code, registration_id, card_code_error = validate_card_code(registration_id)
        if not is_valid_card_code:
            messagebox.showerror("Código do cartão inválido", card_code_error, parent=self.root)
            return

        name = simpledialog.askstring("Cadastrar pessoa", "Digite o nome da pessoa:", parent=self.root)
        if name is None:
            return
        name = name.strip()
        if not name:
            messagebox.showerror("Nome invalido", "O nome nao pode ficar vazio.", parent=self.root)
            return

        existing_user = self.db.get_user_by_registration(registration_id)
        if existing_user:
            should_extend = messagebox.askyesno(
                "Pessoa ja cadastrada",
                f"O código do cartão {registration_id} já existe para {existing_user['name']}.\nDeseja adicionar novas amostras?",
                parent=self.root,
            )
            if not should_extend:
                return

        self.mode = "registering"
        self.registration_session = RegistrationSession(registration_id=registration_id, name=name)
        self.temporal_window.reset()
        self._set_action_buttons_enabled(False)
        self._set_status("Cadastro iniciado: pose frente", (0, 220, 255), duration=3.0)
        self.status_vars["operator"].set("cadastro: frente")
        self.status_vars["recognized_user"].set(name)
        self.status_vars["recognized_id"].set(registration_id)
        self.status_vars["frames"].set(f"0/{self.config.samples_per_pose}")
        self._show_pose_popup("front")

    def cancel_registration(self) -> None:
        if self.mode != "registering":
            return
        self.mode = "recognition"
        self.registration_session = None
        self._set_action_buttons_enabled(True)
        self.temporal_window.reset()
        self._set_status("Cadastro cancelado", (0, 165, 255), duration=2.0)
        self.status_vars["operator"].set("reconhecendo")

    def _process_registration_frame(self, detection: FaceDetection, extra_lines: List[str]) -> None:
        session = self.registration_session
        if session is None:
            self.mode = "recognition"
            return

        pose = session.current_pose
        pose_label = POSE_LABELS[pose]
        elapsed = time.time() - session.pose_started_at
        remaining = max(self.config.pose_capture_timeout - elapsed, 0.0)
        collected = len(session.current_samples)

        self.status_vars["operator"].set(f"cadastro: {pose_label}")
        self.status_vars["frames"].set(f"{collected}/{self.config.samples_per_pose}")
        extra_lines.append(f"Cadastro: {pose_label}")
        extra_lines.append(f"Amostras: {collected}/{self.config.samples_per_pose}")
        extra_lines.append(f"Tempo restante: {remaining:.1f}s")

        if not detection.found:
            extra_lines.append("Centralize o rosto")
        elif detection.pose != pose:
            extra_lines.append(f"Pose atual: {POSE_LABELS[detection.pose]}")
            extra_lines.append(f"Ajuste para: {pose_label}")
        elif detection.embedding is None:
            extra_lines.append(detection.feedback)
        else:
            now = time.time()
            if now - session.last_capture_at >= self.config.sample_capture_interval:
                session.current_samples.append(detection.embedding)
                session.last_capture_at = now
                extra_lines.append("Frame valido capturado")
            else:
                extra_lines.append("Mantenha a pose")

        if len(session.current_samples) >= self.config.samples_per_pose:
            self._accept_current_registration_pose()
            return

        if elapsed >= self.config.pose_capture_timeout:
            if len(session.current_samples) >= self.config.min_samples_per_pose:
                self._accept_current_registration_pose()
            else:
                self._fail_registration(
                    f"Nao foi possivel capturar amostras suficientes para {pose_label}. "
                    f"Minimo: {self.config.min_samples_per_pose}."
                )

    def _accept_current_registration_pose(self) -> None:
        session = self.registration_session
        if session is None:
            return

        pose = session.current_pose
        session.captured_by_pose[pose] = average_embeddings(session.current_samples)
        session.pose_index += 1

        if session.pose_index >= len(session.poses):
            self._complete_registration()
            return

        next_pose = session.current_pose
        session.reset_current_pose()
        self._set_status(f"Cadastro: agora vire para {POSE_LABELS[next_pose]}", (0, 220, 255), duration=3.0)
        self.status_vars["operator"].set(f"cadastro: {POSE_LABELS[next_pose]}")
        self.status_vars["frames"].set(f"0/{self.config.samples_per_pose}")
        self._show_pose_popup(next_pose)

    def _complete_registration(self) -> None:
        session = self.registration_session
        if session is None:
            return

        try:
            user_id = self.db.upsert_user(session.registration_id, session.name)
            for pose, embedding in session.captured_by_pose.items():
                self.db.add_embedding(user_id, pose, embedding)

            self.cache.refresh(force=True)
            self._refresh_database_status()
            self.temporal_window.reset()
            self._set_status("Cadastro concluido", (0, 220, 0), duration=3.0)
            messagebox.showinfo(
                "Cadastro concluido",
                f"Cadastro concluído com sucesso.\n\n{session.name} (código do cartão {session.registration_id}) foi cadastrado com amostras front/left/right.",
                parent=self.root,
            )
        except sqlite3.IntegrityError as exc:
            messagebox.showerror("Erro ao salvar", f"Nao foi possivel salvar o cadastro.\n\n{exc}", parent=self.root)
            logging.exception("Erro de integridade ao salvar cadastro: %s", exc)
        finally:
            self.mode = "recognition"
            self.registration_session = None
            self._set_action_buttons_enabled(True)
            self.status_vars["operator"].set("reconhecendo")

    def _fail_registration(self, message: str) -> None:
        self.mode = "recognition"
        self.registration_session = None
        self._set_action_buttons_enabled(True)
        self.temporal_window.reset()
        self._set_status("Cadastro falhou", (0, 0, 255), duration=3.0)
        self.status_vars["operator"].set("reconhecendo")
        messagebox.showerror("Cadastro falhou", message, parent=self.root)

    def _show_pose_popup(self, pose: str) -> None:
        message = POSE_POPUP_MESSAGES.get(pose)
        if not message:
            return
        messagebox.showinfo("Cadastro guiado", message, parent=self.root)

    def delete_user(self) -> None:
        if self.mode != "recognition":
            messagebox.showinfo("Operacao em andamento", "Finalize ou cancele o cadastro antes de excluir uma pessoa.", parent=self.root)
            return

        if self.delete_window is not None and self.delete_window.winfo_exists():
            self.delete_window.lift()
            self.delete_window.focus_force()
            return

        self.delete_window = tk.Toplevel(self.root)
        self.delete_window.title("Excluir pessoa")
        self.delete_window.geometry("520x430")
        self.delete_window.minsize(360, 300)
        self.delete_window.configure(bg=self.config.sidebar_color)
        self.delete_window.transient(self.root)
        self.delete_window.grab_set()
        self.delete_window.columnconfigure(0, weight=1)
        self.delete_window.rowconfigure(3, weight=1)

        search_var = tk.StringVar()
        registration_var = tk.StringVar()
        display_users: List[sqlite3.Row] = []

        title = tk.Label(
            self.delete_window,
            text="Excluir usuário por código do cartão",
            bg=self.config.sidebar_color,
            fg="#ffffff",
            font=("Segoe UI", 13, "bold"),
            anchor="w",
        )
        title.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))

        search_frame = tk.Frame(self.delete_window, bg=self.config.sidebar_color)
        search_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        search_frame.columnconfigure(0, weight=1)
        tk.Label(
            search_frame,
            text="Buscar por código do cartão",
            bg=self.config.sidebar_color,
            fg="#d9e2ec",
            font=("Segoe UI", 9),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        search_entry = tk.Entry(search_frame, textvariable=search_var, font=("Segoe UI", 10))
        search_entry.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        selected_frame = tk.Frame(self.delete_window, bg=self.config.sidebar_color)
        selected_frame.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))
        selected_frame.columnconfigure(0, weight=1)
        tk.Label(
            selected_frame,
            text="Código do cartão selecionado",
            bg=self.config.sidebar_color,
            fg="#d9e2ec",
            font=("Segoe UI", 9),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        registration_entry = tk.Entry(selected_frame, textvariable=registration_var, font=("Segoe UI", 10))
        registration_entry.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        list_frame = tk.Frame(self.delete_window, bg=self.config.sidebar_color)
        list_frame.grid(row=3, column=0, sticky="nsew", padx=14, pady=(0, 10))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        user_listbox = tk.Listbox(
            list_frame,
            activestyle="dotbox",
            bg="#f8fafc",
            fg="#17202a",
            selectbackground="#A7C7E7",
            selectforeground="#17202a",
            font=("Segoe UI", 10),
            height=8,
            exportselection=False,
        )
        user_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=user_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        user_listbox.configure(yscrollcommand=scrollbar.set)

        button_frame = tk.Frame(self.delete_window, bg=self.config.sidebar_color)
        button_frame.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 14))
        button_frame.columnconfigure(0, weight=1)
        delete_button = tk.Button(
            button_frame,
            text="Excluir usuário selecionado",
            bg=self.config.delete_button_color,
            fg=self.config.button_text_color,
            activebackground=self.config.delete_button_color,
            activeforeground=self.config.button_text_color,
            relief="flat",
            bd=0,
            padx=10,
            pady=8,
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            command=lambda: confirm_delete(),
        )
        delete_button.grid(row=0, column=0, sticky="ew")

        def user_label(user: sqlite3.Row) -> str:
            return f"{user['name']} — código {user['registration_id']}"

        def refresh_user_list() -> None:
            nonlocal display_users
            query = search_var.get().strip().lower()
            users = self.db.list_users()
            display_users = [
                user for user in users if not query or query in str(user["registration_id"]).lower()
            ]

            user_listbox.delete(0, tk.END)
            if not display_users:
                user_listbox.insert(tk.END, "Nenhum usuário encontrado")
                return

            for user in display_users:
                user_listbox.insert(tk.END, user_label(user))

        def fill_selected_registration(_event: Optional[tk.Event] = None) -> None:
            selection = user_listbox.curselection()
            if not selection or not display_users:
                return
            index = selection[0]
            if index >= len(display_users):
                return
            registration_var.set(str(display_users[index]["registration_id"]))

        def confirm_delete() -> None:
            is_valid_card_code, registration_id, card_code_error = validate_card_code(registration_var.get())
            if not is_valid_card_code:
                messagebox.showerror("Código do cartão inválido", card_code_error, parent=self.delete_window)
                return

            user = self.db.get_user_by_registration(registration_id)
            if user is None:
                messagebox.showerror("Usuário não encontrado", "Nenhum usuário cadastrado possui esse código do cartão.", parent=self.delete_window)
                return

            user_text = f"{user['name']} — código {user['registration_id']}"
            confirmed = messagebox.askyesno(
                "Confirmar exclusão",
                f"Tem certeza que deseja excluir o usuário {user_text}?",
                parent=self.delete_window,
            )
            if not confirmed:
                return

            deleted = self.db.delete_user_by_registration_id(registration_id)
            if deleted is None:
                messagebox.showerror("Usuário não encontrado", "O código do cartão informado não existe mais no banco.", parent=self.delete_window)
                return

            self.cache.refresh(force=True)
            self._refresh_database_status()
            self.temporal_window.reset()
            self.status_vars["recognized_user"].set("-")
            self.status_vars["recognized_id"].set("-")
            registration_var.set("")
            refresh_user_list()
            self._set_status("Pessoa excluida", (0, 220, 0), duration=3.0)
            messagebox.showinfo("Pessoa excluída", f"Usuário {user_text} excluído com sucesso.", parent=self.delete_window)

        def close_delete_window() -> None:
            if self.delete_window is not None:
                try:
                    self.delete_window.grab_release()
                    self.delete_window.destroy()
                except tk.TclError:
                    pass
                self.delete_window = None

        search_var.trace_add("write", lambda *_args: refresh_user_list())
        user_listbox.bind("<<ListboxSelect>>", fill_selected_registration)
        self.delete_window.protocol("WM_DELETE_WINDOW", close_delete_window)
        refresh_user_list()
        search_entry.focus_set()

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.register_button.configure(state=state)
        self.delete_button.configure(state=state)

    def _update_detection_status(self, detection: FaceDetection) -> None:
        if detection.found:
            self.status_vars["pose"].set(POSE_LABELS[detection.pose])
        else:
            self.status_vars["pose"].set("aguardando")

    def _refresh_database_status(self) -> None:
        self.status_vars["database"].set(f"conectado: {self.cache.user_count} usuario(s), {self.cache.sample_count} amostra(s)")

    def _update_typing_status(self) -> None:
        ask_before_typing = self.typing_enabled_var.get()
        typing_available = self.config.type_recognized_id and self.keyboard_typer.available
        if ask_before_typing and not typing_available:
            self.typing_enabled_var.set(False)
            ask_before_typing = False
        self._update_enter_typing_status()
        if typing_available:
            enter_status = "com Enter" if self.config.press_enter_after_typing else "sem Enter"
            prompt_status = "pergunta antes" if ask_before_typing else "digita direto"
            self.status_vars["typing"].set(
                f"ativa, {prompt_status}, {enter_status}, max {self.config.max_consecutive_typings_per_id} por código"
            )
        elif not self.keyboard_typer.available:
            self.status_vars["typing"].set("indisponivel")
        else:
            self.status_vars["typing"].set("desativada")

    def _update_enter_typing_status(self) -> None:
        typing_available = self.config.type_recognized_id and self.keyboard_typer.available
        if not typing_available:
            self.enter_typing_enabled_var.set(False)
        self.config.press_enter_after_typing = bool(self.enter_typing_enabled_var.get() and typing_available)
        state = "normal" if typing_available else "disabled"
        if hasattr(self, "enter_typing_check"):
            self.enter_typing_check.configure(state=state)

    def _set_status(self, message: str, color: Tuple[int, int, int], duration: float = 1.5) -> None:
        self.status_message = message
        self.status_color = color
        self.status_until = time.time() + duration
        self.header_status_var.set(f"Status: {message}")

    def _active_status(self) -> Tuple[str, Tuple[int, int, int]]:
        if time.time() <= self.status_until:
            return self.status_message, self.status_color
        return "Procurando rosto", (0, 220, 255)

    def _draw_detection_overlay(self, frame: np.ndarray, detection: FaceDetection, extra_lines: List[str]) -> None:
        if detection.found and detection.box is not None:
            x1, y1, x2, y2 = detection.box
            box_color = (0, 220, 0) if detection.embedding is not None else (0, 190, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame,
                text_for_opencv(f"pose: {POSE_LABELS[detection.pose]}"),
                (x1, max(24, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                box_color,
                2,
                cv2.LINE_AA,
            )

        status_message, status_color = self._active_status()
        lines = [status_message]
        if detection.found:
            lines.append(f"Pose detectada: {POSE_LABELS[detection.pose]}")
            lines.append(f"Face: {detection.face_ratio:.3f} | Nitidez: {detection.sharpness:.1f}")
        else:
            lines.append("Pose detectada: aguardando")
        lines.extend(extra_lines[:5])
        self._draw_text_block(frame, lines, status_color)

    def _draw_text_block(
        self,
        frame: np.ndarray,
        lines: List[str],
        color: Tuple[int, int, int],
        start_y: int = 28,
    ) -> None:
        y = start_y
        for line in lines:
            if not line:
                continue
            safe_line = text_for_opencv(line)
            (text_width, text_height), _ = cv2.getTextSize(safe_line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            cv2.rectangle(frame, (10, y - text_height - 9), (26 + text_width, y + 9), (9, 12, 17), -1)
            cv2.putText(frame, safe_line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
            y += 30

    def _blank_frame(self, message: str) -> np.ndarray:
        frame = np.zeros((self.config.camera_height, self.config.camera_width, 3), dtype=np.uint8)
        cv2.putText(frame, text_for_opencv(message), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    def _show_frame(self, frame: np.ndarray) -> None:
        width = max(self.video_label.winfo_width(), 320)
        height = max(self.video_label.winfo_height(), 240)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        image.thumbnail((width, height), resampling)

        canvas = Image.new("RGB", (width, height), (5, 7, 10))
        x = (width - image.width) // 2
        y = (height - image.height) // 2
        canvas.paste(image, (x, y))

        self.video_image = ImageTk.PhotoImage(image=canvas)
        self.video_label.configure(image=self.video_image)

    def _format_distance(self, distance: float) -> str:
        if not np.isfinite(distance):
            return "-"
        return f"{distance:.3f}"

    def close(self) -> None:
        self._cancel_pending_typing_prompt()

        if self.delete_window is not None:
            try:
                self.delete_window.destroy()
            except tk.TclError:
                pass
            self.delete_window = None

        if not self.running:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            return

        self.running = False
        self.camera.release()
        cv2.destroyAllWindows()
        self.db.close()
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def build_config_from_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Reconhecimento facial em tempo real com votação temporal multi-pose.")
    parser.add_argument("--camera-index", type=int, default=0, help="Índice da webcam.")
    parser.add_argument("--window-size", type=int, default=20, help="Quantidade de frames válidos na janela temporal.")
    parser.add_argument("--min-valid-frames", type=int, default=8, help="Quantidade minima de frames validos antes de decidir.")
    parser.add_argument("--min-votes", type=int, default=6, help="Numero minimo de votos para confirmar reconhecimento.")
    parser.add_argument("--min-vote-ratio", type=float, default=0.30, help="Proporcao minima de votos do vencedor na janela temporal.")
    parser.add_argument("--frame-threshold", type=float, default=1.02, help="Threshold por frame para aceitar candidato.")
    parser.add_argument("--final-threshold", type=float, default=0.90, help="Threshold médio final para confirmar o vencedor.")
    parser.add_argument("--unknown-threshold", type=float, default=1.10, help="Threshold médio para sugerir nova face.")
    parser.add_argument("--samples-per-pose", type=int, default=15, help="Frames validos capturados por pose no cadastro.")
    parser.add_argument("--min-samples-per-pose", type=int, default=8, help="Minimo de frames validos aceitos por pose no cadastro.")
    parser.add_argument("--type-cooldown", type=float, default=5.0, help="Cooldown em segundos para digitacao automatica.")
    parser.add_argument("--max-consecutive-typings", type=int, default=1, help="Maximo de digitacoes consecutivas para o mesmo codigo do cartao.")
    parser.add_argument("--disable-auto-type", action="store_true", help="Inicia com a pergunta para digitar o codigo do cartao desativada.")
    parser.add_argument("--press-enter-after-typing", action="store_true", help="Pressiona Enter apos digitar o codigo do cartao reconhecido.")
    parser.add_argument("--min-window-width", type=int, default=420, help="Largura minima da janela principal.")
    parser.add_argument("--min-window-height", type=int, default=300, help="Altura minima da janela principal.")
    parser.add_argument("--db-path", type=str, default="", help="Caminho opcional para o banco SQLite.")
    parser.add_argument("--key-path", type=str, default="", help="Caminho opcional para a chave Fernet.")
    args = parser.parse_args()

    config = AppConfig(
        camera_index=args.camera_index,
        temporal_window=args.window_size,
        min_valid_frames=args.min_valid_frames,
        min_confirm_votes=args.min_votes,
        min_vote_ratio=args.min_vote_ratio,
        frame_candidate_threshold=args.frame_threshold,
        final_distance_threshold=args.final_threshold,
        unknown_distance_threshold=args.unknown_threshold,
        samples_per_pose=args.samples_per_pose,
        min_samples_per_pose=args.min_samples_per_pose,
        type_recognized_id=not args.disable_auto_type,
        type_cooldown_seconds=args.type_cooldown,
        max_consecutive_typings_per_id=args.max_consecutive_typings,
        press_enter_after_typing=args.press_enter_after_typing,
        min_window_width=args.min_window_width,
        min_window_height=args.min_window_height,
    )

    config.data_dir.mkdir(parents=True, exist_ok=True)
    if args.db_path:
        config.db_path = Path(args.db_path).expanduser().resolve()
        config.data_dir = config.db_path.parent
    if args.key_path:
        config.key_path = Path(args.key_path).expanduser().resolve()
    else:
        config.key_path = config.data_dir / "fernet.key"

    return config


def main() -> None:
    cv2.setUseOptimized(True)
    for warning in collect_environment_warnings():
        logging.warning(warning)
    config = build_config_from_args()
    app = MainApp(config)
    app.start()


if __name__ == "__main__":
    main()
