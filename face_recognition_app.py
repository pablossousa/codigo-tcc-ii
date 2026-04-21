from __future__ import annotations

import argparse
from importlib import metadata as importlib_metadata
import logging
import sqlite3
import time
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
from tkinter import messagebox, simpledialog


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


POSE_LABELS = {
    "front": "frente",
    "left": "esquerda na tela",
    "right": "direita na tela",
}


@dataclass(slots=True)
class AppConfig:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_warmup_frames: int = 12
    frame_candidate_threshold: float = 1.02
    final_distance_threshold: float = 0.90
    unknown_distance_threshold: float = 1.10
    temporal_window: int = 20
    min_confirm_votes: int = 9
    min_vote_ratio: float = 0.45
    samples_per_pose: int = 12
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
    window_name: str = "Reconhecimento Facial em Tempo Real"
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    db_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "faces.db")
    key_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data" / "fernet.key")


@dataclass(slots=True)
class FaceSample:
    sample_id: int
    user_id: int
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
    user_name: str
    distance: float
    effective_distance: float
    matched_pose: str
    is_within_candidate_threshold: bool


@dataclass(slots=True)
class FrameVote:
    user_id: Optional[int]
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
    user_name: Optional[str] = None
    votes: int = 0
    total_frames: int = 0
    vote_ratio: float = 0.0
    mean_distance: float = float("inf")
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
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
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
        required = {"user_id", "name", "created_at"}
        if required.issubset(columns):
            return

        logging.info("Migrando tabela users para o formato atual")
        self.conn.execute("ALTER TABLE users RENAME TO users_legacy")
        self._create_users_table()

        legacy_columns = self._columns("users_legacy")
        id_expr = "user_id" if "user_id" in legacy_columns else ("id" if "id" in legacy_columns else "rowid")
        name_expr = "name" if "name" in legacy_columns else "'user_' || rowid"
        created_expr = "created_at" if "created_at" in legacy_columns else "CURRENT_TIMESTAMP"

        self.conn.execute(
            f"""
            INSERT OR IGNORE INTO users (user_id, name, created_at)
            SELECT {id_expr}, {name_expr}, {created_expr}
            FROM users_legacy
            """
        )
        self.conn.execute("DROP TABLE users_legacy")

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

    def get_user_by_name(self, name: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM users WHERE name = ?", (name.strip(),)).fetchone()

    def upsert_user(self, name: str) -> int:
        clean_name = name.strip()
        existing = self.get_user_by_name(clean_name)
        if existing:
            return int(existing["user_id"])

        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO users (name, created_at) VALUES (?, CURRENT_TIMESTAMP)",
                (clean_name,),
            )
            return int(cursor.lastrowid)

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
                samples.append(
                    FaceSample(
                        sample_id=int(row["sample_id"]),
                        user_id=int(row["user_id"]),
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

            for sample in samples:
                raw_distance = float(np.linalg.norm(embedding - sample.embedding))
                effective_distance = raw_distance * self._pose_weight(detected_pose, sample.pose)
                if effective_distance < best_effective_distance:
                    best_raw_distance = raw_distance
                    best_effective_distance = effective_distance
                    best_pose = sample.pose

            candidate = MatchCandidate(
                user_id=user_id,
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
        if total_frames == 0:
            return TemporalDecision(status="empty", message="Sem frames válidos")

        matched_votes = [vote for vote in self.votes if vote.user_id is not None]
        all_best_distances = [vote.best_distance for vote in self.votes if np.isfinite(vote.best_distance)]

        if not matched_votes:
            mean_distance = float(np.mean(all_best_distances)) if all_best_distances else float("inf")
            if mean_distance >= self.config.unknown_distance_threshold or not all_best_distances:
                return TemporalDecision(
                    status="unknown",
                    total_frames=total_frames,
                    mean_distance=mean_distance,
                    message="Nova face detectada",
                )
            return TemporalDecision(
                status="uncertain",
                total_frames=total_frames,
                mean_distance=mean_distance,
                message="Sem confiança suficiente para reconhecer",
            )

        counter = Counter(vote.user_id for vote in matched_votes if vote.user_id is not None)
        winner_id, winner_votes = counter.most_common(1)[0]
        winner_frames = [vote for vote in matched_votes if vote.user_id == winner_id]
        mean_distance = float(np.mean([vote.distance for vote in winner_frames]))
        vote_ratio = winner_votes / max(total_frames, 1)
        user_name = winner_frames[0].user_name

        if (
            winner_votes >= self.config.min_confirm_votes
            and vote_ratio >= self.config.min_vote_ratio
            and mean_distance <= self.config.final_distance_threshold
        ):
            return TemporalDecision(
                status="confirmed",
                user_id=winner_id,
                user_name=user_name,
                votes=winner_votes,
                total_frames=total_frames,
                vote_ratio=vote_ratio,
                mean_distance=mean_distance,
                message=f"Encontrado: {user_name}",
            )

        best_distance = float(np.mean(all_best_distances)) if all_best_distances else float("inf")
        if best_distance >= self.config.unknown_distance_threshold:
            return TemporalDecision(
                status="unknown",
                total_frames=total_frames,
                mean_distance=best_distance,
                message="Nova face detectada",
            )

        return TemporalDecision(
            status="uncertain",
            user_id=winner_id,
            user_name=user_name,
            votes=winner_votes,
            total_frames=total_frames,
            vote_ratio=vote_ratio,
            mean_distance=mean_distance,
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
            (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
            cv2.rectangle(frame, (10, y - 18), (20 + text_width, y + 8), (0, 0, 0), -1)
            cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
            y += 28

    def _draw_detection_overlay(self, frame: np.ndarray, detection: FaceDetection, extra_lines: List[str]) -> None:
        if detection.found and detection.box is not None:
            x1, y1, x2, y2 = detection.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                frame,
                f"pose: {POSE_LABELS[detection.pose]}",
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
                user_name=None,
                distance=float("inf"),
                best_distance=float("inf"),
                detected_pose=detected_pose,
                matched_pose=None,
                timestamp=time.time(),
            )

        return FrameVote(
            user_id=candidate.user_id if candidate.is_within_candidate_threshold else None,
            user_name=candidate.user_name if candidate.is_within_candidate_threshold else None,
            distance=candidate.distance,
            best_distance=candidate.distance,
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

        user_id = self.db.upsert_user(name)
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


def build_config_from_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Reconhecimento facial em tempo real com votação temporal multi-pose.")
    parser.add_argument("--camera-index", type=int, default=0, help="Índice da webcam.")
    parser.add_argument("--window-size", type=int, default=20, help="Quantidade de frames válidos na janela temporal.")
    parser.add_argument("--min-votes", type=int, default=9, help="Número mínimo de votos para confirmar reconhecimento.")
    parser.add_argument("--frame-threshold", type=float, default=1.02, help="Threshold por frame para aceitar candidato.")
    parser.add_argument("--final-threshold", type=float, default=0.90, help="Threshold médio final para confirmar o vencedor.")
    parser.add_argument("--unknown-threshold", type=float, default=1.10, help="Threshold médio para sugerir nova face.")
    parser.add_argument("--samples-per-pose", type=int, default=12, help="Frames válidos capturados por pose no cadastro.")
    parser.add_argument("--db-path", type=str, default="", help="Caminho opcional para o banco SQLite.")
    parser.add_argument("--key-path", type=str, default="", help="Caminho opcional para a chave Fernet.")
    args = parser.parse_args()

    config = AppConfig(
        camera_index=args.camera_index,
        temporal_window=args.window_size,
        min_confirm_votes=args.min_votes,
        frame_candidate_threshold=args.frame_threshold,
        final_distance_threshold=args.final_threshold,
        unknown_distance_threshold=args.unknown_threshold,
        samples_per_pose=args.samples_per_pose,
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
    app = FaceRecognitionApp(config)
    app.run()


if __name__ == "__main__":
    main()
