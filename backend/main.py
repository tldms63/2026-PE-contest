"""올림픽 아카이브 검색 백엔드 (FastAPI)."""
import json
import os
import sqlite3
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .scoring import evaluate_pose

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(ROOT_DIR, "data", "olympics.db")
POSES_DIR = os.path.join(BASE_DIR, "static", "poses")

app = FastAPI(title="Olympics Archive Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/api/olympics/sports")
def list_sports() -> list[str]:
    """DB에 존재하는 종목명을 중복 없이 정렬해 반환한다 (검색 필터용)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT sport
            FROM olympics
            WHERE sport IS NOT NULL AND sport != ''
            ORDER BY sport
            """
        ).fetchall()
    finally:
        conn.close()

    return [row["sport"] for row in rows]


def _with_raw_fields(row: sqlite3.Row) -> dict:
    """DB에 저장된 raw_json(원본 공공데이터)에서 화면 표시에 쓸 수 있는
    추가 정보(참가국, 영상 길이, 원본 출처 링크)를 꺼내 응답에 덧붙인다."""
    record = dict(row)
    raw = record.pop("raw_json", None)

    record["country"] = None
    record["duration_sec"] = None
    record["source_url"] = None

    if raw:
        try:
            raw_obj = json.loads(raw)
        except (TypeError, ValueError):
            raw_obj = {}

        record["country"] = raw_obj.get("country_kor_nm") or None

        duration = raw_obj.get("mv_time_len")
        try:
            record["duration_sec"] = int(duration) if duration not in (None, "") else None
        except (TypeError, ValueError):
            record["duration_sec"] = None

        record["source_url"] = raw_obj.get("item_url") or None

    return record


@app.get("/api/olympics/search")
def search_olympics(keyword: Optional[str] = None, sport: Optional[str] = None) -> list[dict]:
    """title/summary를 keyword로, sport를 sport로 부분 일치 검색한다 (둘 다 선택값, AND 결합)."""
    query = "SELECT id, title, sport, event_date, summary, video_url, raw_json FROM olympics WHERE 1=1"
    params: list[str] = []

    if keyword:
        query += " AND (title LIKE ? OR summary LIKE ?)"
        like = f"%{keyword}%"
        params.extend([like, like])

    if sport:
        query += " AND sport LIKE ?"
        params.append(f"%{sport}%")

    query += " ORDER BY id"

    conn = get_connection()
    try:
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    return [_with_raw_fields(row) for row in rows]


@app.get("/api/olympics/{match_id}/keypoints")
def get_keypoints(match_id: int) -> FileResponse:
    """extract_pose.py가 미리 추출해 둔 관절 좌표 시퀀스를 반환한다.

    static/poses/{id}.json이 아직 없으면(추출 전이면) 404를 반환한다.
    """
    pose_path = os.path.join(POSES_DIR, f"{match_id}.json")
    if not os.path.isfile(pose_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"id={match_id}에 대한 포즈 데이터가 없습니다. "
                "extract_pose.py로 먼저 추출해야 합니다."
            ),
        )
    return FileResponse(pose_path, media_type="application/json")


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


class PoseFrame(BaseModel):
    """웹캠에서 한 프레임마다 뽑은 33개 관절 좌표. 포즈를 못 찾은 프레임은
    landmarks를 생략하거나 null로 보내면 된다 (extract_pose.py 출력과 동일한 형식)."""

    landmarks: Optional[list[LandmarkPoint]] = None


class EvaluateRequest(BaseModel):
    match_id: int
    frames: list[PoseFrame]


def _load_reference_frames(match_id: int) -> Optional[list]:
    """extract_pose.py가 미리 뽑아 둔 기준 영상의 관절 좌표 시퀀스를 읽어온다.
    아직 추출 전이라 파일이 없으면 None을 반환한다 (채점 로직이 알아서 대체 기준으로 처리)."""
    pose_path = os.path.join(POSES_DIR, f"{match_id}.json")
    if not os.path.isfile(pose_path):
        return None
    with open(pose_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [frame.get("landmarks") for frame in data.get("frames", [])]


@app.post("/api/olympics/evaluate")
def evaluate_motion(payload: EvaluateRequest) -> dict:
    """사용자가 따라 한 동작을 채점한다.

    정밀한 선수 코칭이 아니라 학생들이 즐기는 체감형 아케이드 게임이 목표라,
    화면에 감지되어 동작을 시도하기만 해도 최소 점수를 보장하고(scoring.BASE_SCORE),
    종목별 핵심 관절 위주로 아주 너그럽게 채점한다. 자세한 채점 기준은 scoring.py 참고.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT sport FROM olympics WHERE id = ?", (payload.match_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(
            status_code=404, detail=f"id={payload.match_id}에 대한 경기를 찾을 수 없습니다."
        )

    user_frames = [
        [lm.model_dump() for lm in frame.landmarks] if frame.landmarks else None
        for frame in payload.frames
    ]
    ref_frames = _load_reference_frames(payload.match_id)

    return evaluate_pose(row["sport"], user_frames, ref_frames)
