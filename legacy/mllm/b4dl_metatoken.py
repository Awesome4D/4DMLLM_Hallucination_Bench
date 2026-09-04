import json
import math
import os
import re
from collections import Counter
from pathlib import Path


PROMPT_PREFIX = "<video>\n<4DLiDAR>\n<meta>"


def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def require_nuscenes_root(nuscenes_root=None):
    root = nuscenes_root or os.environ.get("NUSCENES_ROOT")
    if not root:
        raise ValueError("Set NUSCENES_ROOT or pass --nuscenes-root.")
    return Path(root)


def metadata_file(nuscenes_root, name):
    root = require_nuscenes_root(nuscenes_root)
    candidates = [
        root / name,
        root / "v1.0-trainval" / name,
        root / "v1.0-mini" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {name} under {root}")


def load_scene_metadata(path):
    scenes = load_json(path)
    return {scene["scene_id"]: scene for scene in scenes}


def frame_index_from_key(key):
    match = re.search(r"(\d+)$", key)
    if not match:
        raise ValueError(f"Could not parse frame key: {key}")
    return int(match.group(1))


def scene_frame_paths(scene):
    lidar_paths = scene.get("paths", {}).get("PATH_LIDAR_TOP", {})
    frames = {}
    for key, value in lidar_paths.items():
        frames[frame_index_from_key(key)] = value
    if not frames:
        raise ValueError(f"No PATH_LIDAR_TOP entries for scene {scene.get('scene_id')}")
    return dict(sorted(frames.items()))


def available_frame_indices(scene):
    return list(scene_frame_paths(scene).keys())


def parse_question_frame_selection(question, frame_indices):
    if not frame_indices:
        raise ValueError("frame_indices cannot be empty")

    last = max(frame_indices)
    first = min(frame_indices)
    text = str(question or "").lower()

    interval_patterns = [
        r"from\s+frames?\s+(\d+)\s+to\s+(?:frame\s+)?(\d+)",
        r"between\s+frames?\s+(\d+)\s+and\s+(?:frame\s+)?(\d+)",
        r"during\s+frames?\s+(\d+)\s+to\s+(?:frame\s+)?(\d+)",
        r"frames?\s+(\d+)\s+to\s+(?:frame\s+)?(\d+)",
        r"frames?\s+(\d+)\s*[-\u2013]\s*(\d+)",
        r"from\s+(\d+)\s+to\s+(\d+)",
    ]
    intervals = []
    for pattern in interval_patterns:
        for match in re.finditer(pattern, text):
            start, end = int(match.group(1)), int(match.group(2))
            if end >= start:
                intervals.append((start, end))

    if intervals:
        start = min(start for start, _ in intervals)
        end = max(end for _, end in intervals)
        return clip_frame(start, frame_indices), clip_frame(end, frame_indices), "question_interval"

    single_patterns = [
        r"(?:at|in|during|by)\s+frame\s+(\d+)",
        r"\bframe\s+(\d+)\b",
    ]
    singles = []
    for pattern in single_patterns:
        singles.extend(int(match.group(1)) for match in re.finditer(pattern, text))
    if singles:
        frame = clip_frame(singles[0], frame_indices)
        start = min(max(frame - 2, first), max(last - 4, first))
        end = min(start + 4, last)
        return start, end, "question_single_frame_window"

    return first, last, "full_scene_fallback"


def clip_frame(frame, frame_indices):
    return min(frame_indices, key=lambda value: abs(value - frame))


def build_sample_data_by_filename(nuscenes_root):
    sample_data = load_json(metadata_file(nuscenes_root, "sample_data.json"))
    by_filename = {}
    for row in sample_data:
        filename = row.get("filename")
        if filename:
            by_filename[filename] = row
            by_filename["/" + filename] = row
    return by_filename


def build_ego_pose_by_token(nuscenes_root):
    ego_pose = load_json(metadata_file(nuscenes_root, "ego_pose.json"))
    return {row["token"]: row for row in ego_pose}


def yaw_from_quaternion(rotation):
    w, x, y, z = rotation
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def load_scene_motion(scene, sample_data_by_filename, ego_pose_by_token):
    frame_paths = scene_frame_paths(scene)
    motion = {}
    for frame, lidar_path in frame_paths.items():
        sample_data = sample_data_by_filename.get(lidar_path)
        if sample_data is None:
            raise KeyError(
                f"Missing sample_data record for {lidar_path} in scene {scene.get('scene_id')}"
            )
        pose = ego_pose_by_token[sample_data["ego_pose_token"]]
        timestamp = pose.get("timestamp") or sample_data.get("timestamp")
        motion[frame] = {
            "translation": pose["translation"],
            "yaw": yaw_from_quaternion(pose["rotation"]),
            "timestamp": float(timestamp) / 1_000_000.0,
        }
    return motion


def displacement_bin(forward, left, distance):
    if distance < 0.2:
        return "with little relative displacement"
    fb = "forward" if forward >= 0 else "backward"
    lr = "left" if left >= 0 else "right"
    if abs(left) < 0.25 * abs(forward):
        return fb
    if abs(forward) < 0.25 * abs(left):
        return lr
    return f"{fb}-{lr}"


def speed_bin(speed):
    if speed < 0.2:
        return "nearly stationary"
    if speed < 2.0:
        return "moving slowly"
    if speed < 6.0:
        return "moving at moderate speed"
    return "moving fast"


def heading_bin(delta_yaw):
    degrees = math.degrees(delta_yaw)
    if degrees < -5.0:
        return "turning right"
    if degrees > 5.0:
        return "turning left"
    return "keeping a similar heading"


def acceleration_bin(acceleration):
    if acceleration < -0.5:
        return "slowing down"
    if acceleration > 0.5:
        return "speeding up"
    return "maintaining similar speed"


def frame_motion_terms(frame, motion):
    if frame == min(motion):
        return None

    prev = max(idx for idx in motion if idx < frame)
    current_pose = motion[frame]
    prev_pose = motion[prev]
    dt = max(current_pose["timestamp"] - prev_pose["timestamp"], 1e-6)

    dx = current_pose["translation"][0] - prev_pose["translation"][0]
    dy = current_pose["translation"][1] - prev_pose["translation"][1]
    yaw = prev_pose["yaw"]
    forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    left = -math.sin(yaw) * dx + math.cos(yaw) * dy
    distance = math.hypot(forward, left)
    speed = distance / dt

    prev_speed = speed
    earlier_frames = [idx for idx in motion if idx < prev]
    if earlier_frames:
        earlier = max(earlier_frames)
        earlier_pose = motion[earlier]
        prev_dt = max(prev_pose["timestamp"] - earlier_pose["timestamp"], 1e-6)
        prev_dx = prev_pose["translation"][0] - earlier_pose["translation"][0]
        prev_dy = prev_pose["translation"][1] - earlier_pose["translation"][1]
        prev_speed = math.hypot(prev_dx, prev_dy) / prev_dt

    return {
        "speed": speed_bin(speed),
        "direction": displacement_bin(forward, left, distance),
        "heading": heading_bin(wrap_angle(current_pose["yaw"] - prev_pose["yaw"])),
        "acceleration": acceleration_bin((speed - prev_speed) / dt),
    }


def frame_description(frame, role, motion):
    role_text = {
        "first": "the first referenced frame",
        "last": "the last referenced frame",
        "single": "the referenced frame",
    }[role]
    terms = frame_motion_terms(frame, motion)
    if terms is None:
        return (
            f"At {role_text}, frame {frame:03d}, this is the initial ego pose of "
            "the sequence, with no previous frame available for relative motion."
        )
    return (
        f"At {role_text}, frame {frame:03d}, the ego vehicle is "
        f"{terms['speed']} {terms['direction']} relative to the previous frame, "
        f"{terms['heading']}, and {terms['acceleration']}."
    )


def build_metatoken_text(start_frame, end_frame, motion):
    if start_frame == end_frame:
        return frame_description(start_frame, "single", motion)
    return " ".join(
        [
            frame_description(start_frame, "first", motion),
            frame_description(end_frame, "last", motion),
        ]
    )


class MetatokenBuilder:
    def __init__(self, scene_metadata_path, nuscenes_root):
        self.scenes = load_scene_metadata(scene_metadata_path)
        self.sample_data_by_filename = build_sample_data_by_filename(nuscenes_root)
        self.ego_pose_by_token = build_ego_pose_by_token(nuscenes_root)
        self._motion_cache = {}

    def scene_motion(self, scene_id):
        if scene_id not in self._motion_cache:
            scene = self.scenes[scene_id]
            self._motion_cache[scene_id] = load_scene_motion(
                scene, self.sample_data_by_filename, self.ego_pose_by_token
            )
        return self._motion_cache[scene_id]

    def build(self, scene_id, question):
        scene = self.scenes[scene_id]
        frame_indices = available_frame_indices(scene)
        start, end, reason = parse_question_frame_selection(question, frame_indices)
        motion = self.scene_motion(scene_id)
        text = build_metatoken_text(start, end, motion)
        return {
            "start_frame": start,
            "end_frame": end,
            "selection_reason": reason,
            "text": text,
        }

    def prompt(self, scene_id, question):
        metatoken = self.build(scene_id, question)
        return build_prompt(question, metatoken["text"]), metatoken


def build_prompt(question, metatoken_text):
    return f"{PROMPT_PREFIX}\n{metatoken_text}\n{question}"


def make_conversation(prompt, answer):
    return [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": answer},
    ]


def convert_row(row, builder, task=None, source_stage=None):
    scene_id = row.get("scene_id")
    question = row.get("question")
    answer = row.get("answer")
    if not scene_id or not question or answer is None:
        return None

    prompt, metatoken = builder.prompt(scene_id, question)
    item = {
        "source": "B4DL",
        "split": row.get("split"),
        "scene_token": row.get("scene_token"),
        "scene_id": scene_id,
        "human_annotation": row.get("human_annotation"),
        "question": question,
        "answer": answer,
        "prompt": prompt,
        "metatoken": metatoken,
        "conversations": make_conversation(prompt, answer),
    }
    if task is not None:
        item["task"] = task
    if source_stage is not None:
        item["source_stage"] = source_stage
    return item


def selection_counts(rows):
    return Counter(
        row.get("metatoken", {}).get("selection_reason", "missing") for row in rows
    )


def print_metatoken_report(rows, label, expected_count=None):
    print(f"{label}: wrote {len(rows)} rows")
    if expected_count is not None and len(rows) != expected_count:
        print(f"WARNING: expected {expected_count} rows, got {len(rows)}")
    print("metatoken selection counts:")
    for key, value in sorted(selection_counts(rows).items()):
        print(f"  {key}: {value}")
    if rows:
        prompt = rows[0]["conversations"][0]["value"]
        print(f"prompt starts with metatoken prefix: {prompt.startswith(PROMPT_PREFIX)}")
        print(f"feature placeholders in first prompt: {prompt.count('<video>')}")
        print(f"literal <4DLiDAR> present: {'<4DLiDAR>' in prompt}")
