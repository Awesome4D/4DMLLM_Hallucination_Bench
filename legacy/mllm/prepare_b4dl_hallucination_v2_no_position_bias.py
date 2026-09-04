import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from b4dl_metatoken import MetatokenBuilder, build_prompt

try:
    from nuscenes.nuscenes import NuScenes
except ImportError:
    NuScenes = None


CATEGORY_FILES = {
    "object_existence": "object_existence.json",
    "temporal_grounding": "temporal_grounding.json",
    "ego_relative_spatial": "ego_relative_spatial.json",
    "motion_action": "motion_action.json",
    "distance_depth": "distance_depth.json",
}

OBJECT_CLASSES = [
    "car",
    "truck",
    "bus",
    "pedestrian",
    "bicycle",
    "motorcycle",
    "trailer",
    "construction vehicle",
    "barrier",
    "traffic cone",
]

SECTORS = [
    "front",
    "front-left",
    "left",
    "back-left",
    "back",
    "back-right",
    "right",
    "front-right",
]

ACTION_LABELS = [
    "becomes closer to the ego vehicle",
    "moves farther away from the ego vehicle",
    "moves left relative to the ego vehicle",
    "moves right relative to the ego vehicle",
    "maintains a similar relative position",
]

ACTION_QUESTION_VERBS = {
    "becomes closer to the ego vehicle": "become closer to the ego vehicle",
    "moves farther away from the ego vehicle": "move farther away from the ego vehicle",
    "moves left relative to the ego vehicle": "shift left relative to the ego vehicle",
    "moves right relative to the ego vehicle": "shift right relative to the ego vehicle",
    "maintains a similar relative position": "maintain a similar relative position",
}

OPTION_LETTERS = ["A", "B", "C", "D"]
BINARY_ANSWER_DIRECTIVE = 'Only answer with "No" or "Yes".'
MCQ_ANSWER_DIRECTIVE = (
    "Please select the correct answer and only return the choice letter "
    "(i.e., A, B, C, D) of your answer. Do not favor any option based on "
    "its position; evaluate all options equally before selecting your answer."
)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def find_table_path(nuscenes_root, name):
    root = Path(nuscenes_root)
    candidates = [
        root / name,
        root / "v1.0-trainval" / name,
        root / "v1.0-mini" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {name} under {root}")


def normalize_nuscenes_dataroot(nuscenes_root, version):
    root = Path(nuscenes_root)
    if root.name == version:
        return root.parent
    return root


def load_table(nuscenes_root, name, key="token"):
    rows = load_json(find_table_path(nuscenes_root, name))
    return rows, {row[key]: row for row in rows}


def normalize_category(category_name):
    category_name = str(category_name or "")
    if category_name.startswith("vehicle.car"):
        return "car"
    if category_name.startswith("vehicle.truck"):
        return "truck"
    if category_name.startswith("vehicle.bus"):
        return "bus"
    if category_name.startswith("vehicle.bicycle"):
        return "bicycle"
    if category_name.startswith("vehicle.motorcycle"):
        return "motorcycle"
    if category_name.startswith("vehicle.trailer"):
        return "trailer"
    if category_name.startswith("vehicle.construction"):
        return "construction vehicle"
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    if category_name == "movable_object.barrier":
        return "barrier"
    if category_name == "movable_object.trafficcone":
        return "traffic cone"
    return None


def quat_to_rot(rotation):
    w, x, y, z = rotation
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def global_to_ego(global_xyz, ego_pose):
    translation = ego_pose["translation"]
    shifted = [global_xyz[i] - translation[i] for i in range(3)]
    rot = quat_to_rot(ego_pose["rotation"])
    # Inverse rotation is transpose for unit quaternion rotation matrices.
    return [
        rot[0][0] * shifted[0] + rot[1][0] * shifted[1] + rot[2][0] * shifted[2],
        rot[0][1] * shifted[0] + rot[1][1] * shifted[1] + rot[2][1] * shifted[2],
        rot[0][2] * shifted[0] + rot[1][2] * shifted[1] + rot[2][2] * shifted[2],
    ]


def sector_from_xy(x, y):
    angle = math.degrees(math.atan2(y, x))
    if -22.5 <= angle < 22.5:
        return "front"
    if 22.5 <= angle < 67.5:
        return "front-left"
    if 67.5 <= angle < 112.5:
        return "left"
    if 112.5 <= angle < 157.5:
        return "back-left"
    if angle >= 157.5 or angle < -157.5:
        return "back"
    if -157.5 <= angle < -112.5:
        return "back-right"
    if -112.5 <= angle < -67.5:
        return "right"
    return "front-right"


def parse_interval(text):
    text = str(text or "").lower()
    patterns = [
        r"from\s+frames?\s+(\d+)\s+to\s+(?:frame\s+)?(\d+)",
        r"from\s+(\d+)\s+to\s+(\d+)",
        r"between\s+frames?\s+(\d+)\s+and\s+(?:frame\s+)?(\d+)",
        r"frames?\s+(\d+)\s*[-\u2013]\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if start <= end:
                return start, end
    return None


def interval_iou(a, b):
    if not a or not b:
        return 0.0
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    inter = max(0, end - start)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def frame_label(frame_idx):
    return f"{frame_idx:03d}"


def interval_text(interval):
    return f"from frame {interval[0]:03d} to frame {interval[1]:03d}"


def with_mcq_options(question, options):
    option_lines = [f"{letter}. {option}" for letter, option in zip(OPTION_LETTERS, options)]
    return question.strip() + "\n" + "\n".join(option_lines)


def shuffled_options(rng, correct, distractors):
    options = list(distractors) + [correct]
    rng.shuffle(options)
    correct_letter = OPTION_LETTERS[options.index(correct)]
    return options, correct_letter


def append_answer_directive(question, question_type):
    question = question.strip()
    if question_type == "binary":
        directive = BINARY_ANSWER_DIRECTIVE
    elif question_type == "mcq":
        directive = MCQ_ANSWER_DIRECTIVE
    else:
        return question
    if directive in question:
        return question
    return question + "\n" + directive


def add_prompt_fields(item, metatoken_builder=None):
    answer = item["answer"]
    item["question"] = append_answer_directive(item["question"], item.get("question_type"))
    question = item["question"]
    if metatoken_builder is not None:
        metatoken = metatoken_builder.build(item["scene_id"], question)
        item["metatoken"] = metatoken
        item["prompt"] = build_prompt(question, metatoken["text"])
        item["conversations"] = [
            {"from": "human", "value": item["prompt"]},
            {"from": "gpt", "value": answer},
        ]
    else:
        item["prompt"] = "<video>\n" + question
        item["conversations"] = [
            {"from": "human", "value": item["prompt"]},
            {"from": "gpt", "value": answer},
        ]


class RawMetadataIndex:
    def __init__(self, nuscenes_root, scene_metadata_path, split):
        self.scene_rows = load_json(scene_metadata_path)
        if split != "all":
            self.scene_rows = [row for row in self.scene_rows if row.get("split") == split]
        self.scene_id_by_token = {row["scene_token"]: row["scene_id"] for row in self.scene_rows}
        self.scene_meta_by_id = {row["scene_id"]: row for row in self.scene_rows}

        _, self.raw_scene_by_token = load_table(nuscenes_root, "scene.json")
        _, self.sample_by_token = load_table(nuscenes_root, "sample.json")
        sample_data_rows, self.sample_data_by_token = load_table(nuscenes_root, "sample_data.json")
        annotation_rows, self.annotation_by_token = load_table(nuscenes_root, "sample_annotation.json")
        _, self.ego_pose_by_token = load_table(nuscenes_root, "ego_pose.json")
        _, self.calibrated_sensor_by_token = load_table(nuscenes_root, "calibrated_sensor.json")
        _, self.sensor_by_token = load_table(nuscenes_root, "sensor.json")
        _, self.instance_by_token = load_table(nuscenes_root, "instance.json")
        _, self.category_by_token = load_table(nuscenes_root, "category.json")
        self.category_name_by_instance = self._build_category_names()
        self.channel_by_sample_data_token = self._build_sample_data_channels(sample_data_rows)
        self.sample_data_by_sample_token = self._group_sample_data(sample_data_rows)
        self.annotations_by_sample_token = self._group_annotations(annotation_rows)

        self.frames = []
        self.frames_by_scene_id = defaultdict(dict)
        self._build_frames()

    def _build_category_names(self):
        names = {}
        for instance_token, instance in self.instance_by_token.items():
            category = self.category_by_token.get(instance.get("category_token"), {})
            if category.get("name"):
                names[instance_token] = category["name"]
        return names

    def _build_sample_data_channels(self, rows):
        channels = {}
        for row in rows:
            channel = row.get("channel")
            if not channel:
                calibrated = self.calibrated_sensor_by_token.get(
                    row.get("calibrated_sensor_token"), {}
                )
                sensor = self.sensor_by_token.get(calibrated.get("sensor_token"), {})
                channel = sensor.get("channel")
            if channel:
                channels[row["token"]] = channel
        return channels

    def _group_sample_data(self, rows):
        grouped = defaultdict(dict)
        for row in rows:
            sample_token = row.get("sample_token")
            channel = row.get("channel") or self.channel_by_sample_data_token.get(row.get("token"))
            if sample_token and channel:
                grouped[sample_token][channel] = row
        return grouped

    @staticmethod
    def _group_annotations(rows):
        grouped = defaultdict(list)
        for row in rows:
            sample_token = row.get("sample_token")
            if sample_token:
                grouped[sample_token].append(row)
        return grouped

    def _build_frames(self):
        for scene_meta in self.scene_rows:
            scene_token = scene_meta["scene_token"]
            raw_scene = self.raw_scene_by_token.get(scene_token)
            if raw_scene is None:
                continue
            sample_token = raw_scene["first_sample_token"]
            frame_idx = 0
            max_frames = int(scene_meta.get("num_frames") or 10**9)
            while sample_token and frame_idx < max_frames:
                sample = self.sample_by_token[sample_token]
                frame = self._frame_from_sample(scene_meta, frame_idx, sample)
                self.frames.append(frame)
                self.frames_by_scene_id[scene_meta["scene_id"]][frame_idx] = frame
                sample_token = sample.get("next")
                frame_idx += 1

    def _frame_from_sample(self, scene_meta, frame_idx, sample):
        sample_token = sample["token"]
        if "data" in sample and "LIDAR_TOP" in sample["data"]:
            lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_data = self.sample_data_by_token[lidar_token]
        else:
            lidar_data = self.sample_data_by_sample_token[sample_token].get("LIDAR_TOP")
            if lidar_data is None:
                raise KeyError(f"Missing LIDAR_TOP sample_data for sample {sample_token}")
        ego_pose = self.ego_pose_by_token[lidar_data["ego_pose_token"]]
        objects = []
        if "anns" in sample:
            annotations = [self.annotation_by_token[token] for token in sample.get("anns", [])]
        else:
            annotations = self.annotations_by_sample_token.get(sample_token, [])
        for ann in annotations:
            category_name = ann.get("category_name") or self.category_name_by_instance.get(
                ann.get("instance_token")
            )
            obj_class = normalize_category(category_name)
            if obj_class not in OBJECT_CLASSES:
                continue
            if ann.get("num_lidar_pts") is not None and ann.get("num_lidar_pts", 0) <= 0:
                continue
            x, y, z = global_to_ego(ann["translation"], ego_pose)
            distance = math.hypot(x, y)
            objects.append(
                {
                    "class": obj_class,
                    "instance_token": ann.get("instance_token"),
                    "annotation_token": ann.get("token"),
                    "x": x,
                    "y": y,
                    "z": z,
                    "distance": distance,
                    "sector": sector_from_xy(x, y),
                    "num_lidar_pts": ann.get("num_lidar_pts"),
                }
            )
        return {
            "scene_token": scene_meta["scene_token"],
            "scene_id": scene_meta["scene_id"],
            "frame": frame_idx,
            "sample_token": sample_token,
            "timestamp": sample.get("timestamp"),
            "objects": objects,
        }


class DevkitMetadataIndex:
    def __init__(self, nuscenes_root, scene_metadata_path, split, version):
        if NuScenes is None:
            raise ImportError("nuscenes-devkit is not installed.")
        self.scene_rows = load_json(scene_metadata_path)
        if split != "all":
            self.scene_rows = [row for row in self.scene_rows if row.get("split") == split]
        self.scene_id_by_token = {row["scene_token"]: row["scene_id"] for row in self.scene_rows}
        self.scene_meta_by_id = {row["scene_id"]: row for row in self.scene_rows}
        dataroot = normalize_nuscenes_dataroot(nuscenes_root, version)
        self.nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=True)
        self.frames = []
        self.frames_by_scene_id = defaultdict(dict)
        self._build_frames()

    def _build_frames(self):
        for scene_meta in self.scene_rows:
            scene_token = scene_meta["scene_token"]
            raw_scene = self.nusc.get("scene", scene_token)
            sample_token = raw_scene["first_sample_token"]
            frame_idx = 0
            max_frames = int(scene_meta.get("num_frames") or 10**9)
            while sample_token and frame_idx < max_frames:
                sample = self.nusc.get("sample", sample_token)
                frame = self._frame_from_sample(scene_meta, frame_idx, sample)
                self.frames.append(frame)
                self.frames_by_scene_id[scene_meta["scene_id"]][frame_idx] = frame
                sample_token = sample.get("next")
                frame_idx += 1

    def _frame_from_sample(self, scene_meta, frame_idx, sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = self.nusc.get("sample_data", lidar_token)
        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        objects = []
        for ann_token in sample.get("anns", []):
            ann = self.nusc.get("sample_annotation", ann_token)
            obj_class = normalize_category(ann.get("category_name"))
            if obj_class not in OBJECT_CLASSES:
                continue
            if ann.get("num_lidar_pts") is not None and ann.get("num_lidar_pts", 0) <= 0:
                continue
            x, y, z = global_to_ego(ann["translation"], ego_pose)
            distance = math.hypot(x, y)
            objects.append(
                {
                    "class": obj_class,
                    "instance_token": ann.get("instance_token"),
                    "annotation_token": ann.get("token"),
                    "x": x,
                    "y": y,
                    "z": z,
                    "distance": distance,
                    "sector": sector_from_xy(x, y),
                    "num_lidar_pts": ann.get("num_lidar_pts"),
                }
            )
        return {
            "scene_token": scene_meta["scene_token"],
            "scene_id": scene_meta["scene_id"],
            "frame": frame_idx,
            "sample_token": sample["token"],
            "timestamp": sample.get("timestamp"),
            "objects": objects,
        }


def build_metadata_index(args):
    if args.metadata_backend == "devkit":
        return DevkitMetadataIndex(
            args.nuscenes_root,
            args.scene_metadata,
            args.split,
            args.nuscenes_version,
        )
    if args.metadata_backend == "raw":
        return RawMetadataIndex(args.nuscenes_root, args.scene_metadata, args.split)
    try:
        return DevkitMetadataIndex(
            args.nuscenes_root,
            args.scene_metadata,
            args.split,
            args.nuscenes_version,
        )
    except Exception as exc:
        print(f"WARNING: devkit metadata loading failed, falling back to raw JSON: {exc}")
        return RawMetadataIndex(args.nuscenes_root, args.scene_metadata, args.split)


def base_item(category, question_type, frame):
    return {
        "source": "B4DL",
        "task": category,
        "category": category,
        "question_type": question_type,
        "split": "test",
        "scene_token": frame["scene_token"],
        "scene_id": frame["scene_id"],
        "frame": frame["frame"],
        "sample_token": frame["sample_token"],
    }


def generate_object_existence(index, rng, count, metatoken_builder=None):
    binary_count = count // 2
    mcq_count = count - binary_count
    yes_target = binary_count // 2
    no_target = binary_count - yes_target

    yes_candidates = []
    no_candidates = []
    mcq_candidates = []
    for frame in index.frames:
        present = sorted({obj["class"] for obj in frame["objects"]})
        absent = [obj for obj in OBJECT_CLASSES if obj not in present]
        if present:
            for obj_class in present:
                yes_candidates.append((frame, obj_class))
            if len(absent) >= 3:
                for obj_class in present:
                    mcq_candidates.append((frame, obj_class, absent))
        if absent:
            for obj_class in absent:
                no_candidates.append((frame, obj_class))

    rows = []
    for answer, candidates, target in [("Yes.", yes_candidates, yes_target), ("No.", no_candidates, no_target)]:
        for frame, obj_class in rng.sample(candidates, min(target, len(candidates))):
            question = f"Does {obj_class} exist in frame {frame_label(frame['frame'])}?"
            item = base_item("object_existence", "binary", frame)
            item.update({"question": question, "answer": answer, "object_class": obj_class})
            add_prompt_fields(item, metatoken_builder)
            rows.append(item)

    for frame, correct, absent in rng.sample(mcq_candidates, min(mcq_count, len(mcq_candidates))):
        distractors = rng.sample(absent, 3)
        options, correct_letter = shuffled_options(rng, correct, distractors)
        question = with_mcq_options(
            f"Which of the following object classes is visible in frame {frame_label(frame['frame'])}?",
            options,
        )
        item = base_item("object_existence", "mcq", frame)
        item.update(
            {
                "question": question,
                "answer": correct_letter,
                "options": dict(zip(OPTION_LETTERS, options)),
                "correct_option": correct_letter,
                "correct_answer": correct,
                "object_class": correct,
            }
        )
        add_prompt_fields(item, metatoken_builder)
        rows.append(item)
    rng.shuffle(rows)
    return rows, {"candidates": {"binary_yes": len(yes_candidates), "binary_no": len(no_candidates), "mcq": len(mcq_candidates)}}


def generate_temporal_grounding(time_rows, scene_meta_by_id, rng, count, metatoken_builder=None):
    candidates = []
    for idx, row in enumerate(time_rows):
        gt = parse_interval(row.get("answer"))
        scene = scene_meta_by_id.get(row.get("scene_id"))
        if gt is None or scene is None:
            continue
        max_frame = int(scene.get("num_frames", 40)) - 1
        if gt[0] < 0 or gt[1] > max_frame:
            continue
        candidates.append((idx, row, gt, max_frame))

    rows = []
    for source_idx, row, gt, max_frame in rng.sample(candidates, min(count, len(candidates))):
        distractors = []
        attempts = 0
        while len(distractors) < 3 and attempts < 1000:
            attempts += 1
            length = rng.randint(5, 19)
            if length > max_frame:
                continue
            start = rng.randint(0, max_frame - length)
            cand = (start, start + length)
            if cand == gt or cand in distractors:
                continue
            if interval_iou(cand, gt) <= 0.2:
                distractors.append(cand)
        if len(distractors) < 3:
            continue
        correct = interval_text(gt)
        options, correct_letter = shuffled_options(rng, correct, [interval_text(x) for x in distractors])
        question = with_mcq_options(row["question"], options)
        item = {
            "source": "B4DL",
            "task": "temporal_grounding",
            "category": "temporal_grounding",
            "question_type": "mcq",
            "split": row.get("split", "test"),
            "scene_token": row.get("scene_token"),
            "scene_id": row.get("scene_id"),
            "source_index": source_idx,
            "source_question": row.get("question"),
            "gt_interval": list(gt),
            "question": question,
            "answer": correct_letter,
            "options": dict(zip(OPTION_LETTERS, options)),
            "correct_option": correct_letter,
            "correct_answer": correct,
        }
        add_prompt_fields(item, metatoken_builder)
        rows.append(item)
    return rows, {"candidates": {"mcq": len(candidates)}}


def nearest_by_class(frame):
    by_class = {}
    for obj in frame["objects"]:
        current = by_class.get(obj["class"])
        if current is None or obj["distance"] < current["distance"]:
            by_class[obj["class"]] = obj
    return by_class


def generate_ego_relative_spatial(index, rng, count, metatoken_builder=None):
    binary_count = count // 2
    mcq_count = count - binary_count
    yes_target = binary_count // 2
    no_target = binary_count - yes_target

    candidates = []
    for frame in index.frames:
        for obj_class, obj in nearest_by_class(frame).items():
            candidates.append((frame, obj_class, obj))

    rows = []
    for frame, obj_class, obj in rng.sample(candidates, min(yes_target, len(candidates))):
        position = obj["sector"]
        question = (
            f"In frame {frame_label(frame['frame'])}, is the nearest {obj_class} at the "
            f"{position} position relative to the ego vehicle?"
        )
        item = base_item("ego_relative_spatial", "binary", frame)
        item.update({"question": question, "answer": "Yes.", "object_class": obj_class, "position": position})
        add_prompt_fields(item, metatoken_builder)
        rows.append(item)

    no_source = rng.sample(candidates, min(no_target, len(candidates)))
    for frame, obj_class, obj in no_source:
        wrong_position = rng.choice([sector for sector in SECTORS if sector != obj["sector"]])
        question = (
            f"In frame {frame_label(frame['frame'])}, is the nearest {obj_class} at the "
            f"{wrong_position} position relative to the ego vehicle?"
        )
        item = base_item("ego_relative_spatial", "binary", frame)
        item.update(
            {
                "question": question,
                "answer": "No.",
                "object_class": obj_class,
                "position": wrong_position,
                "correct_position": obj["sector"],
            }
        )
        add_prompt_fields(item, metatoken_builder)
        rows.append(item)

    for frame, obj_class, obj in rng.sample(candidates, min(mcq_count, len(candidates))):
        correct = obj["sector"]
        distractors = rng.sample([sector for sector in SECTORS if sector != correct], 3)
        options, correct_letter = shuffled_options(rng, correct, distractors)
        question = with_mcq_options(
            f"In frame {frame_label(frame['frame'])}, at which position is the nearest {obj_class} relative to the ego vehicle?",
            options,
        )
        item = base_item("ego_relative_spatial", "mcq", frame)
        item.update(
            {
                "question": question,
                "answer": correct_letter,
                "options": dict(zip(OPTION_LETTERS, options)),
                "correct_option": correct_letter,
                "correct_answer": correct,
                "object_class": obj_class,
                "position": correct,
            }
        )
        add_prompt_fields(item, metatoken_builder)
        rows.append(item)
    rng.shuffle(rows)
    return rows, {"candidates": {"binary_mcq": len(candidates)}}


def action_truth(start_obj, end_obj):
    distance_delta = end_obj["distance"] - start_obj["distance"]
    lateral_delta = end_obj["y"] - start_obj["y"]
    truths = {
        "becomes closer to the ego vehicle": distance_delta <= -2.0,
        "moves farther away from the ego vehicle": distance_delta >= 2.0,
        "moves left relative to the ego vehicle": lateral_delta >= 1.0,
        "moves right relative to the ego vehicle": lateral_delta <= -1.0,
        "maintains a similar relative position": abs(distance_delta) < 1.0 and abs(lateral_delta) < 0.5,
    }
    return truths, distance_delta, lateral_delta


def object_by_instance(frame):
    return {obj["instance_token"]: obj for obj in frame["objects"] if obj.get("instance_token")}


def generate_motion_action(index, rng, count, metatoken_builder=None):
    yes_target = count // 2
    no_target = count - yes_target
    yes_candidates = []
    no_candidates = []

    for scene_id, frames_by_idx in index.frames_by_scene_id.items():
        frame_indices = sorted(frames_by_idx)
        if len(frame_indices) < 9:
            continue
        for start in frame_indices:
            end = start + 8
            if end not in frames_by_idx:
                continue
            start_frame = frames_by_idx[start]
            end_frame = frames_by_idx[end]
            end_by_instance = object_by_instance(end_frame)
            for obj_class, start_obj in nearest_by_class(start_frame).items():
                end_obj = end_by_instance.get(start_obj.get("instance_token"))
                if end_obj is None or end_obj["class"] != obj_class:
                    continue
                truths, distance_delta, lateral_delta = action_truth(start_obj, end_obj)
                for action, is_true in truths.items():
                    candidate = (start_frame, end_frame, obj_class, action, distance_delta, lateral_delta)
                    if is_true:
                        yes_candidates.append(candidate)
                    else:
                        no_candidates.append(candidate)

    rows = []
    for answer, candidates, target in [("Yes.", yes_candidates, yes_target), ("No.", no_candidates, no_target)]:
        for start_frame, end_frame, obj_class, action, distance_delta, lateral_delta in rng.sample(candidates, min(target, len(candidates))):
            question = (
                f"During frames {frame_label(start_frame['frame'])}-{frame_label(end_frame['frame'])}, "
                f"does the nearest {obj_class} {ACTION_QUESTION_VERBS[action]}?"
            )
            item = base_item("motion_action", "binary", start_frame)
            item.update(
                {
                    "question": question,
                    "answer": answer,
                    "start_frame": start_frame["frame"],
                    "end_frame": end_frame["frame"],
                    "object_class": obj_class,
                    "action": action,
                    "distance_delta": distance_delta,
                    "lateral_delta": lateral_delta,
                }
            )
            add_prompt_fields(item, metatoken_builder)
            rows.append(item)
    rng.shuffle(rows)
    return rows, {"candidates": {"binary_yes": len(yes_candidates), "binary_no": len(no_candidates)}}


def generate_distance_depth(index, rng, count, metatoken_builder=None):
    candidates = []
    for frame in index.frames:
        nearest = nearest_by_class(frame)
        if len(nearest) < 4:
            continue
        ordered = sorted(nearest.items(), key=lambda kv: kv[1]["distance"])
        correct_class, correct_obj = ordered[0]
        distractor_classes = [obj_class for obj_class, _ in ordered[1:]]
        if len(distractor_classes) >= 3:
            candidates.append((frame, correct_class, correct_obj, distractor_classes))

    rows = []
    for frame, correct_class, correct_obj, distractor_classes in rng.sample(candidates, min(count, len(candidates))):
        distractors = rng.sample(distractor_classes, 3)
        options, correct_letter = shuffled_options(rng, correct_class, distractors)
        question = with_mcq_options(
            f"In frame {frame_label(frame['frame'])}, which object class is closest to the ego vehicle?",
            options,
        )
        item = base_item("distance_depth", "mcq", frame)
        item.update(
            {
                "question": question,
                "answer": correct_letter,
                "options": dict(zip(OPTION_LETTERS, options)),
                "correct_option": correct_letter,
                "correct_answer": correct_class,
                "distance_meters": correct_obj["distance"],
            }
        )
        add_prompt_fields(item, metatoken_builder)
        rows.append(item)
    return rows, {"candidates": {"mcq": len(candidates)}}


def category_stats(rows):
    stats = {
        "count": len(rows),
        "question_type_counts": dict(Counter(row.get("question_type") for row in rows)),
        "correct_option_counts": dict(Counter(row.get("correct_option") for row in rows if row.get("correct_option"))),
        "object_class_counts": dict(Counter(row.get("object_class") for row in rows if row.get("object_class"))),
        "answer_counts": dict(Counter(row.get("answer") for row in rows)),
    }
    if rows and any(row.get("position") for row in rows):
        stats["sector_counts"] = dict(Counter(row.get("position") for row in rows if row.get("position")))
    if rows and any(row.get("action") for row in rows):
        stats["action_counts"] = dict(Counter(row.get("action") for row in rows if row.get("action")))
    return stats


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes-root", default=os.environ.get("NUSCENES_ROOT"))
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument(
        "--metadata-backend",
        default="devkit",
        choices=["devkit", "raw", "auto"],
        help="Use nuScenes devkit metadata access, raw JSON access, or devkit with raw fallback.",
    )
    parser.add_argument("--scene-metadata", default=str(repo_root / "nuScenes-B4DL" / "metadata" / "scene_metadata.json"))
    parser.add_argument("--time-grounding-json", default=str(repo_root / "mllm" / "b4dl_dataset" / "test" / "time_grounding.json"))
    parser.add_argument("--output-dir", default=str(repo_root / "mllm" / "b4dl_dataset" / "hallucination_v2_no_position_bias"))
    parser.add_argument(
        "--metatoken-output-dir",
        default=str(repo_root / "mllm" / "b4dl_dataset" / "hallucination_v2_no_position_bias_metatoken"),
    )
    parser.add_argument("--samples-per-category", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--with-metatoken", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.nuscenes_root:
        raise ValueError("Set NUSCENES_ROOT or pass --nuscenes-root.")

    index = build_metadata_index(args)
    time_rows = load_json(args.time_grounding_json)

    output_specs = [(Path(args.output_dir), None)]
    if args.with_metatoken:
        builder = MetatokenBuilder(args.scene_metadata, args.nuscenes_root)
        output_specs.append((Path(args.metatoken_output_dir), builder))

    for output_dir, metatoken_builder in output_specs:
        rng = random.Random(args.seed)
        datasets = {}
        generation_stats = {}
        datasets["object_existence"], generation_stats["object_existence"] = generate_object_existence(
            index, rng, args.samples_per_category, metatoken_builder
        )
        datasets["temporal_grounding"], generation_stats["temporal_grounding"] = generate_temporal_grounding(
            time_rows, index.scene_meta_by_id, rng, args.samples_per_category, metatoken_builder
        )
        datasets["ego_relative_spatial"], generation_stats["ego_relative_spatial"] = generate_ego_relative_spatial(
            index, rng, args.samples_per_category, metatoken_builder
        )
        datasets["motion_action"], generation_stats["motion_action"] = generate_motion_action(
            index, rng, args.samples_per_category, metatoken_builder
        )
        datasets["distance_depth"], generation_stats["distance_depth"] = generate_distance_depth(
            index, rng, args.samples_per_category, metatoken_builder
        )

        summary = {
            "samples_per_category_requested": args.samples_per_category,
            "seed": args.seed,
            "split": args.split,
            "nuscenes_root": str(args.nuscenes_root),
            "scene_metadata": str(args.scene_metadata),
            "metatoken": metatoken_builder is not None,
            "sector_definition": {
                "coordinate_frame": "ego frame; x forward, y left",
                "sectors": SECTORS,
                "bin_width_degrees": 45,
            },
            "generation_stats": generation_stats,
            "category_stats": {},
        }
        for category, rows in datasets.items():
            save_json(output_dir / CATEGORY_FILES[category], rows)
            summary["category_stats"][category] = category_stats(rows)
            print(f"{output_dir}: wrote {len(rows)} {category} samples")
        save_json(output_dir / "summary.json", summary)
        print(f"Wrote summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
