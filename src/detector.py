from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import supervision as sv

import ultralytics
from ultralytics import YOLO

ultralytics.checks()


class VideoProcessor:
    def __init__(self, model_path: str, source_paths: Dict[str, pd.DataFrame], target_paths: List[str], save_paths: Optional[List[str]]=None):
        self.model = YOLO(model_path)
        self.source_paths = source_paths
        self.target_paths = target_paths
        self.save_paths = save_paths
        self.data_tracker = {}

    def callback(self, frame: np.ndarray, _: int) -> np.ndarray:
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)
        
        new_data = []
        for class_id, tracker_id, box in zip(detections.class_id, detections.tracker_id, detections.xyxy):
            x_min, y_min, x_max, y_max = box
            new_data.append({
                "tracker_id": tracker_id,
                "class_id": class_id,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            })

        self.detections = pd.concat([self.detections, pd.DataFrame(new_data)], ignore_index=True)

        labels = [
            f"#{tracker_id} {results.names[class_id]}"
            for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
        ]

        annotated_frame = self.annotator.annotate(frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        return self.trace_annotator.annotate(annotated_frame, detections=detections)

    def process_video(self, **kwargs) -> Dict[str, pd.DataFrame]:
        for idx, ((source_name, source_path), target_path) in enumerate(zip(self.source_paths.items(), self.target_paths)):
            self.detections = pd.DataFrame(columns=["tracker_id", "class_id", "x_min", "y_min", "x_max", "y_max"])
            self.tracker = sv.ByteTrack()
            self.annotator = sv.RoundBoxAnnotator()
            self.label_annotator = sv.LabelAnnotator()
            self.trace_annotator = sv.TraceAnnotator()
            sv.process_video(
                source_path=source_path,
                target_path=target_path,
                callback=self.callback,
                **kwargs
            )
            self.data_tracker[source_name] = self.detections
            self.data_tracker[source_name]["tracker_id"] = self.data_tracker[source_name]["tracker_id"].replace(
                {
                    id_tracker: idx2 for idx2, id_tracker in enumerate(self.data_tracker[source_name]["tracker_id"].unique())
                }
            )
            if self.save_paths:
                self.data_tracker[source_name].to_parquet(self.save_paths[idx], index=False)
        return self.data_tracker
