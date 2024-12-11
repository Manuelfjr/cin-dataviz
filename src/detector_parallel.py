from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import supervision as sv
import ultralytics
from ultralytics import YOLO
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

ultralytics.checks()

class VideoProcessor:
    def __init__(self, model_path, source_paths, target_paths, save_paths=None):
        self.model_path = model_path
        self.source_paths = source_paths
        self.target_paths = target_paths
        self.save_paths = save_paths
        self.data_tracker = {}
        self.model = YOLO(self.model_path)  # Inicialize o modelo aqui

    def callback(self, frame: np.ndarray, _: int, model: YOLO, tracker: sv.ByteTrack, annotator: sv.RoundBoxAnnotator, label_annotator: sv.LabelAnnotator, trace_annotator: sv.TraceAnnotator) -> np.ndarray:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        
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

        annotated_frame = annotator.annotate(frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        return trace_annotator.annotate(annotated_frame, detections=detections)

    def callback_wrapper(self, frame, index, model, tracker, annotator, label_annotator, trace_annotator):
        return self.callback(frame, index, model, tracker, annotator, label_annotator, trace_annotator)

    def process_single_video(self, source_name, source_path, target_path, idx, model_path, **kwargs):
        self.detections = pd.DataFrame(columns=["tracker_id", "class_id", "x_min", "y_min", "x_max", "y_max"])
        self.tracker = sv.ByteTrack()
        self.annotator = sv.RoundBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        
        sv.process_video(
            source_path=source_path,
            target_path=target_path,
            callback=lambda frame, index: self.callback_wrapper(frame, index, self.model, self.tracker, self.annotator, self.label_annotator, self.trace_annotator),
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
        return source_name, self.detections

    def process_video(self, use_parallel: bool = False, **kwargs) -> Dict[str, pd.DataFrame]:
        if use_parallel:
            num_cores = multiprocessing.cpu_count()
            with ProcessPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(self.process_single_video, source_name, source_path, target_path, idx, self.model_path, **kwargs)
                    for idx, ((source_name, source_path), target_path) in enumerate(zip(self.source_paths.items(), self.target_paths))
                ]
                for future in as_completed(futures):
                    source_name, detections = future.result()
                    self.data_tracker[source_name] = detections
        else:
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
                print(self.data_tracker[source_name])
                if self.save_paths:
                    self.data_tracker[source_name].to_parquet(self.save_paths[idx], index=False)
        return self.data_tracker