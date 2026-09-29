"""Workstation diagnostics layered over the existing detection overlay."""
import cv2
from .overlay import draw_detections


def draw_network_frame(frame, processed, frame_id, rx_fps):
    output = draw_detections(frame, processed.detections, rx_fps)
    lines = [f'Frame: {frame_id}  Backend: {processed.backend.upper()}',
             f'RX FPS: {rx_fps:.1f}  Processing: {processed.processing_ms:.2f} ms']
    for index, target in enumerate(processed.detections, 1):
        lines.append(f'#{index} {target.detected_color.upper()} normalized=({target.centroid_norm_x:.3f}, {target.centroid_norm_y:.3f})')
    # Limit the status panel to available vertical space; target geometry stays visible.
    for index, line in enumerate(lines[:max(1, (frame.shape[0]-60)//19)]):
        cv2.putText(output, line, (12, 51+19*index), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0,0,0), 3)
        cv2.putText(output, line, (12, 51+19*index), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255,255,255), 1)
    return output
