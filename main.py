from utils import (read_video,
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import numpy as np
import pandas as pd
from copy import deepcopy
import pickle
import os


def interpolate_player_detections(player_detections):
    """Fill frames where a player was temporarily lost using linear interpolation."""
    import pandas as pd
    all_ids = {pid for frame in player_detections for pid in frame}
    for pid in all_ids:
        coords = [[frame[pid][0], frame[pid][1], frame[pid][2], frame[pid][3]]
                  if pid in frame else [None, None, None, None]
                  for frame in player_detections]
        df = pd.DataFrame(coords, columns=['x1', 'y1', 'x2', 'y2']).astype(float)
        df = df.interpolate().bfill().ffill()
        for i, row in df.iterrows():
            if pid not in player_detections[i]:
                player_detections[i][pid] = row.tolist()
    return player_detections


def _keypoints_from_corners(p0, p1, p2, p3):
    """Compute all 14 tennis court keypoints from the 4 doubles corners.

    Convention (matches CourtLineDetector training data and MiniCourt):
      0=far-left  1=far-right  2=near-left  3=near-right  (doubles corners)
      4=far-left-singles  5=near-left-singles
      6=far-right-singles 7=near-right-singles
      8=far-left-service  9=far-right-service
      10=near-left-service 11=near-right-service
      12=far-T  13=near-T
    """
    t_alley = 1.37 / 10.97   # alley fraction of doubles width

    # Court-depth fractions: 0=far baseline, 1=near baseline
    t_far_svc  = 5.48 / 23.76  # NO_MANS_LAND_HEIGHT / full court
    t_near_svc = 1 - t_far_svc

    def lerp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    # Points along the far and near baselines
    p4 = lerp(p0, p1, t_alley)            # far-left singles
    p6 = lerp(p0, p1, 1 - t_alley)        # far-right singles
    p5 = lerp(p2, p3, t_alley)            # near-left singles (note: p2=near-left, p3=near-right)
    p7 = lerp(p2, p3, 1 - t_alley)        # near-right singles

    # Left and right singles sidelines parameterised by depth fraction t (0=far,1=near)
    def row(t):
        left  = lerp(p0, p2, t)            # doubles left sideline
        right = lerp(p1, p3, t)            # doubles right sideline
        sl    = lerp(left, right, t_alley)        # singles left at this depth
        sr    = lerp(left, right, 1 - t_alley)    # singles right at this depth
        return sl, sr

    p8,  p9  = row(t_far_svc)
    p10, p11 = row(t_near_svc)
    p12 = lerp(p8,  p9,  0.5)
    p13 = lerp(p10, p11, 0.5)

    pts = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13]
    return np.array([v for p in pts for v in p], dtype=float)


def _estimate_keypoints_blue_court(frame):
    """Detect court keypoints by isolating the blue court surface.

    Works for indoor blue hard courts shot from any angle.
    Returns None if the court surface cannot be reliably found.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Isolate the blue court surface
    blue_mask = cv2.inRange(hsv, (95, 55, 55), (135, 255, 255))
    # Ignore the top third — ceiling lights / reflections
    blue_mask[:int(h * 0.33), :] = 0

    kernel = np.ones((7, 7), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN,  kernel, iterations=2)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    court = max(contours, key=cv2.contourArea)
    if cv2.contourArea(court) < 0.05 * h * w:
        return None  # too small — probably not the court

    # Approximate to a simple polygon and find the 4 extreme corners
    eps = 0.02 * cv2.arcLength(court, True)
    approx = cv2.approxPolyDP(court, eps, True).reshape(-1, 2).astype(float)

    # Classify: top-left, top-right, bottom-left, bottom-right
    cx, cy = approx.mean(axis=0)
    tl = min(approx, key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2 if p[0] < cx and p[1] < cy else float('inf'))
    tr = min(approx, key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2 if p[0] > cx and p[1] < cy else float('inf'))
    bl = min(approx, key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2 if p[0] < cx and p[1] > cy else float('inf'))
    br = min(approx, key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2 if p[0] > cx and p[1] > cy else float('inf'))

    # If quadrant approach yields duplicates fall back to bounding-box corners
    if len({id(x) for x in [tl, tr, bl, br]}) < 4:
        xs, ys = approx[:, 0], approx[:, 1]
        tl = approx[np.argmin(xs + ys)]
        tr = approx[np.argmin(-xs + ys)]
        bl = approx[np.argmin(xs - ys)]
        br = approx[np.argmax(xs + ys)]

    # p0=far-left, p1=far-right (smaller y = higher in image = far end)
    if tl[1] < bl[1]:
        p0, p1, p2, p3 = tuple(tl), tuple(tr), tuple(bl), tuple(br)
    else:
        p0, p1, p2, p3 = tuple(bl), tuple(br), tuple(tl), tuple(tr)

    return _keypoints_from_corners(p0, p1, p2, p3)


def _estimate_keypoints_white_lines(frame):
    """Original white-line row-sum approach — works well for broadcast footage."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    row_sum = white.sum(axis=1).astype(float)
    kernel = np.ones(5) / 5
    smoothed = np.convolve(row_sum, kernel, mode='same')
    threshold = smoothed.max() * 0.15
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > threshold and smoothed[i] >= smoothed[i-1] and smoothed[i] >= smoothed[i+1]:
            if not peaks or i - peaks[-1] > 20:
                peaks.append(i)

    h_lines = []
    for y in peaks:
        xs = np.where(white[y] > 0)[0]
        if len(xs) > w * 0.2:
            h_lines.append((y, int(xs.min()), int(xs.max())))

    if len(h_lines) < 2:
        h_lines = [
            (int(h * 0.28), int(w * 0.30), int(w * 0.70)),
            (int(h * 0.79), int(w * 0.19), int(w * 0.81)),
        ]

    h_lines.sort(key=lambda x: x[0])
    y_far,  x_far_l,  x_far_r  = h_lines[0]
    y_near, x_near_l, x_near_r = h_lines[-1]

    p0 = (float(x_far_l),  float(y_far))
    p1 = (float(x_far_r),  float(y_far))
    p2 = (float(x_near_l), float(y_near))
    p3 = (float(x_near_r), float(y_near))
    return _keypoints_from_corners(p0, p1, p2, p3)


def estimate_court_keypoints(frame):
    """Auto-select the best court detection strategy for this frame."""
    kp = _estimate_keypoints_blue_court(frame)
    if kp is not None:
        print("Court detection: blue-court surface method")
        return kp
    print("Court detection: white-line method")
    return _estimate_keypoints_white_lines(frame)


def analyze_video(input_video_path, output_video_path, player_tracker, ball_tracker):
    print(f"\n=== Analyzing: {input_video_path} ===")
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    stub_dir = "tracker_stubs"

    video_frames = read_video(input_video_path)

    # Player detection
    player_stub = os.path.join(stub_dir, f"{video_name}_player_detections.pkl")
    use_player_stub = video_name == "input_video"  # only original has pre-built stub
    if use_player_stub:
        player_stub = os.path.join(stub_dir, "player_detections.pkl")
    player_detections = player_tracker.detect_frames(
        video_frames, read_from_stub=use_player_stub, stub_path=player_stub
    )

    # Ball detection
    ball_stub = os.path.join(stub_dir, f"{video_name}_ball_detections.pkl")
    use_ball_stub = video_name == "input_video"
    if use_ball_stub:
        ball_stub = os.path.join(stub_dir, "ball_detections.pkl")
    ball_detections = ball_tracker.detect_frames(
        video_frames, read_from_stub=use_ball_stub, stub_path=ball_stub
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court keypoints
    kp_stub = os.path.join(stub_dir, f"{video_name}_court_keypoints.pkl")
    if os.path.exists(kp_stub):
        with open(kp_stub, 'rb') as f:
            court_keypoints = pickle.load(f)
    else:
        court_keypoints = estimate_court_keypoints(video_frames[0])
        with open(kp_stub, 'wb') as f:
            pickle.dump(court_keypoints, f)

    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Remap track IDs to stable 1/2 so downstream code works regardless of tracker ID
    chosen_ids = sorted({pid for frame in player_detections for pid in frame})
    id_map = {old: new for new, old in enumerate(chosen_ids, start=1)}
    player_detections = [{id_map[pid]: bbox for pid, bbox in frame.items()} for frame in player_detections]

    # Interpolate missing player positions (fill gaps where tracker temporarily loses a player)
    player_detections = interpolate_player_detections(player_detections)

    mini_court = MiniCourt(video_frames[0])
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    player_mini_court_detections, ball_mini_court_detections = \
        mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections, ball_detections, court_keypoints
        )

    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,
        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]

    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        # Skip if either frame is missing both players or ball
        if len(player_mini_court_detections[start_frame]) < 2 or \
           len(player_mini_court_detections[end_frame]) < 2 or \
           1 not in ball_mini_court_detections[start_frame] or \
           1 not in ball_mini_court_detections[end_frame]:
            continue
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24

        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1]
        )
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(
            player_positions.keys(),
            key=lambda pid: measure_distance(player_positions[pid], ball_mini_court_detections[start_frame][1])
        )

        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id]
        )
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )
        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent
        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()
    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_1_number_of_shots']

    # Draw output
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    for frame in output_video_frames:
        for i in range(0, len(court_keypoints), 2):
            x, y = int(court_keypoints[i]), int(court_keypoints[i+1])
            cv2.putText(frame, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, output_video_path)
    print(f"Saved: {output_video_path}")


def main():
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')

    videos = [
        ("input_videos/input_video.mp4",   "output_videos/output_video.avi"),
        ("input_videos/tennis_match1.mp4", "output_videos/tennis_match1_output.avi"),
        ("input_videos/tennis_sarp.mp4",   "output_videos/tennis_sarp_output.avi"),
    ]

    for input_path, output_path in videos:
        analyze_video(input_path, output_path, player_tracker, ball_tracker)


if __name__ == "__main__":
    main()
