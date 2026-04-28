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


def estimate_court_keypoints(frame):
    """Estimate 14 court keypoints from a video frame using classical CV."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    row_sum = white.sum(axis=1).astype(float)
    # Smooth and find peaks
    kernel = np.ones(5) / 5
    smoothed = np.convolve(row_sum, kernel, mode='same')
    threshold = smoothed.max() * 0.15
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > threshold and smoothed[i] >= smoothed[i-1] and smoothed[i] >= smoothed[i+1]:
            if not peaks or i - peaks[-1] > 20:
                peaks.append(i)

    # Find x extents at each candidate row
    h_lines = []
    for y in peaks:
        xs = np.where(white[y] > 0)[0]
        if len(xs) > w * 0.2:
            h_lines.append((y, int(xs.min()), int(xs.max())))

    if len(h_lines) < 2:
        # Fallback: use frame fractions
        h_lines = [
            (int(h * 0.28), int(w * 0.30), int(w * 0.70)),
            (int(h * 0.79), int(w * 0.19), int(w * 0.81)),
        ]

    h_lines.sort(key=lambda x: x[0])
    y_far, x_far_l, x_far_r = h_lines[0]
    y_near, x_near_l, x_near_r = h_lines[-1]

    # Estimate net and service line y positions using court proportions
    y_net = int(y_far + (y_near - y_far) * (11.88 / (11.88 * 2)))
    y_far_svc = int(y_far + (y_net - y_far) * (5.48 / 11.88))
    y_near_svc = int(y_near - (y_near - y_net) * (5.48 / 11.88))

    def interp_x(y, y0, x0, y1, x1):
        if y1 == y0:
            return x0
        return int(x0 + (x1 - x0) * (y - y0) / (y1 - y0))

    def singles_inset(x_l, x_r):
        return int((x_r - x_l) * (1.37 / 10.97))

    # Doubles corners
    p0 = (x_far_l, y_far)
    p1 = (x_far_r, y_far)
    p2 = (x_near_l, y_near)
    p3 = (x_near_r, y_near)

    # Singles corners
    inset_far = singles_inset(x_far_l, x_far_r)
    inset_near = singles_inset(x_near_l, x_near_r)
    p4 = (x_far_l + inset_far, y_far)
    p6 = (x_far_r - inset_far, y_far)
    p5 = (x_near_l + inset_near, y_near)
    p7 = (x_near_r - inset_near, y_near)

    # Service line points (on singles sidelines)
    p8x = interp_x(y_far_svc, y_far, p4[0], y_near, p5[0])
    p9x = interp_x(y_far_svc, y_far, p6[0], y_near, p7[0])
    p10x = interp_x(y_near_svc, y_far, p4[0], y_near, p5[0])
    p11x = interp_x(y_near_svc, y_far, p6[0], y_near, p7[0])
    p8 = (p8x, y_far_svc)
    p9 = (p9x, y_far_svc)
    p10 = (p10x, y_near_svc)
    p11 = (p11x, y_near_svc)

    # Service T marks
    p12 = ((p8x + p9x) // 2, y_far_svc)
    p13 = ((p10x + p11x) // 2, y_near_svc)

    points = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13]
    return np.array([v for p in points for v in p], dtype=float)


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
