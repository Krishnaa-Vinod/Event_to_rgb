#!/usr/bin/env python3
"""
Export raw events from ROS bags into E2VID-compatible format.

This script:
1. Reads MCAP/bag files using the source pipeline's bag_reader and evt3_decoder
2. Extracts raw events and saves them as text files for E2VID/FireNet
3. Exports corresponding RGB frames with timestamps for evaluation
4. Handles EVT3 decoding and maintains temporal continuity
"""

import os
import sys
import argparse
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm

# Add source pipeline path for imports
sys.path.append('/scratch/kvinod/eventnavpp_datagen_pipeline')
from bag_reader import read_bag, deserialize, get_topics
from evt3_decoder import EVT3StreamDecoder

def export_events_from_bag(
    bag_dir: str,
    output_dir: str,
    event_topic: str = "/event_camera/events",
    rgb_topic: str = "/cam_sync/cam0/image_raw",
    max_duration_s: float = None,
    target_fps: float = 30.0
):
    """
    Export events and RGB frames from a ROS bag.

    Args:
        bag_dir: Path to ROS bag directory
        output_dir: Output directory for exported data
        event_topic: Event camera topic name
        rgb_topic: RGB camera topic name
        max_duration_s: Maximum duration to process (None = all)
        target_fps: Target framerate for RGB export

    Returns:
        dict: Export summary with counts and paths
    """

    print(f"=== Exporting Events from {bag_dir} ===")

    # Create output directories
    output_path = Path(output_dir)
    events_dir = output_path / "events"
    rgb_dir = output_path / "rgb"
    events_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    # Initialize decoder
    evt3_decoder = EVT3StreamDecoder(width=1280, height=720, verbose=True)

    # Storage for events and RGB frames
    all_events = []
    rgb_frames = []

    # Get topic info
    topics = get_topics(bag_dir)
    print(f"Available topics: {list(topics.keys())}")

    if event_topic not in topics:
        print(f"Warning: Event topic '{event_topic}' not found in bag")
        return None

    if rgb_topic not in topics:
        print(f"Warning: RGB topic '{rgb_topic}' not found in bag")
        return None

    # Process messages
    print("Processing bag messages...")
    event_count = 0
    rgb_count = 0
    start_time_ns = None

    for msg in tqdm(read_bag(bag_dir, topics={event_topic, rgb_topic})):

        # Track timing
        if start_time_ns is None:
            start_time_ns = msg.timestamp

        elapsed_s = (msg.timestamp - start_time_ns) / 1e9
        if max_duration_s and elapsed_s > max_duration_s:
            print(f"Reached max duration {max_duration_s}s, stopping")
            break

        if msg.topic == event_topic:
            # Decode event packet
            try:
                event_msg = deserialize(msg)
                raw_events = bytes(event_msg.events)

                # Decode EVT3 events
                events = evt3_decoder.decode(raw_events)

                if len(events) > 0:
                    # Convert timestamps from microseconds to seconds (E2VID format)
                    events_for_export = events.copy()
                    events_for_export[:, 2] = events[:, 2] / 1e6  # µs -> s
                    all_events.append(events_for_export)
                    event_count += len(events)

            except Exception as e:
                print(f"Error decoding event packet: {e}")
                continue

        elif msg.topic == rgb_topic:
            # Decode RGB image
            try:
                img_msg = deserialize(msg)

                # Convert ROS image to OpenCV format
                if img_msg.encoding == "rgb8":
                    img_array = np.frombuffer(img_msg.data, dtype=np.uint8)
                    img = img_array.reshape(img_msg.height, img_msg.width, 3)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif img_msg.encoding == "bgr8":
                    img_array = np.frombuffer(img_msg.data, dtype=np.uint8)
                    img_bgr = img_array.reshape(img_msg.height, img_msg.width, 3)
                elif img_msg.encoding == "bayer_rggb8":
                    # Handle Bayer pattern (common in event cameras)
                    img_array = np.frombuffer(img_msg.data, dtype=np.uint8)
                    bayer_img = img_array.reshape(img_msg.height, img_msg.width)
                    # Debayer to BGR using OpenCV
                    img_bgr = cv2.cvtColor(bayer_img, cv2.COLOR_BayerRG2BGR)
                elif img_msg.encoding in ["bayer_grbg8", "bayer_gbrg8", "bayer_bggr8"]:
                    # Handle other Bayer patterns
                    img_array = np.frombuffer(img_msg.data, dtype=np.uint8)
                    bayer_img = img_array.reshape(img_msg.height, img_msg.width)
                    if img_msg.encoding == "bayer_grbg8":
                        img_bgr = cv2.cvtColor(bayer_img, cv2.COLOR_BayerGR2BGR)
                    elif img_msg.encoding == "bayer_gbrg8":
                        img_bgr = cv2.cvtColor(bayer_img, cv2.COLOR_BayerGB2BGR)
                    elif img_msg.encoding == "bayer_bggr8":
                        img_bgr = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGR)
                else:
                    print(f"Unsupported encoding: {img_msg.encoding}")
                    continue

                # Store RGB frame with timestamp
                timestamp_s = msg.timestamp / 1e9
                rgb_frames.append({
                    'timestamp_s': timestamp_s,
                    'timestamp_ns': msg.timestamp,
                    'image': img_bgr,
                    'width': img_msg.width,
                    'height': img_msg.height
                })
                rgb_count += 1

            except Exception as e:
                print(f"Error decoding RGB frame: {e}")
                continue

    print(f"Collected {event_count} events and {rgb_count} RGB frames")

    if not all_events:
        print("No events collected!")
        return None

    # Concatenate all events
    events_combined = np.concatenate(all_events, axis=0)

    # Sort by timestamp
    sort_idx = np.argsort(events_combined[:, 2])
    events_sorted = events_combined[sort_idx]

    print(f"Total events: {len(events_sorted)}")
    print(f"Time range: {events_sorted[0, 2]:.3f} - {events_sorted[-1, 2]:.3f} seconds")
    print(f"Duration: {events_sorted[-1, 2] - events_sorted[0, 2]:.3f} seconds")

    # Save events in E2VID format
    events_file = events_dir / "events.txt"
    print(f"Saving events to {events_file}")

    with open(events_file, 'w') as f:
        # Write header (width height)
        f.write(f"1280 720\n")
        # Write events: t x y pol
        for event in events_sorted:
            t, x, y, pol = event[2], int(event[0]), int(event[1]), int(event[3])
            f.write(f"{t:.9f} {x} {y} {pol}\n")

    # Save RGB frames
    print(f"Saving {len(rgb_frames)} RGB frames")
    rgb_manifest = []

    for i, frame_data in enumerate(rgb_frames):
        # Save image
        img_filename = f"rgb_{i:06d}.png"
        img_path = rgb_dir / img_filename
        cv2.imwrite(str(img_path), frame_data['image'])

        # Add to manifest
        rgb_manifest.append({
            'index': i,
            'filename': img_filename,
            'timestamp_s': frame_data['timestamp_s'],
            'timestamp_ns': frame_data['timestamp_ns'],
            'width': frame_data['width'],
            'height': frame_data['height']
        })

    # Save RGB manifest
    manifest_file = output_path / "rgb_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(rgb_manifest, f, indent=2)

    # Save export summary
    summary = {
        'bag_dir': str(bag_dir),
        'output_dir': str(output_dir),
        'event_count': int(event_count),
        'rgb_count': int(rgb_count),
        'events_file': str(events_file),
        'rgb_dir': str(rgb_dir),
        'manifest_file': str(manifest_file),
        'time_range_s': [float(events_sorted[0, 2]), float(events_sorted[-1, 2])],
        'duration_s': float(events_sorted[-1, 2] - events_sorted[0, 2])
    }

    summary_file = output_path / "export_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Export complete! Summary saved to {summary_file}")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Export events from ROS bag")
    parser.add_argument("bag_dir", help="Path to ROS bag directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--max-duration", type=float, help="Max duration in seconds")
    parser.add_argument("--event-topic", default="/event_camera/events",
                       help="Event camera topic")
    parser.add_argument("--rgb-topic", default="/cam_sync/cam0/image_raw",
                       help="RGB camera topic")

    args = parser.parse_args()

    if not os.path.exists(args.bag_dir):
        print(f"Error: Bag directory does not exist: {args.bag_dir}")
        sys.exit(1)

    summary = export_events_from_bag(
        bag_dir=args.bag_dir,
        output_dir=args.output,
        event_topic=args.event_topic,
        rgb_topic=args.rgb_topic,
        max_duration_s=args.max_duration
    )

    if summary:
        print("\n=== Export Summary ===")
        print(f"Events: {summary['event_count']:,}")
        print(f"RGB frames: {summary['rgb_count']:,}")
        print(f"Duration: {summary['duration_s']:.1f}s")
        print(f"Files saved to: {summary['output_dir']}")
    else:
        print("Export failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
