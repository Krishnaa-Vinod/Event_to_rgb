#!/usr/bin/env python3
"""
Generate time-surface baseline images from raw events.

Time-surface method: each pixel's intensity decays exponentially from 
the timestamp of its most recent event.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import json
import time
from pathlib import Path
from tqdm import tqdm

# Add source pipeline path
sys.path.append('/scratch/kvinod/eventnavpp_datagen_pipeline')
from bag_reader import read_bag, deserialize
from evt3_decoder import EVT3StreamDecoder

class TimeSurface:
    """
    Time-surface representation for event cameras.
    
    Each pixel stores the timestamp of its most recent event.
    Intensity is computed as exponential decay: exp(-(t - t_last) / tau)
    """
    
    def __init__(self, width: int = 1280, height: int = 720, tau_ms: float = 50.0):
        self.width = width
        self.height = height
        self.tau_ms = tau_ms  # Decay time constant in milliseconds
        self.tau_us = tau_ms * 1000  # Convert to microseconds for computation
        
        # Initialize time surface (last event timestamp at each pixel)
        self.last_timestamp = np.zeros((height, width), dtype=np.int64)
        self.has_event = np.zeros((height, width), dtype=bool)
        
        print(f"Initialized {width}x{height} time surface with tau={tau_ms}ms")
    
    def update(self, events: np.ndarray):
        """
        Update time surface with new events.
        
        Args:
            events: Nx4 array [x, y, timestamp_us, polarity]
        """
        if len(events) == 0:
            return
            
        # Extract coordinates and timestamps
        x_coords = events[:, 0].astype(int)
        y_coords = events[:, 1].astype(int)
        timestamps = events[:, 2].astype(np.int64)
        
        # Clip coordinates to image bounds
        valid_mask = (x_coords >= 0) & (x_coords < self.width) & \
                     (y_coords >= 0) & (y_coords < self.height)
        
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        timestamps = timestamps[valid_mask]
        
        if len(x_coords) > 0:
            # Update last timestamp for each pixel (keep most recent)
            # Use advanced indexing to handle multiple events per pixel
            for i in range(len(x_coords)):
                x, y, t = x_coords[i], y_coords[i], timestamps[i]
                if not self.has_event[y, x] or t > self.last_timestamp[y, x]:
                    self.last_timestamp[y, x] = t
                    self.has_event[y, x] = True
    
    def render(self, current_time_us: int) -> np.ndarray:
        """
        Render time surface as intensity image at current time.
        
        Args:
            current_time_us: Current timestamp in microseconds
            
        Returns:
            Intensity image [0, 255] as uint8
        """
        # Compute time differences
        time_diff = current_time_us - self.last_timestamp
        
        # Apply exponential decay: exp(-dt / tau)
        intensity = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Only compute for pixels that have had events
        valid_pixels = self.has_event & (time_diff >= 0)
        
        if np.any(valid_pixels):
            # Exponential decay
            intensity[valid_pixels] = np.exp(-time_diff[valid_pixels].astype(np.float32) / self.tau_us)
        
        # Convert to [0, 255] uint8
        intensity_uint8 = (intensity * 255).astype(np.uint8)
        
        return intensity_uint8

def generate_time_surface_sequence(
    events_file: str,
    output_dir: str,
    tau_ms: float = 50.0,
    frame_rate_hz: float = 20.0,
    max_duration_s: float = None
):
    """
    Generate time-surface image sequence from events file.
    
    Args:
        events_file: Path to events.txt file (E2VID format)
        output_dir: Output directory for frames
        tau_ms: Decay time constant in milliseconds  
        frame_rate_hz: Output frame rate
        max_duration_s: Maximum duration to process
        
    Returns:
        dict: Generation summary
    """
    
    print(f"=== Generating Time-Surface Baseline ===")
    print(f"Events: {events_file}")
    print(f"Output: {output_dir}")
    print(f"Tau: {tau_ms}ms, Frame rate: {frame_rate_hz}Hz")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read events from file
    print("Loading events...")
    events_data = []
    
    with open(events_file, 'r') as f:
        lines = f.readlines()
    
    # Skip header line (width height)
    header = lines[0].strip().split()
    width, height = int(header[0]), int(header[1])
    
    print(f"Image size: {width}x{height}")
    
    # Parse events
    for line in tqdm(lines[1:], desc="Parsing events"):
        parts = line.strip().split()
        if len(parts) >= 4:
            t_s = float(parts[0])
            x = int(parts[1])  
            y = int(parts[2])
            pol = int(parts[3])
            
            # Convert timestamp to microseconds
            t_us = int(t_s * 1e6)
            
            events_data.append([x, y, t_us, pol])
    
    events_array = np.array(events_data)
    print(f"Loaded {len(events_array)} events")
    
    if len(events_array) == 0:
        print("No events found!")
        return None
        
    # Get time range
    start_time_us = events_array[0, 2]
    end_time_us = events_array[-1, 2]
    duration_s = (end_time_us - start_time_us) / 1e6
    
    print(f"Event time range: {start_time_us} to {end_time_us} µs")
    print(f"Duration: {duration_s:.2f} seconds")
    
    # Limit duration if requested
    if max_duration_s:
        target_end_us = start_time_us + int(max_duration_s * 1e6)
        if target_end_us < end_time_us:
            mask = events_array[:, 2] <= target_end_us
            events_array = events_array[mask]
            end_time_us = target_end_us
            duration_s = max_duration_s
            print(f"Limited to {duration_s:.2f} seconds, {len(events_array)} events")
    
    # Initialize time surface
    time_surface = TimeSurface(width, height, tau_ms)
    
    # Generate frames at regular intervals
    frame_interval_us = int(1e6 / frame_rate_hz)  # Frame interval in microseconds
    frames_generated = []
    
    current_frame_time_us = start_time_us
    event_idx = 0
    
    frame_count = 0
    start_generation = time.time()
    
    print(f"Generating frames every {frame_interval_us/1000:.1f}ms...")
    
    while current_frame_time_us <= end_time_us:
        # Add all events up to current frame time
        while event_idx < len(events_array) and events_array[event_idx, 2] <= current_frame_time_us:
            # Add single event to time surface
            event = events_array[event_idx:event_idx+1]
            time_surface.update(event)
            event_idx += 1
        
        # Render time surface at current time
        frame = time_surface.render(current_frame_time_us)
        
        # Save frame
        timestamp_s = current_frame_time_us / 1e6
        frame_filename = f"frame_{frame_count:06d}_t{timestamp_s:.3f}.png"
        frame_path = output_path / frame_filename
        
        cv2.imwrite(str(frame_path), frame)
        
        frames_generated.append({
            'index': frame_count,
            'filename': frame_filename,
            'timestamp_us': int(current_frame_time_us),
            'timestamp_s': timestamp_s,
            'events_processed': event_idx
        })
        
        frame_count += 1
        current_frame_time_us += frame_interval_us
        
        if frame_count % 10 == 0:
            print(f"Generated {frame_count} frames, processed {event_idx}/{len(events_array)} events")
    
    generation_time = time.time() - start_generation
    
    print(f"Generated {frame_count} frames in {generation_time:.1f}s")
    print(f"Generation FPS: {frame_count / generation_time:.1f}")
    
    # Save manifest
    manifest = {
        'method': 'time_surface',
        'events_file': str(events_file),
        'output_dir': str(output_path),
        'tau_ms': tau_ms,
        'frame_rate_hz': frame_rate_hz,
        'image_size': [width, height],
        'frames': frames_generated,
        'timing': {
            'generation_time_s': generation_time,
            'frames_generated': frame_count,
            'events_processed': len(events_array),
            'generation_fps': frame_count / generation_time
        }
    }
    
    manifest_file = output_path / "time_surface_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest saved to: {manifest_file}")
    return manifest

def generate_from_bag(
    bag_dir: str,
    output_dir: str,
    tau_ms: float = 50.0,
    frame_rate_hz: float = 20.0,
    max_duration_s: float = None,
    event_topic: str = "/event_camera/events"
):
    """
    Generate time-surface baseline directly from ROS bag.
    """
    
    print(f"=== Time-Surface from Bag ===")
    print(f"Bag: {bag_dir}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize decoder and time surface
    evt3_decoder = EVT3StreamDecoder(width=1280, height=720)
    time_surface = TimeSurface(width=1280, height=720, tau_ms=tau_ms)
    
    # Process bag messages
    start_time_ns = None
    frames_generated = []
    frame_count = 0
    
    frame_interval_ns = int(1e9 / frame_rate_hz)
    next_frame_time_ns = None
    
    start_generation = time.time()
    
    for msg in read_bag(bag_dir, topics={event_topic}):
        if start_time_ns is None:
            start_time_ns = msg.timestamp
            next_frame_time_ns = start_time_ns + frame_interval_ns
        
        elapsed_s = (msg.timestamp - start_time_ns) / 1e9
        if max_duration_s and elapsed_s > max_duration_s:
            break
        
        if msg.topic == event_topic:
            # Decode events
            try:
                event_msg = deserialize(msg)
                raw_events = bytes(event_msg.events)
                events = evt3_decoder.decode(raw_events)
                
                if len(events) > 0:
                    # Update time surface
                    time_surface.update(events)
                
                # Check if it's time to generate a frame
                while msg.timestamp >= next_frame_time_ns:
                    # Render frame
                    current_time_us = next_frame_time_ns // 1000
                    frame = time_surface.render(current_time_us)
                    
                    # Save frame
                    timestamp_s = next_frame_time_ns / 1e9
                    frame_filename = f"frame_{frame_count:06d}_t{timestamp_s:.3f}.png"
                    frame_path = output_path / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    
                    frames_generated.append({
                        'index': frame_count,
                        'filename': frame_filename,
                        'timestamp_ns': int(next_frame_time_ns),
                        'timestamp_s': timestamp_s
                    })
                    
                    frame_count += 1
                    next_frame_time_ns += frame_interval_ns
                    
                    if frame_count % 10 == 0:
                        print(f"Generated {frame_count} frames at t={timestamp_s:.2f}s")
                        
            except Exception as e:
                print(f"Error processing event packet: {e}")
                continue
    
    generation_time = time.time() - start_generation
    
    print(f"Generated {frame_count} frames in {generation_time:.1f}s")
    
    # Save manifest
    manifest = {
        'method': 'time_surface_bag_direct',
        'bag_dir': str(bag_dir),
        'output_dir': str(output_path),
        'tau_ms': tau_ms,
        'frame_rate_hz': frame_rate_hz,
        'frames': frames_generated,
        'timing': {
            'generation_time_s': generation_time,
            'frames_generated': frame_count,
            'generation_fps': frame_count / generation_time
        }
    }
    
    manifest_file = output_path / "time_surface_bag_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest saved to: {manifest_file}")
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Generate time-surface baseline")
    parser.add_argument("input", help="Path to events.txt file or bag directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--tau-ms", type=float, default=50.0, 
                       help="Decay time constant in milliseconds")
    parser.add_argument("--fps", type=float, default=20.0, help="Output frame rate")
    parser.add_argument("--max-duration", type=float, help="Max duration in seconds")
    parser.add_argument("--from-bag", action="store_true", 
                       help="Input is bag directory (not events.txt)")
    parser.add_argument("--event-topic", default="/event_camera/events",
                       help="Event topic (for bag input)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input does not exist: {args.input}")
        sys.exit(1)

    try:
        if args.from_bag:
            result = generate_from_bag(
                bag_dir=args.input,
                output_dir=args.output,
                tau_ms=args.tau_ms,
                frame_rate_hz=args.fps,
                max_duration_s=args.max_duration,
                event_topic=args.event_topic
            )
        else:
            result = generate_time_surface_sequence(
                events_file=args.input,
                output_dir=args.output,
                tau_ms=args.tau_ms,
                frame_rate_hz=args.fps,
                max_duration_s=args.max_duration
            )

        if result:
            timing = result.get("timing", {})
            print(f"\n=== Time-Surface Generation Results ===")
            print(f"Frames: {timing.get('frames_generated', 0)}")
            print(f"Generation time: {timing.get('generation_time_s', 0):.1f}s")
            print(f"Generation FPS: {timing.get('generation_fps', 0):.1f}")
        else:
            print("Time-surface generation failed!")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
