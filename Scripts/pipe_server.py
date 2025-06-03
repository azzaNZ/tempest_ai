#!/usr/bin/env python3
"""
Pipe server for Tempest AI.
Replaces socket-based communication with named pipes for better reliability.
"""

# Prevent direct execution
if __name__ == "__main__":
    print("This is not the main application, run 'main.py' instead")
    exit(1)

import os
import sys
import time
import struct
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import random
import traceback
from datetime import datetime
import tempfile

# Import from config.py
from config import (
    SERVER_CONFIG,
    MODEL_DIR,
    LATEST_MODEL_PATH,
    ACTION_MAPPING,
    metrics,
    RL_CONFIG
)

# Import from aimodel.py
from aimodel import (
    parse_frame_data, 
    get_expert_action, 
    expert_action_to_index, 
    encode_action_to_game
)

class PipeServer:
    """Pipe-based server to handle communication with MAME Lua script"""
    def __init__(self, agent, safe_metrics):
        print("Initializing PipeServer")
        self.agent = agent
        self.metrics = safe_metrics
        self.running = True
        self.client_states = {}  # Dictionary to store per-client state
        self.client_lock = threading.Lock()  # Lock for client dictionaries
        self.shutdown_event = threading.Event()  # Event to signal shutdown
        
        # Create pipe paths
        self.pipe_dir = tempfile.gettempdir()
        self.input_pipe_path = os.path.join(self.pipe_dir, "tempest_ai_input")
        self.output_pipe_path = os.path.join(self.pipe_dir, "tempest_ai_output")
        
        # Pipe file handles
        self.input_pipe = None
        self.output_pipe = None
        
        # Client tracking
        self.client_id = 0
        self.client_state = None
        
    def create_pipes(self):
        """Create named pipes for communication"""
        try:
            # Remove existing pipes if they exist
            for pipe_path in [self.input_pipe_path, self.output_pipe_path]:
                if os.path.exists(pipe_path):
                    os.unlink(pipe_path)
            
            # Create named pipes (FIFOs) on Unix-like systems
            if os.name == 'posix':
                os.mkfifo(self.input_pipe_path)
                os.mkfifo(self.output_pipe_path)
                print(f"Created named pipes: {self.input_pipe_path}, {self.output_pipe_path}")
            else:
                # On Windows, we'll use regular files with polling
                # Create empty files that will be used for communication
                with open(self.input_pipe_path, 'w') as f:
                    pass
                with open(self.output_pipe_path, 'w') as f:
                    pass
                print(f"Created communication files: {self.input_pipe_path}, {self.output_pipe_path}")
                
            return True
        except Exception as e:
            print(f"Error creating pipes: {e}")
            traceback.print_exc()
            return False
    
    def open_pipes(self):
        """Open pipes for reading and writing"""
        try:
            if os.name == 'posix':
                # On Unix-like systems, open FIFOs
                print("Opening input pipe for reading...")
                self.input_pipe = open(self.input_pipe_path, 'rb')
                print("Opening output pipe for writing...")
                self.output_pipe = open(self.output_pipe_path, 'wb')
            else:
                # On Windows, open files in binary mode
                print("Opening communication files...")
                self.input_pipe = open(self.input_pipe_path, 'rb')
                self.output_pipe = open(self.output_pipe_path, 'wb')
                
            print("Pipes opened successfully")
            return True
        except Exception as e:
            print(f"Error opening pipes: {e}")
            traceback.print_exc()
            return False
    
    def close_pipes(self):
        """Close pipe handles"""
        try:
            if self.input_pipe:
                self.input_pipe.close()
                self.input_pipe = None
            if self.output_pipe:
                self.output_pipe.close()
                self.output_pipe = None
            print("Pipes closed")
        except Exception as e:
            print(f"Error closing pipes: {e}")
    
    def cleanup_pipes(self):
        """Remove pipe files"""
        try:
            for pipe_path in [self.input_pipe_path, self.output_pipe_path]:
                if os.path.exists(pipe_path):
                    os.unlink(pipe_path)
            print("Pipe files cleaned up")
        except Exception as e:
            print(f"Error cleaning up pipes: {e}")
    
    def start(self):
        """Start the pipe server"""
        try:
            # Create pipes
            if not self.create_pipes():
                print("Failed to create pipes")
                return
            
            # Initialize client state
            self.client_state = {
                'last_state': None,
                'last_action_idx': None,
                'last_action_source': None,
                'total_reward': 0,
                'was_done': False,
                'episode_dqn_reward': 0,
                'episode_expert_reward': 0,
                'connected_time': datetime.now(),
                'frames_processed': 0,
                'fps': 0.0,
                'frames_last_second': 0,
                'last_fps_update': time.time()
            }
            
            # Update metrics
            with self.client_lock:
                metrics.client_count = 1
            
            print("Waiting for MAME to connect...")
            
            # Open pipes (this will block until MAME connects)
            if not self.open_pipes():
                print("Failed to open pipes")
                return
            
            print("MAME connected, starting communication loop...")
            
            # Main communication loop
            while self.running and not self.shutdown_event.is_set():
                try:
                    # Read data length (2 bytes, big-endian unsigned short)
                    length_data = self.input_pipe.read(2)
                    if not length_data or len(length_data) < 2:
                        print("Connection closed or incomplete length data")
                        break
                    
                    data_length = struct.unpack(">H", length_data)[0]
                    
                    # Read the actual data
                    data = self.input_pipe.read(data_length)
                    if not data or len(data) < data_length:
                        print(f"Incomplete data packet ({len(data) if data else 0}/{data_length} bytes)")
                        continue
                    
                    # Process the frame data
                    self.process_frame(data)
                    
                except Exception as e:
                    print(f"Error in communication loop: {e}")
                    traceback.print_exc()
                    break
            
        except Exception as e:
            print(f"Server error: {e}")
            traceback.print_exc()
        finally:
            print("Pipe server shutting down...")
            self.shutdown_event.set()
            self.close_pipes()
            self.cleanup_pipes()
            
            # Update metrics
            with self.client_lock:
                metrics.client_count = 0
            
            print("Pipe server stopped")
    
    def process_frame(self, data):
        """Process a single frame of data from MAME"""
        try:
            # Parameter count validation
            header_format_peek = ">H"
            peek_size = struct.calcsize(header_format_peek)
            if len(data) >= peek_size:
                num_values_received = struct.unpack(header_format_peek, data[:peek_size])[0]
                if num_values_received != SERVER_CONFIG.params_count:
                    raise ValueError(f"Parameter count mismatch! Expected {SERVER_CONFIG.params_count}, received {num_values_received}")
            else:
                raise ConnectionError("Data too short to read parameter count")
            
            # Parse the frame data
            frame = parse_frame_data(data)
            if not frame:
                print("Failed to parse frame data")
                # Send empty response on parsing failure
                self.send_action(0, 0, 0)
                return
            
            # Update client state
            state = self.client_state
            state['frames_processed'] += 1
            
            # Update FPS tracking
            current_time = time.time()
            state['frames_last_second'] += 1
            elapsed = current_time - state['last_fps_update']
            
            if elapsed >= 1.0:
                state['fps'] = state['frames_last_second'] / elapsed
                state['frames_last_second'] = 0
                state['last_fps_update'] = current_time
            
            # Handle save signal from game
            if frame.save_signal:
                try:
                    if hasattr(self, 'agent') and self.agent:
                        self.agent.save(LATEST_MODEL_PATH)
                        print("Model saved on request from MAME")
                    else:
                        print("Agent not available for saving")
                except Exception as e:
                    print(f"ERROR saving model: {e}")
            
            # Update global metrics
            current_frame = self.metrics.update_frame_count()
            self.metrics.update_epsilon()
            self.metrics.update_expert_ratio()
            self.metrics.update_game_state(frame.enemy_seg, frame.open_level)
            
            # Process previous step's results if available
            if state.get('last_state') is not None and state.get('last_action_idx') is not None:
                if hasattr(self, 'agent') and self.agent:
                    self.agent.step(
                        state['last_state'],
                        np.array([[state['last_action_idx']]]),
                        frame.reward,
                        frame.state,
                        frame.done
                    )
                else:
                    print("Agent not available for step")
                
                # Track rewards
                state['total_reward'] = state.get('total_reward', 0) + frame.reward
                
                # Track which system's rewards based on the previous action source
                prev_action_source = state.get('last_action_source')
                if prev_action_source == "dqn":
                    state['episode_dqn_reward'] = state.get('episode_dqn_reward', 0) + frame.reward
                elif prev_action_source == "expert":
                    state['episode_expert_reward'] = state.get('episode_expert_reward', 0) + frame.reward
            
            # Handle episode completion
            if frame.done:
                if not state.get('was_done', False):
                    self.metrics.add_episode_reward(
                        state.get('total_reward', 0),
                        state.get('episode_dqn_reward', 0),
                        state.get('episode_expert_reward', 0)
                    )
                
                state['was_done'] = True
                # Send empty action on 'done' frame
                self.send_action(0, 0, 0)
                
                # Reset state for next episode
                state['last_state'] = None
                state['last_action_idx'] = None
                state['total_reward'] = 0
                state['episode_dqn_reward'] = 0
                state['episode_expert_reward'] = 0
                return
            
            elif state.get('was_done', False):
                # Reset episode state if previous frame was done
                state['was_done'] = False
                state['total_reward'] = 0
                state['episode_dqn_reward'] = 0
                state['episode_expert_reward'] = 0
            
            # Generate action (only if not 'done')
            self.metrics.increment_total_controls()
            
            # Decide between expert system and DQN
            action_idx = None
            fire, zap, spinner = 0, 0, 0.0
            action_source = "unknown"
            
            if hasattr(self, 'agent') and self.agent:
                if random.random() < self.metrics.get_expert_ratio() and not self.metrics.is_override_active():
                    # Use expert system
                    fire, zap, spinner = get_expert_action(
                        frame.enemy_seg,
                        frame.player_seg,
                        frame.open_level,
                        frame.expert_fire,
                        frame.expert_zap
                    )
                    self.metrics.increment_guided_count()
                    action_source = "expert"
                    action_idx = expert_action_to_index(fire, zap, spinner)
                else:
                    # Use DQN with current epsilon
                    start_time = time.perf_counter()
                    action_idx = self.agent.act(frame.state, self.metrics.get_epsilon())
                    end_time = time.perf_counter()
                    inference_time = end_time - start_time
                    
                    # Update inference metrics
                    with self.metrics.lock:
                        self.metrics.total_inference_time += inference_time
                        self.metrics.total_inference_requests += 1
                    
                    fire, zap, spinner = ACTION_MAPPING[action_idx]
                    action_source = "dqn"
            else:
                print("Agent not available for action generation")
                action_source = "none"
            
            # Store state and action for next iteration
            if action_idx is not None:
                state['last_state'] = frame.state
                state['last_action_idx'] = action_idx
                state['last_action_source'] = action_source
            
            # Send action to game
            game_fire, game_zap, game_spinner = encode_action_to_game(fire, zap, spinner)
            self.send_action(game_fire, game_zap, game_spinner)
            
            # Periodic target network update
            if hasattr(self, 'agent') and self.agent and current_frame % RL_CONFIG.update_target_every == 0:
                self.agent.update_target_network()
            
            # Periodic model saving
            if hasattr(self, 'agent') and self.agent and current_frame % RL_CONFIG.save_interval == 0:
                self.agent.save(LATEST_MODEL_PATH)
        
        except ValueError as ve:
            print(f"Parameter validation error: {ve}")
        except Exception as e:
            print(f"Error processing frame: {e}")
            traceback.print_exc()
            # Send default action on error
            self.send_action(0, 0, 0)
    
    def send_action(self, fire, zap, spinner):
        """Send action back to MAME"""
        try:
            action_data = struct.pack("bbb", fire, zap, spinner)
            self.output_pipe.write(action_data)
            self.output_pipe.flush()  # Ensure data is written immediately
        except Exception as e:
            print(f"Error sending action: {e}")
            traceback.print_exc()
    
    def get_pipe_paths(self):
        """Get the paths to the communication pipes"""
        return self.input_pipe_path, self.output_pipe_path
