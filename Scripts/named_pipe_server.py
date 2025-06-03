#!/usr/bin/env python3
"""
Named Pipe server for Tempest AI (Windows)
Handles communication with MAME Lua script via Windows Named Pipes
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
import win32pipe
import win32file
import win32api
import win32con
from typing import Dict, List, Optional, Tuple, Any
import random
import traceback
from datetime import datetime

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

class NamedPipeServer:
    """Named Pipe-based server for Windows communication with MAME"""
    
    def __init__(self, pipe_name, agent, safe_metrics):
        print(f"Initializing NamedPipeServer: {pipe_name}")
        self.pipe_name = pipe_name
        self.agent = agent
        self.metrics = safe_metrics
        self.running = True
        self.clients = {}  # Dictionary to track active clients
        self.client_states = {}  # Dictionary to store per-client state
        self.client_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
    def start(self):
        """Start the named pipe server"""
        try:
            print(f"Named Pipe server starting on {self.pipe_name}")
            
            # Accept client connections in a loop
            while self.running:
                try:
                    # Create named pipe
                    pipe_handle = win32pipe.CreateNamedPipe(
                        self.pipe_name,
                        win32pipe.PIPE_ACCESS_DUPLEX,
                        win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                        1,  # Max instances
                        65536,  # Out buffer size
                        65536,  # In buffer size
                        0,  # Default timeout
                        None  # Security attributes
                    )
                    
                    if pipe_handle == win32file.INVALID_HANDLE_VALUE:
                        print("Failed to create named pipe")
                        time.sleep(1)
                        continue
                    
                    print(f"Named pipe created, waiting for client connection...")
                    
                    # Wait for client to connect
                    win32pipe.ConnectNamedPipe(pipe_handle, None)
                    
                    # Generate a unique client ID
                    client_id = self.generate_client_id()
                    
                    print(f"New connection on named pipe, ID: {client_id}")
                    
                    # Initialize client state
                    client_state = {
                        'pipe_handle': pipe_handle,
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
                    
                    # Store client information
                    with self.client_lock:
                        self.client_states[client_id] = client_state
                        self.clients[client_id] = pipe_handle
                        metrics.client_count = len(self.clients)
                    
                    # Start a thread to handle this client
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(pipe_handle, client_id),
                        daemon=True,
                        name=f"NamedPipe-Client-{client_id}"
                    )
                      # Store thread and start it
                    with self.client_lock:
                        self.clients[client_id] = client_thread                   
                    
                    client_thread.start()
                    
                    # Wait for client thread to finish before accepting new connections
                    print(f"Client {client_id} connected, named pipe server handling single client")
                    client_thread.join()
                    print(f"Client {client_id} disconnected, ready for new connections")
                    
                except Exception as e:
                    print(f"Error accepting client connection: {e}")
                    traceback.print_exc()
                    time.sleep(1)
            
        except Exception as e:
            print(f"Server error: {e}")
            traceback.print_exc()
        finally:
            print("Named Pipe server shutting down...")
            self.shutdown_event.set()
            self.cleanup_all_clients()
            print("Named Pipe server stopped")
    
    def cleanup_all_clients(self):
        """Clean up all client threads and resources"""
        with self.client_lock:
            client_ids = list(self.clients.keys())
            
            for client_id in client_ids:
                if client_id in self.clients:
                    del self.clients[client_id]
                if client_id in self.client_states:
                    del self.client_states[client_id]
            
            print(f"Cleaned up all {len(client_ids)} clients during shutdown")
    
    def generate_client_id(self):
        """Generate a unique client ID"""
        with self.client_lock:
            all_possible_ids = set(range(SERVER_CONFIG.max_clients))
            current_ids = set(self.clients.keys())
            available_ids = list(all_possible_ids - current_ids)
            
            if available_ids:
                available_ids.sort()
                return available_ids[0]
            
            # Fallback to overflow ID
            overflow_id = SERVER_CONFIG.max_clients + len(self.clients)
            return overflow_id
    
    def handle_client(self, pipe_handle, client_id):
        """Handle communication with a client via named pipe"""
        try:
            buffer_size = 32768
            
            # Initial handshake - read the handshake message
            try:
                result, handshake_data = win32file.ReadFile(pipe_handle, 2)
                if result == 0 and len(handshake_data) == 2:
                    print(f"Client {client_id}: Handshake received")
                else:
                    print(f"Client {client_id}: Invalid handshake")
                    return
            except Exception as e:
                print(f"Client {client_id}: Handshake error: {e}")
                return
            
            # Main communication loop
            while self.running and not self.shutdown_event.is_set():
                try:
                    # Read data length (2 bytes)
                    result, length_data = win32file.ReadFile(pipe_handle, 2)
                    if result != 0 or len(length_data) < 2:
                        print(f"Client {client_id}: Failed to read length header")
                        break
                    
                    data_length = struct.unpack(">H", length_data)[0]
                    
                    # Read the actual data
                    data = b""
                    remaining = data_length
                    
                    while remaining > 0:
                        result, chunk = win32file.ReadFile(pipe_handle, min(buffer_size, remaining))
                        if result != 0:
                            print(f"Client {client_id}: Error reading data chunk")
                            break
                        data += chunk
                        remaining -= len(chunk)
                    
                    if len(data) < data_length:
                        print(f"Client {client_id}: Incomplete data packet ({len(data)}/{data_length} bytes)")
                        continue
                    
                    # Parameter count validation
                    try:
                        header_format_peek = ">H"
                        peek_size = struct.calcsize(header_format_peek)
                        if len(data) >= peek_size:
                            num_values_received = struct.unpack(header_format_peek, data[:peek_size])[0]
                            if num_values_received != SERVER_CONFIG.params_count:
                                raise ValueError(f"Parameter count mismatch! Expected {SERVER_CONFIG.params_count}, received {num_values_received}")
                    except ValueError as ve:
                        print(f"Client {client_id}: {ve}. Closing connection.")
                        break
                    except Exception as e:
                        print(f"Client {client_id}: Error checking parameter count: {e}")
                        break
                    
                    # Parse the frame data
                    frame = parse_frame_data(data)
                    if not frame:
                        print(f"Client {client_id}: Failed to parse frame data")
                        win32file.WriteFile(pipe_handle, struct.pack("bbb", 0, 0, 0))
                        continue
                    
                    # Get client state
                    with self.client_lock:
                        if client_id not in self.client_states:
                            print(f"Client {client_id}: State not found, disconnected. Aborting frame.")
                            break
                        
                        state = self.client_states[client_id]
                        state['frames_processed'] += 1
                        
                        # Update client-specific FPS tracking
                        current_time = time.time()
                        state['frames_last_second'] += 1
                        elapsed = current_time - state['last_fps_update']
                        
                        if elapsed >= 1.0:
                            state['fps'] = state['frames_last_second'] / elapsed
                            state['frames_last_second'] = 0
                            state['last_fps_update'] = current_time
                    
                    # Handle save signal
                    if frame.save_signal:
                        try:
                            if hasattr(self, 'agent') and self.agent:
                                self.agent.save(LATEST_MODEL_PATH)
                            else:
                                print(f"Client {client_id}: Agent not available for saving")
                        except Exception as e:
                            print(f"Client {client_id}: ERROR saving model: {e}")
                    
                    # Update global metrics
                    current_frame = self.metrics.update_frame_count()
                    self.metrics.update_epsilon()
                    self.metrics.update_expert_ratio()
                    self.metrics.update_game_state(frame.enemy_seg, frame.open_level)
                    
                    # Process previous step's results
                    if state.get('last_state') is not None and state.get('last_action_idx') is not None:
                        if hasattr(self, 'agent') and self.agent:
                            self.agent.step(
                                state['last_state'],
                                np.array([[state['last_action_idx']]]),
                                frame.reward,
                                frame.state,
                                frame.done
                            )
                        
                        # Track rewards
                        state['total_reward'] = state.get('total_reward', 0) + frame.reward
                        
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
                        win32file.WriteFile(pipe_handle, struct.pack("bbb", 0, 0, 0))
                        
                        # Reset state for next episode
                        state['last_state'] = None
                        state['last_action_idx'] = None
                        state['total_reward'] = 0
                        state['episode_dqn_reward'] = 0
                        state['episode_expert_reward'] = 0
                        continue
                    
                    elif state.get('was_done', False):
                        # Reset episode state if previous frame was done
                        state['was_done'] = False
                        state['total_reward'] = 0
                        state['episode_dqn_reward'] = 0
                        state['episode_expert_reward'] = 0
                    
                    # Generate action
                    self.metrics.increment_total_controls()
                    
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
                            # Use DQN
                            start_time = time.perf_counter()
                            action_idx = self.agent.act(frame.state, self.metrics.get_epsilon())
                            end_time = time.perf_counter()
                            inference_time = end_time - start_time
                            
                            with self.metrics.lock:
                                self.metrics.total_inference_time += inference_time
                                self.metrics.total_inference_requests += 1
                            
                            fire, zap, spinner = ACTION_MAPPING[action_idx]
                            action_source = "dqn"
                    else:
                        print(f"Client {client_id}: Agent not available for action generation")
                        action_source = "none"
                    
                    # Store state and action for next iteration
                    if action_idx is not None:
                        state['last_state'] = frame.state
                        state['last_action_idx'] = action_idx
                        state['last_action_source'] = action_source
                    
                    # Send action to game
                    game_fire, game_zap, game_spinner = encode_action_to_game(fire, zap, spinner)
                    win32file.WriteFile(pipe_handle, struct.pack("bbb", game_fire, game_zap, game_spinner))
                    
                    # Periodic target network update (only from client 0)
                    if client_id == 0 and hasattr(self, 'agent') and self.agent and current_frame % RL_CONFIG.update_target_every == 0:
                        self.agent.update_target_network()
                    
                    # Periodic model saving (only from client 0)
                    if client_id == 0 and hasattr(self, 'agent') and self.agent and current_frame % RL_CONFIG.save_interval == 0:
                        self.agent.save(LATEST_MODEL_PATH)
                        
                except Exception as e:
                    print(f"Error handling client {client_id}: {e}")
                    traceback.print_exc()
                    break
                    
        except Exception as e:
            print(f"Fatal error handling client {client_id}: {e}")
            traceback.print_exc()
        finally:
            # Cleanup
            try:
                win32file.CloseHandle(pipe_handle)
            except:
                pass
            
            with self.client_lock:
                client_exists = client_id in self.client_states
                if client_exists:
                    del self.client_states[client_id]
                if client_id in self.clients:
                    self.clients[client_id] = None
                
                metrics.client_count = len([c for c in self.clients.values() if c is not None])
                if client_exists:
                    print(f"Client {client_id} cleanup complete. Active clients: {metrics.client_count}")
    
    def stop(self):
        """Stop the server"""
        self.running = False
        self.shutdown_event.set()
