#!/usr/bin/env python
import tensorflow as tf
import os
import logging
import time


# def trace_to_each_tier_raw(hybrid, log_file="logs/replay_latencies.csv"):

#     trace_df = pd.read_csv(hybrid.application, names=["offset", "size", "type"], header=None)
#     trace_df = trace_df[(trace_df["type"] == "Write") & (trace_df["size"] == 4096)]
#     trace_df = trace_df.astype({"offset": int, "size": int})

#     os.makedirs(os.path.dirname(log_file), exist_ok=True)
#     gLBA = {
#         "fast": hybrid.gLBAFast,
#         "mid": hybrid.gLBAMid,
#         "slow": hybrid.gLBASlow
#     }

#     with open(log_file, "w") as f:
#         f.write("offset,size,tier_name,latency_us\n")

#         for tier_name, device, gLBA_key in [
#             ("fast", hybrid.fastDevice, "fast"),
#             ("mid",  hybrid.midDevice,  "mid"),
#             ("slow", hybrid.slowDevice, "slow"),
#         ]:
#             for _, row in trace_df.iterrows():
#                 offset = row["offset"]
#                 size = row["size"]
#                 LBA = int(gLBA[gLBA_key])
#                 gLBA[gLBA_key] += size  # increment for next write

#                 start = time.perf_counter()
#                 hybrid.my_functions.sibyl_write(device, LBA, size)
#                 end = time.perf_counter()
#                 latency_us = (end - start) * 1e6

#                 f.write(f"{offset},{size},{tier_name},{latency_us:.2f}\n")


# Add a function to print loggings from env
def print_IO_loggings(environment, policy):
    old_level = logging.getLogger().level  # Save current logging level
    
    # reset timestep to 0 first
    time_step = environment.reset()
    # i: num of timestep
    i=0

    # trace_to_each_tier_raw(environment.pyenv.envs[0])

    start_time = time.time()
    # run the I/O loop
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        i+=1
        # comment out this if condition if you want to print all the loggings
        # use the getter function to get the trace length
        if i + 2 > environment.pyenv.envs[0].get_trace_length():
            # change to debug level only for the last timestep
            logging.getLogger().setLevel(logging.DEBUG)  # Enable debug messages
    
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60
    print(f"Total time elapsed: {elapsed_minutes:.6f} minutes")
    logging.info(f"Total time elapsed: {elapsed_minutes:.6f} minutes")


    # print the logging info
    if hasattr(environment, "_debug_info"):
        print("\n--- I/O running Info ---")
        for key, value in environment._debug_info.items():
            print(f"{key}: {value:.6f}")
        print("------------------\n")
    else:
        print("Debug info not available")
    # Restore original logging level
    logging.getLogger().setLevel(old_level)


def compute_avg_return(environment, policy, num_episodes):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        i=0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            i +=1
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def create_recurrent_network(
    input_fc_layer_units,
    lstm_size,
    output_fc_layer_units,
    num_actions):
  rnn_cell = tf.keras.layers.StackedRNNCells(
      [fused_lstm_cell(s) for s in lstm_size])
  return models.Sequential(
      [dense(num_units) for num_units in input_fc_layer_units]
      + [dynamic_unroll_layer.DynamicUnroll(rnn_cell)]
      + [dense(num_units) for num_units in output_fc_layer_units]
      + [logits(num_actions)])  
def create_feedforward_network(fc_layer_units, num_actions):
  return sequential.Sequential(
      [dense(num_units) for num_units in fc_layer_units]
      + [logits(num_actions)])

def create_recurrent_network(
        input_fc_layer_units,
        lstm_size,
        output_fc_layer_units,
        num_actions):
      rnn_cell = tf.keras.layers.StackedRNNCells(
          [fused_lstm_cell(s) for s in lstm_size])
      return sequential.Sequential(
          [dense(num_units) for num_units in input_fc_layer_units]
          + [dynamic_unroll_layer.DynamicUnroll(rnn_cell)]
          + [dense(num_units) for num_units in output_fc_layer_units]
          + [logits(num_actions)])

