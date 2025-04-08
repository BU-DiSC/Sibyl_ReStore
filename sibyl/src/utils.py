#!/usr/bin/env python
import tensorflow as tf
import logging

# Add a function to print loggings from env
def print_IO_loggings(environment, policy):
    old_level = logging.getLogger().level  # Save current logging level
    
    # reset timestep to 0 first
    time_step = environment.reset()
    # i: num of timestep
    i=0

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

