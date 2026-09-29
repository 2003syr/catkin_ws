# TEB planar baseline and teacher

`planar_base_teb_demo.launch` starts the mature ROS Navigation baseline:

- `global_planner` builds the global path;
- `teb_local_planner` optimizes nonholonomic local trajectories;
- the bridge applies only tracked-base `[v, omega]` commands;
- both costmaps and TEB use the conservative chassis-and-folded-arm footprint.

## Observable teacher dataset

`planar_base_teb_dataset_collection.launch` cycles through six labelled poses:
central, upper and lower far-side goals plus their return poses. This supplies
both bypass directions, return motion and different terminal orientations. It
only retains complete episodes reported successful by `move_base` with no
non-ground `/base_contacts`.

The teacher is intentionally observable. The label is TEB's final normalized
`[linear.x, angular.z]` command, while the 62-D learner observation contains:

```
TEB local pose subgoal (4)
+ measured [v, omega] (2)
+ 36-bin LiDAR (36)
+ five-point TEB local-path preview (10)
+ path metrics (3)
+ previous teacher action (2)
+ path validity mask (5)
```

The dataset is rewritten after every accepted episode, so completed data
survives an interrupted collection. It can be consumed by
`train_planar_tracking_bc.py`; PPO should only start after deterministic BC
evaluation passes. When route labels are present, BC performs an
episode-level stratified split so each sufficiently represented route appears
in both training and validation. Collection also enforces a configurable
minimum success count for every route, preventing easy routes from hiding a
failed route.

## Closed-loop BC evaluation

`planar_base_teb_bc_evaluation.launch` keeps TEB online only for its local
path and reference command. TEB publishes to `/teb_reference_cmd_vel`, which
is not connected to the tracked-base bridge. The deterministic BC policy
publishes `[v, omega]` to `/learner_cmd_vel`; this is the only command accepted
by the bridge.

ROS Melodic nodes run under Python 2 while the trained PyTorch checkpoint
requires Python 3. The evaluation therefore uses two processes:

1. `teb_bc_policy_server.py` loads the checkpoint in the Python 3 virtual
   environment and serves deterministic inference on port 5559.
2. `evaluate_teb_planar_bc.py` constructs the exact 62-D observation in ROS,
   requests an action, applies an emergency-only LiDAR stop, and executes it.

Per-route output reports success, contact collision, timeout, minimum goal
distance, emergency shield count, learner-to-TEB action MSE, and mean policy
inference latency. A six-route summary applies the configured success and
collision gate.

## PPO from the validated BC actor

`planar_base_teb_ppo_training.launch` exposes a socket environment on port
5558 while preserving exactly the same 62-D TEB observation contract. TEB
continues to replan, but `/teb_reference_cmd_vel` remains disconnected from
the base. PPO actions alone are published on `/learner_cmd_vel`.

The PPO trainer accepts the validated model through `--bc-checkpoint`.
Only the actor and observation normalizer are restored; the critic is new.
Passing the 30-episode TEB dataset through `--bc-dataset` adds a decaying
offline behavior anchor so the first PPO updates do not immediately erase the
collision-free BC behavior.

For a newly initialized critic, use equal nonzero
`--actor-freeze-steps` and `--normalizer-freeze-steps`. During this warm-up
window, rollout actions still include the configured exploration noise but
gradient updates affect only the independent critic. This preserves the
validated deterministic BC policy until value estimates have seen complete
TEB routes.
