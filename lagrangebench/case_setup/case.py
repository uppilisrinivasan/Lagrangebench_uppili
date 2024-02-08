"""Case setup functions."""

from typing import Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
from jax import Array, jit, lax, vmap
from jax_md import space
from jax_md.dataclasses import dataclass, static_field
from jax_md.partition import NeighborList, NeighborListFormat

from lagrangebench.data.utils import get_dataset_stats
from lagrangebench.defaults import defaults
from lagrangebench.train.strats import add_gns_noise

from .features import FeatureDict, TargetDict, physical_feature_builder
from .partition import neighbor_list

TrainCaseOut = Tuple[Array, FeatureDict, TargetDict, NeighborList]
EvalCaseOut = Tuple[FeatureDict, NeighborList]
SampleIn = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]         ###### added third input

AllocateFn = Callable[[Array, SampleIn, float, int], TrainCaseOut]
AllocateEvalFn = Callable[[SampleIn], EvalCaseOut]

PreprocessFn = Callable[[Array, SampleIn, float, NeighborList, int], TrainCaseOut]
PreprocessEvalFn = Callable[[SampleIn, NeighborList], EvalCaseOut]

IntegrateFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
IntegrateTempFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]         #######

TempDiffFn = Tuple[jnp.ndarray, jnp.ndarray]



@dataclass
class CaseSetupFn:
    """Dataclass that contains all functions required to setup the case and simulate.

    Attributes:
        allocate: AllocateFn, runs the preprocessing without having a NeighborList as
            input.
        preprocess: PreprocessFn, takes positions from the dataloader, computes
            velocities, adds random-walk noise if needed, then updates the neighbor
            list, and return the inputs to the neural network as well as the targets.
        allocate_eval: AllocateEvalFn, same as allocate, but without noise addition
            and without targets.
        preprocess_eval: PreprocessEvalFn, same as allocate_eval, but jit-able.
        integrate: IntegrateFn, semi-implicit Euler integrations step respecting
            all boundary conditions.
        displacement: space.DisplacementFn, displacement function aware of boundary
            conditions (periodic on non-periodic).
        normalization_stats: Dict, normalization statisticss for input velocities and
            output acceleration.
    """

    allocate: AllocateFn = static_field()
    preprocess: PreprocessFn = static_field()
    allocate_eval: AllocateEvalFn = static_field()
    preprocess_eval: PreprocessEvalFn = static_field()
    integrate: IntegrateFn = static_field()
    integrate_temp: IntegrateTempFn = static_field()
    displacement: space.DisplacementFn = static_field()
    normalization_stats: Dict = static_field()
    temp_diff: TempDiffFn = static_field()


def case_builder(
    box: Tuple[float, float, float],
    metadata: Dict,
    input_seq_length: int,
    isotropic_norm: bool = defaults.isotropic_norm,
    noise_std: float = defaults.noise_std,
    external_force_fn: Optional[Callable] = None,
    magnitude_features: bool = defaults.magnitude_features,
    neighbor_list_backend: str = defaults.neighbor_list_backend,
    neighbor_list_multiplier: float = defaults.neighbor_list_multiplier,
    dtype: jnp.dtype = defaults.dtype,
):
    """Set up a CaseSetupFn that contains every required function besides the model.

    Inspired by the `partition.neighbor_list` function in JAX-MD.

    The core functions are:
        * allocate, allocate memory for the neighbors list.
        * preprocess, update the neighbors list.
        * integrate, semi-implicit Euler respecting periodic boundary conditions.

    Args:
        box: Box xyz sizes of the system.
        metadata: Dataset metadata dictionary.
        input_seq_length: Length of the input sequence.
        isotropic_norm: Whether to normalize dimensions equally.
        noise_std: Noise standard deviation.
        external_force_fn: External force function.
        magnitude_features: Whether to add velocity magnitudes in the features.
        neighbor_list_backend: Backend of the neighbor list.
        neighbor_list_multiplier: Capacity multiplier of the neighbor list.
        dtype: Data type.
    """
    normalization_stats = get_dataset_stats(metadata, isotropic_norm, noise_std)

    # apply PBC in all directions or not at all
    if jnp.array(metadata["periodic_boundary_conditions"]).any():
        displacement_fn, shift_fn = space.periodic(side=jnp.array(box))
    else:
        displacement_fn, shift_fn = space.free()
        
    def temp_diff_fn(temp1: jnp.ndarray, temp2: jnp.ndarray):     #######
        return temp2-temp1
    
    def temp_shift_fn(temp: jnp.ndarray, dT: jnp.ndarray):
    # Define your logic for shifting temperature based on displacement
        return temp + dT

    displacement_fn_set = vmap(displacement_fn, in_axes=(0, 0))
    temp_displacement_fn_set = vmap(temp_diff_fn, in_axes=(0,0))        ##########

    neighbor_fn = neighbor_list(
        displacement_fn,
        jnp.array(box),
        backend=neighbor_list_backend,
        r_cutoff=metadata["default_connectivity_radius"],
        capacity_multiplier=neighbor_list_multiplier,
        mask_self=False,
        format=NeighborListFormat.Sparse,
        num_particles_max=metadata["num_particles_max"],
        pbc=metadata["periodic_boundary_conditions"],
    )

    feature_transform = physical_feature_builder(
        bounds=metadata["bounds"],
        normalization_stats=normalization_stats,
        connectivity_radius=metadata["default_connectivity_radius"],
        displacement_fn=displacement_fn,
        pbc=metadata["periodic_boundary_conditions"],
        magnitude_features=magnitude_features,
        external_force_fn=external_force_fn,
    )

    def _compute_target(pos_input: jnp.ndarray, temp_input: jnp.ndarray) -> TargetDict:
        # displacement(r1, r2) = r1-r2  # without PBC

        n_total_points = pos_input.shape[0]
        current_velocity = displacement_fn_set(pos_input[:, 1], pos_input[:, 0])
        next_velocity = displacement_fn_set(pos_input[:, 2], pos_input[:, 1])
        current_acceleration = next_velocity - current_velocity

        acc_stats = normalization_stats["acceleration"]
        normalized_acceleration = (
            current_acceleration - acc_stats["mean"]
        ) / acc_stats["std"]

        vel_stats = normalization_stats["velocity"]
        normalized_velocity = (next_velocity - vel_stats["mean"]) / vel_stats["std"]

        current_temp_diff = temp_displacement_fn_set(temp_input[:, 1], temp_input[:, 0])        ##### added temp_diff
        next_temp_diff =  temp_displacement_fn_set(temp_input[:, 2], temp_input[:, 1])          ##### treat it like velocity

        '''
        current_temp_diff = temp_displacement_fn_set(temp_input[:, 1][:,jnp.newaxis], temp_input[:, 0][:,jnp.newaxis])        ##### added temp_diff
        new_temp = jnp.hstack((current_temp_diff[:, 0:1], current_temp_diff[:, 2:]))
        current_temp_diff = new_temp.reshape(n_total_points, -1)
        next_temp_diff =  temp_displacement_fn_set(temp_input[:, 2][:,jnp.newaxis], temp_input[:, 1][:,jnp.newaxis])          ##### treat it like velocity
        new_temp2 = jnp.hstack((next_temp_diff[:, 0:1], next_temp_diff[:, 2:]))
        next_temp_diff = new_temp2.reshape(n_total_points, -1)
        '''
        temp_diff_stats = normalization_stats["temp_diff"]                                      ##### get stats
        normalized_temp_diff = (next_temp_diff - temp_diff_stats["mean"]) / temp_diff_stats["std"]  #### normalize temp_diff

        return {
            "acc": normalized_acceleration,
            "vel": normalized_velocity,
            "pos": pos_input[:, -1],
            "temp": temp_input[:, -1],              ###### added temp_input
            "temp_diff" : normalized_temp_diff[:, jnp.newaxis],     ##### added temp_diff
        }

    def _preprocess(
        sample: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], ##### added another input value for temp_input
        neighbors: Optional[NeighborList] = None,
        is_allocate: bool = False,
        mode: str = "train",
        **kwargs,  # key, noise_std, unroll_steps
    ) -> Union[TrainCaseOut, EvalCaseOut]:
        pos_input = jnp.asarray(sample[0], dtype=dtype)
        particle_type = jnp.asarray(sample[1])
        temp_input = jnp.asarray(sample[2], dtype=dtype)            ##### temperature is third input 

        #threshold = 3 * 0.02
        #indices = pos_input[:,:, 0] < threshold
        #temp_input = jnp.where(indices, 1.0 , temp_input)

        if mode == "train":
            key, noise_std = kwargs["key"], kwargs["noise_std"]
            unroll_steps = kwargs["unroll_steps"]
            if pos_input.shape[1] > 1:
                key, pos_input, temp_input = add_gns_noise(
                    key, pos_input, temp_input, particle_type, input_seq_length, noise_std, shift_fn
                )

        # allocate the neighbor list
        most_recent_position = pos_input[:, input_seq_length - 1]
        num_particles = (particle_type != -1).sum()
        if is_allocate:
            neighbors = neighbor_fn.allocate(
                most_recent_position, num_particles=num_particles
            )
        else:
            neighbors = neighbors.update(
                most_recent_position, num_particles=num_particles
            )

        # selected features
        features = feature_transform(pos_input[:, :input_seq_length], temp_input[:, :input_seq_length], neighbors)   #### added temp-input in features
        

        if mode == "train":
            # compute target acceleration. Inverse of postprocessing step.
            # the "-2" is needed because we need the most recent position and one before
            slice_begin = (0, input_seq_length - 2 + unroll_steps, 0)
            slice_size = (pos_input.shape[0], 3, pos_input.shape[2])

            slice_begin2 = (0, input_seq_length - 2 + unroll_steps)         ##### slice for temp_input similar to position
            slice_size2 = (temp_input.shape[0], 3)

            target_dict = _compute_target(
                lax.dynamic_slice(pos_input, slice_begin, slice_size), 
                lax.dynamic_slice(temp_input, slice_begin2, slice_size2)    #### added temp_input to target dict 
            )
            
            return key, features, target_dict, neighbors
        if mode == "eval":
            return features, neighbors

    def allocate_fn(key, sample, noise_std=0.0, unroll_steps=0):
        return _preprocess(
            sample,
            key=key,
            noise_std=noise_std,
            unroll_steps=unroll_steps,
            is_allocate=True,
        )

    @jit
    def preprocess_fn(key, sample, noise_std, neighbors, unroll_steps=0):
        return _preprocess(
            sample, neighbors, key=key, noise_std=noise_std, unroll_steps=unroll_steps
        )

    def allocate_eval_fn(sample):
        
        return _preprocess(sample, is_allocate=True, mode="eval")

    @jit
    def preprocess_eval_fn(sample, neighbors):
        return _preprocess(sample, neighbors, mode="eval")

    @jit
    def integrate_fn(normalized_in, position_sequence):
        """Euler integrator to get position shift."""
        assert any([key in normalized_in for key in ["pos", "vel", "acc"]])

        if "pos" in normalized_in:
            # Zeroth euler step
            return normalized_in["pos"]
        else:
            most_recent_position = position_sequence[:, -1]
            if "vel" in normalized_in:
                # invert normalization
                velocity_stats = normalization_stats["velocity"]
                new_velocity = velocity_stats["mean"] + (
                    normalized_in["vel"] * velocity_stats["std"]
                )
            elif "acc" in normalized_in:
                # invert normalization.
                acceleration_stats = normalization_stats["acceleration"]
                acceleration = acceleration_stats["mean"] + (
                    normalized_in["acc"] * acceleration_stats["std"]
                )
                # Second Euler step
                most_recent_velocity = displacement_fn_set(
                    most_recent_position, position_sequence[:, -2]
                )
                new_velocity = most_recent_velocity + acceleration  # * dt = 1

            # First Euler step
            return shift_fn(most_recent_position, new_velocity)
    
    @jit
    def integrate_temp_fn(normalized_in, position_sequence ,temp_sequence):
        """Euler integrator to get position shift."""
        assert any([key in normalized_in for key in ["temp", "temp_diff"]])

        if "temp" in normalized_in:
            # Zeroth euler step
            return normalized_in["temp"]
        
        else:
            #n_total_points = temp_sequence.shape[0]
            most_recent_temperature = temp_sequence[:, -1]

            first_velocity = displacement_fn_set(position_sequence[:, 2], position_sequence[:, 1])
            temp_diff_stats = normalization_stats["temp_diff"]
            new_temp_diff = temp_diff_stats["mean"] + (
                    normalized_in["temp_diff"] * temp_diff_stats["std"]
            )
            acceleration_stats = normalization_stats["acceleration"]
            acceleration = acceleration_stats["mean"] + (
            normalized_in["acc"] * acceleration_stats["std"]
            )
            # Second Euler step
            most_recent_position = position_sequence[:, -1]
            most_recent_velocity = displacement_fn_set(
            most_recent_position, position_sequence[:, -2]
            )
            second_velocity = most_recent_velocity + acceleration  # * dt = 1
            velocity_product = jnp.sqrt(jnp.sum(second_velocity * second_velocity, axis=1))
            velocity_product = velocity_product[:,jnp.newaxis]
            vel_temp_diff_product = jnp.sum(velocity_product*new_temp_diff, axis=1)
            advected_temperature = most_recent_temperature - 50.0*vel_temp_diff_product   #+new_temp_diff + jnp.ravel(new_temp_diff)
            #alpha = float(0.03162 / (1011 * 1000))
            diffused_temperature = advected_temperature + 0.0*(temp_sequence[:,-2]- 2*temp_sequence[:,-1] + temp_sequence[:,0]) + 1.0*jnp.ravel(new_temp_diff)
            #diffused_temperature = advected_temperature + jnp.ravel(new_temp_diff) #+  
            diffused_temperature = jnp.clip(diffused_temperature, 1.0, 1.23)
            


            '''
            most_recent_temp = temp_sequence[:, -1]
            most_recent_temp = most_recent_temp[:,jnp.newaxis]
            if "temp_diff" in normalized_in:
                # invert normalization
                temp_diff_stats = normalization_stats["temp_diff"]
                new_temp_diff = temp_diff_stats["mean"] + (
                    normalized_in["temp_diff"] * temp_diff_stats["std"]
                )
            '''
            '''    
            elif "acc" in normalized_in:
                # invert normalization.
                acceleration_stats = normalization_stats["acceleration"]
                acceleration = acceleration_stats["mean"] + (
                    normalized_in["acc"] * acceleration_stats["std"]
                )
                # Second Euler step
                most_recent_velocity = displacement_fn_set(
                    most_recent_position, position_sequence[:, -2]
                )
                new_velocity = most_recent_velocity + acceleration  # * dt = 1
            '''
            # First Euler step
            return diffused_temperature #temp_shift_fn(most_recent_temp, new_temp_diff)
    
    return CaseSetupFn(
        allocate_fn,
        preprocess_fn,
        allocate_eval_fn,
        preprocess_eval_fn,
        integrate_fn,
        integrate_temp_fn,
        displacement_fn,
        normalization_stats,
        temp_diff_fn,
    )
