from __future__ import annotations
import numpy as np
from typing import List
from selfplay import SingleSample, DualTurnSample


def rotate_2d_array(arr_2d: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr_2d, k=k)


def flip_2d_array(arr_2d: np.ndarray) -> np.ndarray:
    return np.fliplr(arr_2d)


def rotate_obs(obs: np.ndarray, k: int) -> np.ndarray:
    return np.array([rotate_2d_array(obs[i], k) for i in range(obs.shape[0])])


def flip_obs(obs: np.ndarray) -> np.ndarray:
    return np.array([flip_2d_array(obs[i]) for i in range(obs.shape[0])])


def rotate_policy(pi: np.ndarray, k: int, board_size: int = 13) -> np.ndarray:
    pi_2d = pi.reshape(board_size, board_size)
    pi_2d_rot = rotate_2d_array(pi_2d, k)
    return pi_2d_rot.flatten()


def flip_policy(pi: np.ndarray, board_size: int = 13) -> np.ndarray:
    pi_2d = pi.reshape(board_size, board_size)
    pi_2d_flip = flip_2d_array(pi_2d)
    return pi_2d_flip.flatten()


def rotate_mask(mask: np.ndarray, k: int, board_size: int = 13) -> np.ndarray:
    mask_2d = mask.reshape(board_size, board_size)
    mask_2d_rot = rotate_2d_array(mask_2d, k)
    return mask_2d_rot.flatten()


def flip_mask(mask: np.ndarray, board_size: int = 13) -> np.ndarray:
    mask_2d = mask.reshape(board_size, board_size)
    mask_2d_flip = flip_2d_array(mask_2d)
    return mask_2d_flip.flatten()


def augment_single_sample(sample, board_size: int = 13):
    augmented = []
    
    if isinstance(sample, dict):
        obs = sample["obs"]
        pi = sample["pi"]
        z = sample["z"]
        mask = sample["mask"]
    else:
        obs = sample.obs
        pi = sample.pi
        z = 0.0  
        mask = sample.mask
    
    for k in range(4):
        obs_rot = rotate_obs(obs, k)
        pi_rot = rotate_policy(pi, k, board_size)
        mask_rot = rotate_mask(mask, k, board_size)
        
        augmented.append({
            "obs": obs_rot,
            "pi": pi_rot,
            "z": z,
            "mask": mask_rot
        })
        
        obs_flip = flip_obs(obs_rot)
        pi_flip = flip_policy(pi_rot, board_size)
        mask_flip = flip_mask(mask_rot, board_size)
        
        augmented.append({
            "obs": obs_flip,
            "pi": pi_flip,
            "z": z,
            "mask": mask_flip
        })
    
    return augmented


def augment_single_samples(samples, board_size: int = 13):
    augmented = []
    for sample in samples:
        augmented.extend(augment_single_sample(sample, board_size))
    return augmented

def augment_dual_sample(sample, board_size: int = 13):
    augmented = []
    
    if isinstance(sample, dict):
        obs_first = sample["obs"]
        cond_first = sample["cond_first"]
        pi1 = sample["pi1"]
        pi2 = sample["pi2"]
        z = sample["z"]
        mask1 = sample["mask1"]
        mask2 = sample["mask2"]
    else:
        obs_first = sample.obs_first
        cond_first = sample.cond_first
        pi1 = sample.pi1
        pi2 = sample.pi2
        z = 0.0  
        mask1 = sample.mask1
        mask2 = sample.mask2
    
    for k in range(4):
        obs_rot = rotate_obs(obs_first, k)
        cond_rot = rotate_obs(cond_first, k)
        
        pi1_rot = rotate_policy(pi1, k, board_size)
        pi2_rot = rotate_policy(pi2, k, board_size)
        
        mask1_rot = rotate_mask(mask1, k, board_size)
        mask2_rot = rotate_mask(mask2, k, board_size)
        
        augmented.append({
            "obs": obs_rot,
            "cond_first": cond_rot,
            "pi1": pi1_rot,
            "pi2": pi2_rot,
            "z": z,
            "mask1": mask1_rot,
            "mask2": mask2_rot
        })
        
        obs_flip = flip_obs(obs_rot)
        cond_flip = flip_obs(cond_rot)
        pi1_flip = flip_policy(pi1_rot, board_size)
        pi2_flip = flip_policy(pi2_rot, board_size)
        mask1_flip = flip_mask(mask1_rot, board_size)
        mask2_flip = flip_mask(mask2_rot, board_size)
        
        augmented.append({
            "obs": obs_flip,
            "cond_first": cond_flip,
            "pi1": pi1_flip,
            "pi2": pi2_flip,
            "z": z,
            "mask1": mask1_flip,
            "mask2": mask2_flip
        })
    
    return augmented


def augment_dual_samples(samples, board_size: int = 13):
    augmented = []
    for sample in samples:
        augmented.extend(augment_dual_sample(sample, board_size))
    return augmented


def get_augmentation_stats(original_count: int, augmented_count: int) -> dict:
    multiplier = augmented_count / original_count if original_count > 0 else 0
    return {
        "original_samples": original_count,
        "augmented_samples": augmented_count,
        "multiplier": multiplier,
        "expected_multiplier": 8
    }

