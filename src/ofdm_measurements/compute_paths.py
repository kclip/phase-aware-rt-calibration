from sionna.rt import Scene, Paths


def compute_paths_traces(scene: Scene, max_depth: int, num_samples: int, check_scene: bool = False):
    return scene.trace_paths(
        max_depth=max_depth,
        method="fibonacci",
        num_samples=num_samples,
        diffraction=False, scattering=False, edge_diffraction=False,
        check_scene=check_scene
    )


def compute_paths_from_traces(scene: Scene, traced_paths):
    paths = scene.compute_fields(*traced_paths)
    paths.normalize_delays = False

    return paths


def compute_paths(scene: Scene, max_depth: int, num_samples: int, check_scene: bool = False) -> Paths:
    traced_paths = compute_paths_traces(
        scene=scene,
        max_depth=max_depth,
        num_samples=num_samples,
        check_scene=check_scene
    )
    return compute_paths_from_traces(scene=scene, traced_paths=traced_paths)
