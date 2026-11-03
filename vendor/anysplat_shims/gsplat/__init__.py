"""AnySplat imports gsplat for its training-time renderer only; inference never rasterizes."""


def rasterization(*a, **k):
    raise RuntimeError("gsplat rasterization is not available (CUDA-only); mercymek renders splats in the browser")
