"""Host-side I/O for the GPU pipeline (the only package where host arrays are handled).

Everything here reads or writes bytes, or performs the ONE-TIME string-id -> int64 encoding
(cached on /scratch). Computation happens in ``ber.gpu`` on CUDA tensors.
"""
