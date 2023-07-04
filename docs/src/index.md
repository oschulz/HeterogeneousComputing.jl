# HeterogeneousComputing.jl

HeterogeneousComputing.jl provides tools to ease heterogeneous computing in Julia.

Currently only CPU and CUDA units are supports, support for AMDGPU, oneAPI and Metal will be added as well.

The package provides [`AbstractComputeUnit`](@ref) and [`GenContext`](@ref) which allow for determining which compute unit given content is currently located on, moving it to other compute units, and generating new data on specific compute units.

The idea is that a `AbstractComputeUnit` or an `AbstractComputeUnit` (which compines a compute unit with a random number generator and a desired numerical precision) can be passed around and propagate through an application, so that the computational context is always available.

Example:

```julia
using HeterogeneousComputing
using Random, Adapt, CUDA, StructArrays, Tables

# Generate some data on a GPU:
cpu_data = StructArray((
    a = rand(Float32, 1000),
    b = rand(Float32, 1000),
))

# Verify data is on GPU:
cpu_unit = CPUnit()
get_compute_unit(cpu_data) == cpu_unit

# Get compute unit for the current CUDA device:
gpu_unit = AbstractComputeUnit(CUDA.device())

# Move the data to the GPU:

gpu_data = adapt(gpu_unit, cpu_data)

# Specify a context for content generation.

gpu_ctx = GenContext{Float32}(gpu_unit)

# Generate a new data column on the GPU and add it to the data. A context
# is also an `AbstractRNG`, so we can generate random data of the desired
# numerical precision on the chosed unit very conventiently:

c = rand!(allocate_array(gpu_ctx, 1000))
d = rand(gpu_ctx, 1000)

new_gpu_data = new_gpu_data = StructArray(
    merge(Tables.columns(gpu_data), (c = c, d = c))
)

# Move the data back to the CPU:

new_cpu_data = adapt(cpu_unit, new_gpu_data)
```

# Support for KernelAbstractions

HeterogeneousComputing also has support for
[`KernelAbstractions`](https://github.com/JuliaGPU/KernelAbstractions.jl),
a `KernelAbstractions.Backend` can be derived from an `AbstractComputeUnit`:

```julia
using KernelAbstractions
ka_backend = KernelAbstractions.Backend(gpu_unit)
```

This way, KernelAbstractions backends don't need to be tracked independently through a an application.
