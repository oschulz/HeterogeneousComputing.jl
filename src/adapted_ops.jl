# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

const _CPUArrayLike = Union{Array, SubArray{<:Any,<:Any,<:Array}}
const _LinIndexCPUArrayLike = Union{Array, SubArray{<:Any,<:Any,<:Array,<:Any,true}}
const _NonLinIndexView = SubArray{<:Any,<:Any,<:Array,<:Any,false}


# Override for adapters that need special handling, to convert them to other adapters:
@inline pp_adapter(to::Adapter) where Adapter = to
@inline pp_adapter(::Type{Adapter}) where Adapter = Adapter
@inline pp_adapter(m::Module) = pp_module_adapter(Val(nameof(m)))

pp_module_adapter(Base.@nospecialize(module_name::Val)) = throw(ArgumentError("No default pp_adapter defined for module $(only(typeof(module_name).parameters))"))


const _CurriedAdapt{Adapter} = Base.Fix1{typeof(Adapt.adapt), Adapter}


"""
    adaptfunc(to)

Return a function that adapts objects to 

Returns a function that is semantically equivalent to
`obj -> Adapt.adapt(pp_adapter(to), obj)`.

Examples:

```julia
⋮ = adaptfunc(Strided.StridedView)
⋰ = adaptfunc(StrideArrays.StrideArray)
# etc.
```

but you can also use (potentially less type-stable):

```julia
adaptfunc(Strided)
adaptfunc(StrideArrays)
# etc.
```
"""
function adaptfunc end

@inline adaptfunc(to::Adapter) where Adapter = Base.Fix1(Adapt.adapt, pp_adapter(to))
@inline adaptfunc(::Type{Adapter}) where Adapter = Base.Fix1(Adapt.adapt, pp_adapter(Adapter))


"""
    adapted_call(f, af, args...)

Call the function `f` with the arguments `args...`, with both `f` (in case
`f` is a closure`) and `args...` adapted via the function `af`.

Semanically equivalent to `af(f)(map(af, args)...)`.

`af` will typically be the result of `af = adaptfunc(something_to_adapt_to)`
(see [`adapted_bcast`](@ref) for examples regarding `af`).
"""
function adapted_call end

@inline adapted_call(f::F, af::_CurriedAdapt, args...) where F = af(f)(map(af, args)...)
@inline adapted_call(f::F, to::A, args...) where {F,A} = adapted_call(f, adaptfunc(to), args...)
@inline adapted_call(f::F, ::Type{Adapter}, args...) where {F,Adapter} = adapted_call(f, adaptfunc(Adapter), args...)


"""
    adapted_bcast(f, af, args...)

Broadcast function `f` over the arguments `args...`, with both `f` (in case
`f` is a closure`)` and `args...` adapted via the function `af`.

Semanically equivalent to `broadcast(af(f), map(af, args)...)`.

`af` will typically be the result of `af = adaptfunc(something_to_adapt_to)`.

For example:

```julia
⋮ = adaptfunc(Strided.StridedView)
⋰ = adaptfunc(StrideArrays.StrideArray)

adapted_bcast(⋮, f, A, B) # Use Strided to multi-thread f.(A, B)
adapted_bcast(⋰, f, A, B) # Use StrideArrays to multi-thread f.(A, B)
```

See also [`adaptfunc`](@ref).
"""
function adapted_bcast end

@inline adapted_bcast(f::F, af::_CurriedAdapt, args...) where F =  broadcast(af(f), map(af, args)...)
@inline adapted_bcast(f::F, to::A, args...) where {F,A} = adapted_bcast(f, adaptfunc(to), args...)
@inline adapted_bcast(f::F, ::Type{Adapter}, args...) where {F,Adapter} = adapted_bcast(f, adaptfunc(Adapter), args...)

@inline adapted_bcast!(f::F, af::_CurriedAdapt, args...) where F = broadcast!(af(f), map(af, args)...)
@inline adapted_bcast!(f::F, to::A, args...) where {F,A} = adapted_bcast!(f, adaptfunc(to), args...)
@inline adapted_bcast!(f::F, ::Type{Adapter}, args...) where {F,Adapter} = adapted_bcast!(f, adaptfunc(Adapter), args...)


"""
    struct SpecialTypeAdapter{T} end

Adapter to an (array) type `T` (for `Adapt`) that requires special handling.

# Implementation

`SpecialTypeAdapter` instances should result from calls to
`ParallelProcessingTools.pp_adapter(to_something)`, they should not be
constructed by user code directly.

There is no default specialization for
`Adapt.adapt_storage(::SpecialTypeAdapter, obj)`, it should be specialized for
each type `T` individually.
"""
struct SpecialTypeAdapter{T} end

const PPTypeAdaptFunc{T} = _CurriedAdapt{SpecialTypeAdapter{T}}
Base.show(@nospecialize(io::IO), @nospecialize(f::PPTypeAdaptFunc{T})) where T = print(io, "adaptfunc($(nameof(T)))")
Base.show(@nospecialize(io::IO), ::MIME"text/plain", @nospecialize(f::PPTypeAdaptFunc{T})) where T = show(io, f)

@inline pp_adapter(::Type{Array}) = SpecialTypeAdapter{Array}()
Adapt.adapt_storage(::SpecialTypeAdapter{Array}, A::AbstractArray) = Array(A)
