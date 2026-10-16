# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

# Field recursion of `get_compute_unit`. Included last, since generated
# functions only see code defined before them.

# `visited` is `nothing` or a `Base.IdSet` of visited objects that may be part of
# reference loops:
@inline _get_cunit(x, visited) = _get_cunit_via(get_compute_unit_impl(x), x, visited)

@inline _get_cunit_via(cunit, @nospecialize(x), @nospecialize(visited)) = cunit
@inline _get_cunit_via(::_RecurseFields, x, visited) = _get_fields_cunit(x, visited)

const _max_inline_depth = 8

function _fields_cunit_expr(@nospecialize(T::Type), xex, vex, depth::Int)
    _is_primitive_data(T) && return :(ComputeUnitIndependent())
    # Types that may close reference loops need to be tracked:
    depth > 0 && _may_close_ref_loop(T) && return :(_get_fields_cunit($xex, $vex))
    body = Expr(:block)
    cunit = gensym(:cunit)
    push!(body.args, :($cunit = ComputeUnitIndependent()))
    for i in 1:fieldcount(T)
        FT = fieldtype(T, i)
        f = gensym(:field)
        fcunit = Expr(:let, :($f = getfield($xex, $i)), _field_cunit_expr(FT, f, vex, depth))
        if ismutabletype(T) && !isbitstype(FT)
            fcunit = :(isdefined($xex, $i) ? $fcunit : ComputeUnitIndependent())
        end
        new_cunit = gensym(:cunit)
        push!(body.args, :($new_cunit = merge_compute_units($fcunit, $cunit)))
        cunit = new_cunit
    end
    push!(body.args, cunit)
    return body
end

function _field_cunit_expr(@nospecialize(FT::Type), f::Symbol, vex, depth::Int)
    if !isconcretetype(FT) || _has_cunit_rule(FT) || depth >= _max_inline_depth
        return :(_get_cunit($f, $vex))
    else
        rule = gensym(:rule)
        return quote
            $rule = get_compute_unit_impl($f)
            $rule isa _RecurseFields ? $(_fields_cunit_expr(FT, f, vex, depth + 1)) : $rule
        end
    end
end

# Primitive types and tuples of them (e.g. the data of static arrays) are
# independent of compute units, so there's no need to generate code for
# each element:
_is_primitive_data(@nospecialize(T::Type)) = isprimitivetype(T) || (T <: Tuple && all(isprimitivetype, fieldtypes(T)))

# Types with compute unit rules in HeterogeneousComputing itself, no need to
# inline recursion code for them:
function _has_cunit_rule(@nospecialize(T::Type))
    return T <: Union{Type,Array,AbstractComputeUnit,GPUArraysCore.AbstractGPUArray,AbstractRNG,GenContext} ||
           (isdefined(Core, :Memory) && T <: Core.Memory)
end

# Whether objects of type T may be part of a reference loop. Loops need a
# mutable object that can reach itself, through fields of non-concrete type
# or through a chain of concretely typed fields.
function _may_close_ref_loop(@nospecialize(T::Type))
    ismutabletype(T) || return false
    seen = Set{Any}()
    pending = Any[fieldtypes(T)...]
    while !isempty(pending)
        FT = pop!(pending)
        isbitstype(FT) && continue
        (!isconcretetype(FT) || FT === T) && return true
        FT in seen && continue
        push!(seen, FT)
        append!(pending, fieldtypes(FT))
    end
    return false
end

struct _AlreadyVisited end

_visit!(::Nothing, x) = _visit!(Base.IdSet{Any}(), x)
_visit!(visited::Base.IdSet{Any}, x) = x in visited ? _AlreadyVisited() : push!(visited, x)


# The recursion through fields of concrete types is inlined into a single
# generated function body, since inference gives up on recursive calls of
# the same method for nested types:
@generated function _get_fields_cunit(x, visited)
    body = Expr(:block)
    if _may_close_ref_loop(x)
        push!(body.args, :(visited = _visit!(visited, x)))
        push!(body.args, :(visited isa _AlreadyVisited && return ComputeUnitIndependent()))
    end
    push!(body.args, _fields_cunit_expr(x, :x, :visited, 0))
    return body
end
