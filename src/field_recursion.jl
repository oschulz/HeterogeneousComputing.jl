# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

# Field recursion of `get_compute_unit`. Included last, since generated
# functions only see code defined before them.

# Sibling paths share an identity set, created when a possible loop is reached.
@inline _get_cunit(x, visited) = _get_cunit_via(get_compute_unit_impl(x), x, visited)

@inline _get_cunit_via(cunit, @nospecialize(x), @nospecialize(visited)) = cunit
@inline _get_cunit_via(::_RecurseFields, x, visited) = _get_fields_cunit(x, visited)

const _max_inline_depth = 8
const _max_inline_fields = 64

function _fields_cunit_expr(@nospecialize(T::Type), xex, vex, depth::Int, budget::Int = _max_inline_fields)
    n = fieldcount(T)
    n == 0 && return :(ComputeUnitIndependent())
    # Keep homogeneous primitive tuples (e.g. static array data) compact,
    # while allowing individual elements to have their own compute unit rules.
    T <: Tuple && isprimitivetype(eltype(T)) && return :(_loop_fields_cunit($xex, $vex))
    # Types that may close reference loops need to be tracked:
    depth > 0 && _may_close_ref_loop(T) && return :(_get_fields_cunit($xex, $vex))
    n > _max_inline_fields && return :(_loop_fields_cunit($xex, $vex))
    n > budget && return :(_get_fields_cunit($xex, $vex))
    # Reserve one field access per sibling, then share the remaining budget.
    child_budget = (budget - n) ÷ n
    body = Expr(:block)
    if n > 1
        shared = gensym(:visited)
        push!(body.args, :($shared = _share_visited($vex)))
        vex = shared
    end
    cunit = gensym(:cunit)
    push!(body.args, :($cunit = ComputeUnitIndependent()))
    for i in 1:n
        FT = fieldtype(T, i)
        f = gensym(:field)
        fcunit = Expr(:let, :($f = getfield($xex, $i)), _field_cunit_expr(FT, f, vex, depth, child_budget))
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

function _field_cunit_expr(@nospecialize(FT::Type), f::Symbol, vex, depth::Int, budget::Int)
    if !isconcretetype(FT) || _has_cunit_rule(FT) || depth >= _max_inline_depth || budget == 0
        return :(_get_cunit($f, $vex))
    else
        rule = gensym(:rule)
        return quote
            $rule = get_compute_unit_impl($f)
            $rule isa _RecurseFields ? $(_fields_cunit_expr(FT, f, vex, depth + 1, budget)) : $rule
        end
    end
end

@inline function _loop_field_cunit(x, i, visited)
    ismutabletype(typeof(x)) && !isdefined(x, i) && return ComputeUnitIndependent()
    # The caller visits exactly 1:fieldcount(typeof(x)) on a nonempty record.
    return _get_cunit(getfield(x, i, false), visited)
end

function _loop_fields_cunit(x, visited)
    visited = _share_visited(visited)
    cunit = _loop_field_cunit(x, 1, visited)
    for i in 2:fieldcount(typeof(x))
        cunit = merge_compute_units(_loop_field_cunit(x, i, visited), cunit)
    end
    return cunit
end

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
        # Ordinary arrays have a compute unit rule and never recurse into storage.
        (isbitstype(FT) || FT <: Array) && continue
        if FT isa Union
            push!(pending, FT.a, FT.b)
            continue
        end
        (!isconcretetype(FT) || FT === T) && return true
        FT in seen && continue
        push!(seen, FT)
        append!(pending, fieldtypes(FT))
    end
    return false
end

_share_visited(visited) = visited
_share_visited(::Nothing) = Ref{Base.IdSet{Any}}()

_visit!(::Nothing, x) = push!(Base.IdSet{Any}(), x)
_visit!(seen::Base.IdSet{Any}, x) = x in seen ? nothing : push!(seen, x)
function _visit!(shared::Base.RefValue{Base.IdSet{Any}}, x)
    isassigned(shared) || (shared[] = Base.IdSet{Any}())
    return _visit!(shared[], x)
end


# The recursion through fields of concrete types is inlined into a single
# generated function body, since inference gives up on recursive calls of
# the same method for nested types:
@generated function _get_fields_cunit(x, visited)
    body = Expr(:block)
    if _may_close_ref_loop(x)
        push!(body.args, :(visited = _visit!(visited, x)))
        push!(body.args, :(visited === nothing && return ComputeUnitIndependent()))
    end
    push!(body.args, _fields_cunit_expr(x, :x, :visited, 0))
    return body
end
