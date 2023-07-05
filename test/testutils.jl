# This file is a part of HeterogeneousComputing.jl, licensed under the MIT License (MIT).

import ArraysOfArrays, FillArrays, StructArrays


function gen_testdata()
    (
        x = rand(Float32),
        A = rand(Float16, 3, 4, 5),
        str = "Hello, World!",
        sym = :SomeSymbol,
        sa =  StructArrays.StructArray((
            a = rand(Float16, 100),
            b = ArraysOfArrays.VectorOfSimilarVectors(rand(Float32, 10, 100)),
            c = FillArrays.Fill(Float16(1.5), 100),
            d = rand(-7:15, 100)
        ))
    )
end


function gen_testclosure()
    data = gen_testdata()
    return function _testclosure(args...)
        merge(data, (args = args,))
    end
end
