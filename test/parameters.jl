module TestParameters

# using Revise
using MLJ
using Test

# test unwinding of iterators
iterators = ([1, 2], ["a","b"], ["x", "y", "z"])
@test MLJ.unwind(iterators...) ==
[1  "a"  "x";
 2  "a"  "x";
 1  "b"  "x";
 2  "b"  "x";
 1  "a"  "y";
 2  "a"  "y";
 1  "b"  "y";
 2  "b"  "y";
 1  "a"  "z";
 2  "a"  "z";
 1  "b"  "z";
 2  "b"  "z"]

end
true
