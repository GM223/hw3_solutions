const NUMQUESTIONS = 2
if !(@isdefined SUFFIX)
    SUFFIX = ""
end

questions_to_grade = collect(1:NUMQUESTIONS)
if !isempty(ARGS)
    questions_to_grade = Int[]
    for arg in ARGS
        question = tryparse(Int, arg)
        if !isnothing(question) && (0 < question <= NUMQUESTIONS)
            push!(questions_to_grade, question)
        end
    end
    if isempty(questions_to_grade)
        @warn "Couldn't parse any of the input arguments as question numbers. Enter the question to grade, separated by spaced."
    end
    sort!(questions_to_grade)
end


# Create a module for each question
for i in 1:NUMQUESTIONS 
@eval module $(Symbol("Q" * string(i) * SUFFIX))
include("autograder.jl")
getname() = string(split(string(@__MODULE__), ".")[end])
grade() = Autograder.gradequestion(getname()) 
checktestsets(solutiondir=joinpath(@__DIR__, "..")) = Autograder.checktestsets(getname(), solutiondir)
end
end

solutiondir = joinpath(@__DIR__, "..")

# Grade all of the questions
modules = [@eval $(Symbol("Q" * string(i) * SUFFIX)) for i = 1:NUMQUESTIONS]
results = map(questions_to_grade) do question 
    mod = modules[question]
    mod.checktestsets(solutiondir)
    mod.grade()[1]
end