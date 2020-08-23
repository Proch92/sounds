using ProfileView
using MLDatasets
using Flux
using Statistics
include("gan.jl")
import .Gan


function main()
	train_x, train_y = MNIST.traindata(Float32)
	test_x,  test_y  = MNIST.testdata(Float32)
	tensors = cat(train_x, test_x, dims=3)
	tensors = reshape(tensors, (28*28, size(tensors)[3]))
	tensors = @.(2f0 * tensors - 1f0)
	tensors = tensors |> gpu

	println(size(tensors))

	model = Gan.newmodel()
	Gan.train_gan(model, tensors, 1)
	Gan.savemodel(model)
	Gan.plot_syntethic(model)
end

main()