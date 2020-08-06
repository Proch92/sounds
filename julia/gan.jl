__precompile__()
module Gan
export train_gan, plot_syntethic, newmodel, savemodel, loadmodel

using MLDatasets
using Flux
using Flux: Losses.binarycrossentropy, throttle
# using Flux.Optimise: train!, update!
using Plots
using ProgressMeter
using BSON: @save, @load
using TensorBoardLogger, Logging
global_logger(TBLogger("tensorboard_logs/run", min_level=Logging.Info))


Discriminator() = Chain(
	Dense(28*28, 1024, leakyrelu),
	Dropout(0.2),
	Dense(1024, 512, leakyrelu),
	Dropout(0.2),
	Dense(512, 256, leakyrelu),
	Dropout(0.2),
	Dense(256, 1, sigmoid)
)

Generator() = Chain(
	Dense(100, 256, leakyrelu),
	Dense(256, 512, leakyrelu),
	Dense(512, 1024, leakyrelu),
	Dense(1024, 28*28, tanh)
)

mutable struct Model
	discriminator::Chain
	generator::Chain
end

function savemodel(model::Model)
	@save "ganmodel.bson" model
	return nothing
end

function loadmodel()
	model = newmodel()
    @load "ganmodel.bson" model
    return model
end

function newmodel()
    return Model(Discriminator(), Generator())
end

function noise(m)
    return randn(100, m)
end

function train_discriminator(model, opt, real, batchsize)
	targetreal = ones(batchsize)
	targetfake = zeros(batchsize)
	fake = model.generator(noise(batchsize))
	# loss(x, y) = binarycrossentropy(model.discriminator(x), y)

	# println(size(real))
	# println(size(fake))
	# preds = model.discriminator(real)
	# println(size(preds))
	
	# preds = model.discriminator(real)
	# loglossreal() = @info "discloss_real" binarycrossentropy(preds, targetreal)
	# data = zip(preds, targetreal)
	# train!(
	# 	binarycrossentropy,
	# 	params(model.discriminator),
	# 	data,
	# 	opt,
	# 	cb=throttle(loglossreal, 5)
	# )

	# preds = model.discriminator(fake)
	# loglossfake() = @info "discloss_fake" binarycrossentropy(preds, targetfake)
	# data = zip(preds, targetfake)
	# train!(
	# 	binarycrossentropy,
	# 	params(model.discriminator),
	# 	data,
	# 	opt,
	# 	cb=throttle(loglossfake, 5)
	# )

	loss(x, y) = binarycrossentropy(model.discriminator(x), y)
	Θ = params(model.discriminator)
	
	grads = gradient(() -> loss(real, targetreal), Θ)
	for p in Θ
		Flux.Optimise.update!(opt, p, grads[p])
	end

	grads = gradient(() -> loss(fake, targetfake), Θ)
	for p in Θ
		Flux.Optimise.update!(opt, p, grads[p])
	end

	@info "discloss_real" loss(real, targetreal)
	@info "discloss_fake" loss(fake, targetfake)
end

function train_generator(model, opt, batchsize)
	target = ones(batchsize)
	# loss(x, y) = binarycrossentropy(model.discriminator(x), y)

	# preds = model.discriminator(fake)
	# loglossgen() = @info "genloss" binarycrossentropy(preds, target)
	# data = zip(preds, target)
	# train!(
	# 	binarycrossentropy,
	# 	params(model.generator),
	# 	data,
	# 	opt,
	# 	cb=throttle(loglossgen, 5)
	# )

	loss(x, y) = binarycrossentropy(model.discriminator(x), y)
	Θ = params(model.generator)
	
	grads = gradient(() -> loss(model.generator(noise(batchsize)), target), Θ)
	for p in Θ
		Flux.Optimise.update!(opt, p, grads[p])
	end

	@info "genloss" loss(model.generator(noise(batchsize)), target)
end

function train_gan(model, tensors, epochs, batchsize=64)
	println("training")
	dataloader = Flux.Data.DataLoader(tensors, batchsize=batchsize, shuffle=true)
	optdiscr = ADAM(0.0002)
	optgen = ADAM(0.0002)
    @showprogress for epoch in 1:epochs
    	for batch in dataloader
    		train_discriminator(model, optdiscr, batch, batchsize)
    		train_generator(model, optgen, batchsize)
    	end
    end
end

function plot_syntethic(model, m=9)
	images = model.generator(noise(m))
	images = Gray.(images)
	images = reshape(images, (28, 28, m))
	images = [images[:,:,i] for i in 1:m]
	imgplots = plot.(images[1:m])
	plot(imgplots...)
end

end