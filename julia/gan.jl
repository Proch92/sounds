__precompile__()
module Gan
export train_gan, plot_syntethic, newmodel, savemodel, loadmodel

using MLDatasets
using Flux
using Flux: Losses.logitbinarycrossentropy, throttle
using Zygote
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

function lossdisc(real, fake)
    real_loss = mean(logitbinarycrossentropy.(real, 1f0))
    fake_loss = mean(logitbinarycrossentropy.(fake, 0f0))
    return real_loss + fake_loss
end

lossgen(fake) = mean(logitbinarycrossentropy.(model.discriminator(fake), 1f0))

function train_discriminator(model, opt, real, batchsize)
	fake = model.generator(noise(batchsize))

	Θ = params(model.discriminator)

	loss, back = Flux.pullback(Θ) do
        lossdisc(model.discriminator(real), model.discriminator(fake))
    end
	
	Flux.Optimise.update!(opt, Θ, back(1f0))

	@info "discloss" loss
end

function train_generator(model, opt, batchsize)
	Θ = params(model.generator)
	
	loss, back = Flux.pullback(Θ) do
        lossgen(model.discriminator(model.generator(noise(batchsize))))
    end
	
	Flux.Optimise.update!(opt, Θ, back(1f0))

	@info "genloss" loss
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