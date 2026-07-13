# Optimizing 3DGS to train in seconds

## Lately I've been obsessing over optimizing my 3DGS Slang-based trainer

I've been working on 3DGS for a while at this point, while at EA I spent a lot of time optimizing what we'd call the forward (fwd) part of the pipeline. If you want to ship splats on consumer hardware, that's what you should, at least initially, care about.</br>
Last year though, I started prototyping some stuff in GIGI (great framework to experiment with GFX programming techniques at the speed of light, thank you Alan, its an amazing piece of software). I had heard about Slang, so I decided to prototype a 2d gaussian trainer in it.</br>
If you're new to differentiable rendering, give a read at TinyDiffRast, its a great tutorial series to get your mind set, and once you'll get to the RasterizeThenSplat section of it, the reason why 3DGS just makes sense should also click for you.</br>
Long story short, I've been reading many papers on 3DGS. We can trace some major attack points if we want to improve the convergence speed of it.

### Improve densification
The biggest flaw in the seminal 2023 paper was surely the ADC method. Purely heuristics based and very sensitive to hyperparameters change. Getting the grad threshold matching the original paper was proabably what took the most amount of time. The struggle to get those hyperparams correct motivated me to re-explore some papers I had read and try reimplement their approaches.</br>
1. From FastGS: my favourite approach is also the simplest. Generate an error map from an L2 screen loss, and base your splitting/cloning/pruning decisions based on this mask.

### Projection and binning
3DGS conservatively estimates the gaussian's projections bounds at 3sigma (should be 3.33 sigma in reality but whatever). This first of all does not account for opacity -> if we want our splat's border of influence to be capped at 1f/255f alpha, then this should take into account the current splat's learned opacity Furthermore, we can more conservatively compute which tiles our region of influence actually overlaps via SnugBox and then AccuTile. Both help reduce how many splats get binned in each one of our tiles, reducing pressure both on the radix sort pass, fwd and bwd pass.

### Pure gpu optimization
If you try implementing the backward pass naively and benchmark a training step, you'll quickly realize you are most likely 80/90% bwd pass bound. This is due to the incredibly high atomic contention you're subjecting your poor gpu too. In a naive implementation each work group maps to a tile, so each thread in a WG maps to a pixel, each thread then accesses the its splats via an index buffer given the tile id. Each thread is computing the gradients for the same splat and via an atomicAdd accumulating to main memory. You can image how bad having 256 (16x16) atomicAdds into the same buffer slot can be. [How many cycles is that stalling for?].

### Sorting
I ported the excellent gpu radix sort HLSL implementation by b0nes to slang, and sorting has always stayed well below the 1.5ms for me. Given that on larger scenes with 5-7mln splats the fwd pass (binning, prefix scans, fwd rasterize) can account to 7/8ms and the bwd pass can take up to 13ms, over-optimizing the sorting part is not really going to yield you much of win. Hell, my non-separable-and-definitely-not-too-well-optimizined SSIM pass is taking more than the radix sort at the moment... so I dont really see a great appeal for stochastic transparency approaches at the moment, given that they usually yield noisier gradient optimization and slower convergence, it does not really help me reach my goal (Ideally I'd like to reach 26db in 100 seconds on a 16gb 4080 card, but I dont shy away from the fact that I'm sure this can be somehow achieved in 10 seconds, I know it can...). </br>
Currently it takes me not less than 3-5 minutes though.                    
