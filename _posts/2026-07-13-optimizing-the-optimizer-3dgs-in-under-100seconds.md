# Optimizing 3DGS to train in seconds

## A great local optima

I've been working on 3DGS for almost 3 years at this point, while at EA I spent a lot of time optimizing what we'd call the forward (fwd) part of the pipeline, er the rasterization of the primitive. If you want to ship splats on consumer hardware (for whatever reason), that's what you should, at least initially, care about.<br>
Last year though, I started prototyping some stuff in GIGI (great framework to experiment with GFX programming techniques at the speed of light, thank you Alan, its an amazing piece of software). I had heard about Slang, so I decided to prototype a 2d gaussian trainer in it.<br>
Coming from the GFX world, I was slightly averse to the pytorch environment, and having parts of the optimization process "hidden" behind the its machinery was slowing me down in understanding the whole things. I went through the TinyDiffRast tutorial course to get a better grip on how differentiable rendering (rasterization in particular) works, and walked my way from there. Also, once you'll get to the RasterizeThenSplat section of it, the reason why 3DGS just makes sense, should also click for you.<br>
Long story short, I've been reading many papers on 3DGS, and while focusing on optimizing the hell out of the pure rasterization part of it was fun, it left me wanting to reinvent the weel, which degenerated in me thinking I could write the fastest available gradient-based 3DGS trainer. <br>
While in the last year or so a lot of research has been published on feed-forward, RNN or transformer-based models, my goal has slowly shifted towards proving to myself that while the Bittersweet lesson surely is a reality, in our present day, with a certain amount of effort, we can still match or surpass large pre-trained general networks by leveraging pure problem parameterization and optimization. <br>
Why do I need a universal function approximator to circumvent my lack of willingness to optimize? Is compute-cost infinitely available? No, it's actually rather expensive.<br>
Rant aside, all in all the point I want to make is that 3DGS as a NVS method has landed in great local optima, and by improving on the seminal's work oversights or limitations, we can move our optimization problem from taking 30 minutes, down to a few seconds.<br>

## The optimization landscape

Read a follow-up paper or two on 3DGS and you'll likely end up agreeing with my takes on what the main bottlenecks clearly are. Not ranked by importance and impact:

### Projection and binning
3DGS conservatively estimates the gaussian's projections bounds at 3sigma (should be 3.33 sigma in reality but whatever). This, first of all does, not account for the learned opacity term -> if we want our splat's border of influence to be capped at 1f/255f alpha, then this should take into account the current splat's learned opacity, not only the gaussians densitiy fallof value.<br>
We can, relatively simply, compute which tiles the splat's region of influence overlaps our tiles via SnugBox and then AccuTile. Both help reduce how many splats get binned in each one of our tiles, reducing pressure both on the radix sort pass, fwd and bwd pass. Remember: it's a tile based rasterizer. Divide-et-impera means we need to identify which screen regions each splat influences. If each pixel had to take into account each splat visible in the camera's frustum it would be disastrous to say the least. [AMD's DX11 GPU particles](https://gpuopen.com/learn/gpuparticles11-directx-11-sdk-sample/) sample is a good reference for what a tile-based rasterizer is and why its important to allow efficient rasterization of particle-like primitives. You could always go the quad-based instanced draw call approach right? Sure, and beating the HW rasterizer is not an easy task, but from my experience the VS-PS crossbar section is not made to deal with frames with 100x overdraw. TL; DR: the rasterizer is fast, but overdraw and a huge amount of PS warp scheduling is a performance killer + you cant easily get gradients back from a HW rasterizer. Intel has published a nice paper on [Transforming a Non-Differentiable Rasterizer into a Differentiable One with Stochastic Gradient Estimation](https://arxiv.org/html/2604.28016v1). It's a cool paper, give it a read. But while Finite Differences (er stochastic finite differences in this case). I like the idea, but that gradient is noisier and convergence slower, and given that we now have Slang, a shading language were autodifferention is a first-class citizen component + the fact that the HW rasterizer does not implicitly imply a performance win for a primitive like 3DGS where overdraw is innate to its nature (well, we can mitigate this with some smarter approaches to deal with occlusion). Also, you'll find out that if you want your backward kernels to run fast, the autodiff system cannot be offloaded much of the gradient computations, but it still allows for some mental offload in certain cases. 

### Densification
The biggest flaw in the seminal 2023 paper was the Adaptive Density Control (ADC) method. This is not even an opinion at this point, as the amount of papers that try to specifically improve, tweak, or totally revamp the primitive's population expansion logic during training is likely the biggest corpus of work. <br>
If you're familiar with 3DGS, you must have heard about the concept of **densification**. As training starts, our scene is represented by a sparse set of gaussians, initialized from what, traditionally, something like SfM (or more recently MaSt3r, DuSt3r, etc) give us.<br>
To put things into persective in our **discovery** path (from a set of images to a full 3D reconstruction), I see the input sparse point cloud from SfM + the camera's intrinsic and extrinsic parameters as representing a very likely initial guess of what the scene looks like, meaning using this as a starting point to attempt reconstructing our 3D scene just makes sense, it jump-started our optimization problem into a very good local minima (am I abusing this term? Maybe.)<br>
If you look into what ADC is and how it works, you'll realize it sounds more like a magical researchers intuition that somehow was able to map a signal from the optimization process (the screen-space 2d gradient) to a proxy for **hey we need more primitives here if we want to reconstruct stuff well**. If the splat's grad is above this threshold and within below this size thresholds, clone it, if its larger, split it. The former indicating under-reconstruction while the latter under-recontstruction. It makes sense, but it's a proxy, and the signal is intepreted via pure heuristics as a basis to decide how to increase the splat population in our scene/reconstrution. And because it's an indirect proxy, it is very sensitive to hyperparameters. I cannot even imagine how long it took the INRIA peeps to figure out 0.0002 was indeed the right threshold value that generalized well across scenes.<br>
I'm rambling again.<br>
All this to say that, after 3 years, we've figured some better ways to signal that our scene requires an increase in its primitive population in a certain spot.<br>
First of all: who says that the expansion factor should be a fixed 2? When we split or clone, for each selected splat for densification, most methods have just blindly followed the +1 expansion factor. Parent transforms in two slightly different splats. Looking at this from a high level, I can see how this is a safe bet: we are already using gradient as a proxy for densification, but if we were right, our exploratory path is not tremendously bad, the optimizer will easily find a way to adjust the two children's parameters if the duplication was not really that necessary. I'd compare this as tip-tooeing into the optimization landscape. If only our proxy we're that far away from what the real groundtruth should be, we could make a more informed and decisive guess on both how many and where our children should be. Let me introduce you to [Faster 3D Gaussian Splatting Convergence via Structure-Aware Densification](https://arxiv.org/html/2604.28016v1). This paper does just that: lets anaylize our training images in a more structured way (they use structure tensors so I guess the analogy makes sense) and identify which splats should be densified and by which factor.<br>
Avoiding the tip-tooeing approach to densification shows how many steps in the optimization process were dedicated to adjusting via gradient computation and parameter update a bad guess: the training converges in 3000 steps intead of 30000.


### Pure gpu optimization
If you try implementing the backward pass naively and benchmark a training step, you'll quickly realize you are most likely 80/90% bwd pass bound. I you run a gpu trace, you'll see how the main bottleneck is atomic contention -> a bunch of long scoreboards. This is due to the incredibly high atomic contention you're subjecting your poor gpu too. In a naive implementation, each work group maps to a tile, and each thread in a WG maps to a pixel. Each thread then accesses its splats via an index buffer given the tile id.<br>
Imagine a tile has 10 splats binned to it, meaning during the projection phase we identified these 10 to be the splats that influence any pixel of it. It makes sense to have each thread in your thread group map to a pixel of the tile, in parallel compute the per-pixel gradients for each parameter given the per-pixel loss and then atomically add each gradient value into the gradient buffers (where each idx **n** within it maps to splat **n**). So we're saying that we have 256 (16x16) threads atomically hammering 14+ buffers each into the same slot **n**. We can agree that while mentally simpler to parallely map the problem this way, it's not great.<br>
I'd be happy if each tile would actually contain 10 splats, but its often unfortunately 1-2 orders of magnitude higher. Some tiles from certain camera views can have binned in them 10000k splats, and even after filtering out splats that do not contribute to the final pixel as Transmittance fell below a certain threshold way before them, we may need to blend 1-3k splats in order to reach it (this also raises a question for: damn do we actually need to blend 3000 splats for a single pixel in order to a) get the correct final color b) correctly let the gradient and optimization flow c) look around, 90% of what you see is mostly opaque anyway...). Just hinting at where I think the optima for rendering splats is, even though we are here talking about a primary visibility problem: What's better? Clustered or tiled shading? I'd go with the former :) <br>
Anyhow, we agree that having our loop be:

```
(within each tile)
for each splat: <---- each thread has a loop over each splat within this tile
    for each pixel: <---- each thread gets assigned to a px
        compute or load ppx loss
        compute chain rule and gradient for each parameter
        atomicAdd(buff_grad_forWhateverParamThis1, gradForWhateverParamThis_asInt)
        atomicAdd(buff_grad_forWhateverParamThis2, gradForWhateverParamThis_asInt)
        ...
        atomicAdd(buff_grad_forWhateverParamThisN, gradForWhateverParamThis_asInt)

```

is not great. You could think of accumulating each gradient in groupshared memory first, but the atomic contention is still there. You can try, it wont make a difference.
What if we flipped the problem though?
What if we parallelized over each splat within each tile? That's interesting!
So our loop would become:

```
(within each tile)
for each pixel: <---- each thread has a loop over each pixel within this tile
    int totalGradForWhateverParamThis1 = 0
    int totalGradForWhateverParamThis2 = 0
    ...
    int totalGradForWhateverParamThisN = 0

    for each splat: <---- each thread gets assigned to a splat
        compute or load ppx loss
        compute chain rule and gradient for each parameter
        totalGradForWhateverParamThis1 += gradForWhateverParamThis1_asInt
        totalGradForWhateverParamThis2 += gradForWhateverParamThis2_asInt
        ...
        totalGradForWhateverParamThisN += gradForWhateverParamThisM_asInt

    atomicAdd(buff_grad_forWhateverParamThis1, totalGradForWhateverParamThis1)
    atomicAdd(buff_grad_forWhateverParamThis2, totalGradForWhateverParamThis2)
    ...
    atomicAdd(buff_grad_forWhateverParamThisN, totalGradForWhateverParamThisN)

```

Hey we just traded ALU and some register pressure to save a ton of atomics!<br>
Another trick -> in order to compute the transmittance at each splat, you need the transmittance from each splat in front of it.
Well, assuming our wave size is 32 (64 on AMD hw), during our forward pass we can store some "checkpoints". Meaning that every 32nd blended splat, we store the intermediate color and opacity (1-T). By doing this, we can create a set of **buckets**. Why? So whenever we need the transmittance value for a given splat, we can just compute it via a fast wave-intrinsic operation: waveprefixproduct(T) :O
We keep everything in registers, have a single atomic reduction per wave and bam: we reduced (per tile though, because splats can end up in multiple tiles) the atomic contention to 0. We saved thousands of idle cycles in favour of more ALU work, which GPUs are pretty good at now :D

### Sorting
I ported the excellent gpu radix sort HLSL implementation by b0nes to slang, and sorting has always stayed well below the 1.5ms for me. Given that on larger scenes with 5-7mln splats the fwd pass (binning, prefix scans, fwd rasterize) can account to 7/8ms and the bwd pass can take up to 13ms, over-optimizing the sorting part is not really going to yield you much of win. Hell, my non-separable-and-definitely-not-too-well-optimizined SSIM pass is taking more than the radix sort at the moment... so I dont really see a great appeal for stochastic transparency approaches at the moment, given that they usually yield noisier gradient optimization and slower convergence, it does not really help me reach my goal (Ideally I'd like to reach 26db in 100 seconds on a 16gb 4080 card, but I dont shy away from the fact that I'm sure this can be somehow achieved in 10 seconds, I know it can...). </br>
Currently it takes me not less than 3-5 minutes though.                    
