# audio-embed-experiments

For people coming across this, I sincerely apologize for the mess. This project is very much still ongoing, and many files and functions are purely exploratory. To give a rough idea, I'm trying to see how Oobleck autoencoders ([Evans et al. 2024](https://doi.org/10.48550/arXiv.2407.14358)), and eventually other models, use convolution to perform the Fourier transform on audio waveforms. I'll list some of my current findings as well as research directions planned for the near future. 

## Findings

### 1: Frequency-level representations in VAE latent
An sparse autoencoder (SAE) was trained on the VAE latent. A linear probe trained on SAE features partially recovered frequency-level representations, but could not reconstruct the full original spectrogram. 

See [experiments/sae/sae.py](experiments/sae/sae.py) and [experiments/sae/probe.py](experiments/sae/probe.py). 

### 2: Some convolutional layers may be merged
Convolutional layers in the Oobleck autoencoder are interleaved with [snake beta activation functions](https://www.desmos.com/calculator/bdzfglog7l) ([Ziyin et al. 2020](https://doi.org/10.48550/arXiv.2006.08195))([Lee et al. 2022](https://doi.org/10.48550/arXiv.2206.04658)). By either removing or linearizing them, we can use convolution to combine the convolutional kernels of both layers into a higher-resolution kernel more suitable for analysis. I showed that for some snake activations, ablation preserves the ability of the VAE to reconstruct most audio samples. 

See [experiments/kernel/results/ablation](experiments/kernel/results/ablation) for the results of ablating the first snake layer (**IMPORTANT: `2_ablated_recon.wav` is very loud and abrasive, listen at low volume**). 

## To-do
- [ ] Snake linearization (via linear regression on data) rather than ablation
- [ ] Analyze kernels for similarity to Gabor filters or possible edge detectors
- [ ] Probe earlier layers of the VAE (if conv kernels are Gabor filters, these representations should be monotonically probeable)
- [ ] Try other models (e.g. Wav2vec 2.0 ([Baevski et al. 2020](https://doi.org/10.48550/arXiv.2006.11477)), which uses GELU ([Hendrycks & Gimpel 2016](https://doi.org/10.48550/arXiv.1606.08415)) rather than snake in its convolutional encoder)

## Dependency Quirks
- [diffusers](https://github.com/huggingface/diffusers) is imported via [a fix](https://github.com/huggingface/diffusers/pull/13754/changes/51906b8b3dd43bafa8a3a45f2b62b1af6812b91b) by zxuhan.
- [torchsde](https://github.com/google-research/torchsde) here is [a fork](https://github.com/realBarry123/torchsde) created by me with Claude to fix incompatability with Python 3.13. 