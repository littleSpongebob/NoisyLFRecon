# NoisyLFRecon
## the implementation of "Robust Dense Light Field Reconstruction from Sparse Noisy Sampling"
- download the [ckpt](https://drive.google.com/file/d/131C7qNZHQJmM-GGig1l2ZogOn4u8e8Vm/view?usp=sharing) and [data](https://drive.google.com/file/d/13VHFqq8OZq9Bm3A_EebfeaJ9WlXfeKid/view?usp=sharing)
- unzip the ckpt and data in root directory
  ```
    ./NoisyLFRecon
    ├── average_gradients.py
    ├── bilinear_sampler.py
    ├── bilinear_sampler_y.py
    ├── ckpt
    ├── data
    ├── dataloader.py
    ├── model.py
    ├── NoisyLFRecon
    ├── readme.md
    ├── requirements.txt
    ├── TCW_test_set.txt
    └── test_model.py
    ```
- run "pip install -r requirement.txt" to install required package
- run "python3 test_model.py" to test model in "30sence" dataset
