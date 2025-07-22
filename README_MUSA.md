# Development guid for MUSA user

Note that there are several keys to converting this project to a project that can use MUSA GPU.  

0. This code is for the SDK version 4.0.1 for mtt s80 and that torch_musa Release v2.0.0.  

1. On scripts (like `main_inference.py`, `train.py`) where you are running, add `import torch_musa`.  

2. Delete/comment those codes involving multi-gpu (dont know if really needed since I just have one gpu).  

3. Dont use half-precision training (or also inference?), I have trouble there.  

4. If you encounter problem somewhere like on MUDNN-related problem, since I cannot solve it, I just put that place to CPU, and after the computation finish, put back to MUSA gpu to ensure correct data type.  

5. If you find trouble with the inference audio from the fientunned model, maybe it is your training config problem? (like size of datasets, epoch, learning rates, ...)  