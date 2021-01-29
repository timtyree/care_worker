nvcc step.cu -o step.x
./step.x >> out.png
python3 viewmat.py;xdg-open out.png
