from pathlib import Path

path = Path('/Users/sam/Downloads/python/OpenCV-Test/data/faces')    
print(list(f for f in Path(path).iterdir() if f.is_file()))

