import os
from PIL import Image

output_dir = "/home/qinyuan/LayerDrop/models/may19/fisher_pca128"

checkpoints = os.listdir(output_dir)
checkpoints = list(filter(lambda x: x.endswith("-steps"), checkpoints))
checkpoints = sorted(checkpoints, key=lambda x: int(x[:-6]))

frames = [Image.open(os.path.join(output_dir, checkpoint, "route.png")) for checkpoint in checkpoints]
frame_one = frames[0]
frame_one.save(os.path.join(output_dir, "routes-dynamics.gif"), format="GIF", append_images=frames,
            save_all=True, duration=200, loop=0)