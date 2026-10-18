import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class Renderer:
    def __init__(self, count=4):
        if not 1 <= count <= 16: raise ValueError("Render 1 to 16 cars")
        self.count = count
        cols = min(count, 4)
        rows = (count + cols - 1) // cols
        self.fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False, dpi=80)
        self.axes = axes.flatten()

    def frame(self, env, label=""):
        # CPU transfers are intentionally confined to visualization.
        pos = env.pos[:self.count].cpu().numpy()
        heading = env.heading[:self.count].cpu().numpy()
        obstacles = env.obstacles[:self.count].cpu().numpy()
        rays = env.rays()[:self.count].cpu().numpy()
        offsets = env.offsets.cpu().numpy()
        for i, ax in enumerate(self.axes):
            ax.clear()
            if i >= self.count: ax.set_visible(False); continue
            x, y = pos[i]
            ax.set(xlim=(x-4, x+24), ylim=(-env.cfg["width"]/2-1, env.cfg["width"]/2+1), aspect="equal", title=f"{label} · car {i+1}")
            for wall in (-env.cfg["width"]/2, env.cfg["width"]/2): ax.axhline(wall, color="#64748b")
            ax.axvline(env.cfg["length"], color="green")
            for ox, oy, radius in obstacles[i]: ax.add_patch(Circle((ox, oy), radius, color="#ef4444"))
            for angle, length in zip(heading[i]+offsets, rays[i]):
                ax.plot([x,x+np.cos(angle)*length], [y,y+np.sin(angle)*length], color="#38bdf8", alpha=.3, linewidth=.7)
            ax.add_patch(Circle((x,y), env.cfg["car_radius"], color="#2563eb"))
            ax.arrow(x,y,np.cos(heading[i])*1.5,np.sin(heading[i])*1.5,width=.1,color="#172554")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self): plt.close(self.fig)
