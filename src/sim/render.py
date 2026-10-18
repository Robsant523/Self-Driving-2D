"""Top-down car and road renderer. CPU transfers live only in this module."""
from collections import deque
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, FancyBboxPatch
from matplotlib.transforms import Affine2D


CAR_COLORS = ('#19bde3', '#f59e5b', '#a78bfa', '#77d49b', '#f38ab1', '#f5d76e', '#82a9ff', '#e3ae80',
              '#82e1dc', '#cc9bf3', '#ffad91', '#b6d879', '#e6a6d5', '#8bc7f0', '#d4bd91', '#e0e5ef')


class CrashHistory:
    """Read-only visual history; batch device-to-host transfer happens at render time."""
    def __init__(self, count, limit=200):
        self.count = count
        self.cars = [deque(maxlen=limit) for _ in range(count)]
        self.pending = deque(maxlen=4096)

    def capture(self, info):
        pose = torch.cat((info['final_pos'][:self.count],
                          info['final_heading'][:self.count, None],
                          info['final_steering'][:self.count, None]), -1)
        self.pending.append((info['collision'][:self.count].clone(), pose))

    def drain(self):
        if not self.pending:
            return
        hits = torch.stack([entry[0] for entry in self.pending]).cpu().numpy()
        poses = torch.stack([entry[1] for entry in self.pending]).cpu().numpy()
        self.pending.clear()
        for lane in range(self.count):
            for pose in poses[:, lane][hits[:, lane]]:
                self.cars[lane].append(tuple(pose))


class Renderer:
    def __init__(self, count=4, show_rays=False, camera="follow", history=None, layout="grid"):
        if not 1 <= count <= 16:
            raise ValueError('Render 1 to 16 cars')
        if camera not in ("follow", "overview"):
            raise ValueError("Camera must be follow or overview")
        if layout not in ("grid", "overlay"):
            raise ValueError("Layout must be grid or overlay")
        self.layout = layout
        self.camera = camera
        self.count = count
        self.show_rays = show_rays
        self.history = history if history is not None else CrashHistory(count)
        panels = 1 if layout == "overlay" else count
        cols = min(panels, 2 if panels <= 4 else 4)
        rows = (panels + cols - 1) // cols
        self.fig, axes = plt.subplots(rows, cols, figsize=(12, 4.8 if camera == 'overview' else 7.04) if panels == 1 else (6 * cols, 4 * rows), squeeze=False, dpi=100)
        self.fig.patch.set_facecolor('#102326')
        self.axes = axes.flatten()
        self.fig.subplots_adjust(left=.02, right=.98, bottom=.03, top=.91, wspace=.06, hspace=.16)
        self.title = self.fig.suptitle('', color='white', fontsize=17, fontweight='bold')
        self.map_ax = self.fig.add_axes([.73, .76, .23, .13]) if panels == 1 and camera == "follow" else None

    @staticmethod
    def car(ax, x, y, heading, steering, scale=1., crashed=False, color=CAR_COLORS[0]):
        first_patch = len(ax.patches)
        transform = Affine2D().scale(scale).rotate(heading).translate(x, y) + ax.transData
        # Wheels, front wheel steering, body, glass, lights: x is the forward axis.
        for wheel_x in (-.82, .82):
            for wheel_y in (-.64, .64):
                local = Affine2D().rotate_around(wheel_x, wheel_y, steering if wheel_x > 0 else 0)
                ax.add_patch(Rectangle((wheel_x-.29, wheel_y-.12), .58, .24,
                                      color='#111827', transform=local + transform, zorder=7))
        ax.add_patch(FancyBboxPatch((-1.3, -.57), 2.6, 1.14,
                                   boxstyle='round,pad=0.01,rounding_size=0.22',
                                   facecolor='#b96154' if crashed else color, edgecolor='#e5a99b' if crashed else '#d1f6ff', linewidth=1.2,
                                   transform=transform, zorder=8))
        ax.add_patch(Polygon([[-.65,-.43],[.42,-.43],[.74,-.32],[.74,.32],[.42,.43],[-.65,.43]],
                             facecolor='#132b42', edgecolor='#73d8ee', linewidth=.8,
                             transform=transform, zorder=9))
        ax.add_patch(Rectangle((-.48,-.4), .65,.8, facecolor='#785a50' if crashed else color, transform=transform,zorder=10))
        for yy in (-.44,.27):
            ax.add_patch(Rectangle((1.13,yy),.12,.17,color='#fff5bd',transform=transform,zorder=10))
            ax.add_patch(Rectangle((-1.25,yy),.10,.17,color='#ff5268',transform=transform,zorder=10))

        if crashed:
            for patch in list(ax.patches)[first_patch:]:
                patch.set_alpha(.38)
                patch.set_zorder(5.5)

    def capture(self, info):
        self.history.capture(info)

    def frame(self, env, label=''):
        if self.layout == 'overlay' and env.cfg['obstacles'] and 'fixed_obstacles' not in env.cfg:
            raise ValueError('Overlay requires shared fixed obstacles or no obstacles; use grid for random worlds')
        self.history.drain()
        pos = env.pos[:self.count].cpu().numpy()
        heading = env.heading[:self.count].cpu().numpy()
        speed = env.speed[:self.count].cpu().numpy()
        steering = env.last_action[:self.count, 0].cpu().numpy() * env.cfg['max_steer']
        obstacles = env.obstacles[:self.count].cpu().numpy()
        if env.road is not None:
            lower = env.road.lower.cpu().numpy()
            upper = env.road.upper.cpu().numpy()
            center = (lower + upper) / 2
        else:
            center = np.array([[-30.,0.],[env.cfg['length']+30.,0.]])
            lower = center + [0.,-env.cfg['width']/2]
            upper = center + [0.,env.cfg['width']/2]
        rays = env.rays()[:self.count].cpu().numpy() if self.show_rays else None
        offsets = env.offsets.cpu().numpy()
        course = env.cfg.get("road", {}).get("name", "Bendy road" if env.road else "Driving lab")
        self.title.set_text(f"{course.upper()}  /  {label.upper()}")
        for panel, ax in enumerate(self.axes):
            ax.clear()
            if self.layout == 'grid' and panel >= self.count:
                ax.set_visible(False)
                continue
            i = 0 if self.layout == 'overlay' else panel
            lanes = range(self.count) if self.layout == 'overlay' else (i,)
            x, y = pos[i]
            ax.set_facecolor('#537b50')
            if self.camera == 'overview':
                # Fit only the playable track, not the invisible sensor extensions.
                inside = (center[:, 0] > 0) & (center[:, 0] < env.cfg['length'])
                ys = np.concatenate(([np.interp(0, center[:,0],center[:,1])], center[inside,1],
                                     [np.interp(env.cfg['length'],center[:,0],center[:,1])]))
                pad = max(3., env.cfg['width'] * .35)
                ax.set(xlim=(-pad, env.cfg['length']+pad),
                       ylim=(ys.min()-env.cfg['width']/2-pad,ys.max()+env.cfg['width']/2+pad), aspect='equal')
            else:
                ax.set(xlim=(x-12, x+27), ylim=(y-13, y+13), aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            # Shared geometry means painted boundaries match the physics edges.
            ax.add_patch(Polygon(np.concatenate((lower, upper[::-1])), color='#323b43', zorder=2))
            for edge in (lower, upper):
                ax.plot(edge[:,0],edge[:,1],color='#a7a49b',linewidth=6,zorder=1)
                ax.plot(edge[:,0],edge[:,1],color='#edf2e9',linewidth=1.8,zorder=3)
            ax.plot(center[:,0],center[:,1],color='#f7d978',linewidth=1.8,dashes=(7,7),zorder=3)
            finish_y = np.interp(env.cfg['length'],center[:,0],center[:,1])
            for j in range(12):
                ax.add_patch(Rectangle((env.cfg['length'],finish_y-env.cfg['width']/2+j*env.cfg['width']/12),
                                      .5,env.cfg['width']/12,color='white' if j%2 else '#16212b',zorder=4))
            for ox, oy, radius in obstacles[i]:
                ax.add_patch(Circle((ox,oy),radius,color='#e66f45',ec='#ffcfa5',zorder=5))
            for lane in lanes:
                for cx, cy, ch, cs in self.history.cars[lane]:
                    self.car(ax,cx,cy,ch,cs,min(1.,env.cfg['car_radius']/1.5),crashed=True)
            if self.show_rays:
                for angle, distance in zip(heading[i]+offsets,rays[i]):
                    ax.annotate('',xy=(x+np.cos(angle)*distance,y+np.sin(angle)*distance),xytext=(x,y),
                                arrowprops=dict(arrowstyle='->',color='#7de5ef',alpha=.5,lw=.8,mutation_scale=6),zorder=6)
                if env.road is not None:
                    for lookahead in (5.,12.,20.):
                        target_x = x + lookahead
                        target_y = np.interp(target_x,center[:,0],center[:,1])
                        ax.annotate('',xy=(target_x,target_y),xytext=(x,y),
                                    arrowprops=dict(arrowstyle='->',color='#ffce70',alpha=.85,lw=1.1,linestyle='--',mutation_scale=8),zorder=6)
                        ax.plot(target_x,target_y,'o',color='#ffce70',markersize=2.5,zorder=6)
            # The displayed car fits inside the configured circular collision footprint.
            for lane in lanes:
                px,py = pos[lane]
                self.car(ax,px,py,heading[lane],steering[lane],min(1.,env.cfg['car_radius']/1.5),color=CAR_COLORS[lane])
                if self.count > 1:
                    ax.annotate(str(lane+1),xy=(px,py),xytext=(0,9+9*(lane%2)),textcoords='offset points',
                                ha='center',color=CAR_COLORS[lane],fontsize=7,fontweight='bold',zorder=15,
                                bbox=dict(boxstyle='round,pad=.15',fc='#13262b',ec='none',alpha=.85))
            hud = f"{self.count} TRIAL CARS   |   CAR 1: {speed[0]*3.6:04.1f} km/h" if self.layout == 'overlay' else f"CAR {i+1:02d}   {speed[i]*3.6:04.1f} km/h"
            ax.text(.025,.96,hud,transform=ax.transAxes,
                    color='white',fontsize=11,va='top',bbox=dict(facecolor='#13262b',edgecolor='none',alpha=.9,pad=7),zorder=20)
            ax.text(.025,.035,f"COURSE   {max(0,min(100,x/env.cfg['length']*100)):05.1f}%",transform=ax.transAxes,
                    color='white',fontsize=10,bbox=dict(facecolor='#13262b',edgecolor='none',alpha=.9,pad=5),zorder=20)
        if self.map_ax is not None:
            ax=self.map_ax; ax.clear(); ax.set_facecolor('#13262b')
            ax.plot(center[:,0],center[:,1],color='#9aaea8',linewidth=2)
            ax.scatter(pos[:,0],pos[:,1],s=20,c=CAR_COLORS[:self.count],zorder=3)
            ax.set_xlim(0,env.cfg['length']); ax.set_ylim(center[:,1].min()-5,center[:,1].max()+5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title('COURSE MAP',color='#e5f2e7',fontsize=8,pad=3)
            for spine in ax.spines.values(): spine.set_color('#729383')
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[...,:3].copy()

    def close(self):
        plt.close(self.fig)
