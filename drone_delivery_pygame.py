import os
import sys
import time
import math
import numpy as np

# Suppress Pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pygame.gfxdraw

# Import Algorithms
from welzl_mec import minimum_enclosing_circle as welzl_algo
from skyum import skyum_algo
from approx import meb_approximation as approx_algo
from HighDWelzl import minimum_enclosing_ball as msw_algo
from megiddo import solve_mec_megiddo as megiddo_algo

# Increase recursion limit for Welzl & MSW
sys.setrecursionlimit(10_000_000)

# Colors - Cyber/Drone Theme
BG_COLOR = (15, 17, 23)
GRID_COLOR = (30, 35, 45)
TEXT_COLOR = (230, 230, 240)
ACCENT_COLOR = (124, 106, 247)
POINT_COLOR = (38, 198, 218)     # Teal / Cyan
CENTER_COLOR = (240, 98, 146)    # Pink
CIRCLE_COLOR = (240, 98, 146, 60) # Transparent Pink
HUD_BG = (20, 24, 33, 200)

ALGORITHMS = [
    ("Welzl (Exact)", "welzl"),
    ("Skyum (Exact)", "skyum"),
    ("MSW (Exact N-D)", "msw"),
    ("Approx (ε=0.05)", "approx"),
    ("Megiddo (Exact)", "megiddo")
]

class DroneDeliveryApp:
    def __init__(self):
        pygame.init()
        self.width = 900
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("City Drone Food Delivery - Minimum Enclosing Circle")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 18, bold=True)
        self.title_font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.points = []
        self.algo_idx = 0 # Default to Welzl (Key 1)
        self.center = (0.0, 0.0)
        self.radius = 0.0
        self.calc_time = 0.0
        
        self.running = True
        self.pulse = 0.0

        # JIT Warmup
        self.warmup_jit()

    def warmup_jit(self):
        # Prevent huge lag spikes on first calculation
        dummy = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        d_np = np.array(dummy)
        welzl_algo(dummy)
        approx_algo(d_np, 0.1)
        skyum_algo(dummy)
        msw_algo(dummy, 2)
        megiddo_algo(dummy)

    def compute_mec(self):
        n = len(self.points)
        if n == 0:
            self.center = (0.0, 0.0)
            self.radius = 0.0
            self.calc_time = 0.0
            return
            
        algo_name = ALGORITHMS[self.algo_idx][1]
        t0 = time.perf_counter()
        
        try:
            if algo_name == "welzl":
                self.center, self.radius = welzl_algo(self.points)
            elif algo_name == "skyum":
                self.center, self.radius = skyum_algo(self.points)
            elif algo_name == "msw":
                c, r = msw_algo(self.points, 2)
                self.center, self.radius = tuple(c), r
            elif algo_name == "approx":
                pts_np = np.array(self.points).reshape(-1, 2)
                c, r, _ = approx_algo(pts_np, 0.05)
                self.center, self.radius = tuple(c), r
            elif algo_name == "megiddo":
                c_x, c_y, r = megiddo_algo(self.points)
                self.center = (c_x, c_y)
                self.radius = r
        except Exception as e:
            print(f"Algorithm Error: {e}")
            
        t1 = time.perf_counter()
        self.calc_time = (t1 - t0) * 1000.0 # ms

    def add_point(self, pos):
        self.points.append(pos)
        self.compute_mec()

    def add_cluster(self, pos, count=50, spread=40):
        # Normal distribution around cursor
        pts = np.random.randn(count, 2) * spread + np.array(pos)
        for p in pts:
            # Keep inside screen
            x = max(10, min(self.width-10, p[0]))
            y = max(10, min(self.height-10, p[1]))
            self.points.append((x, y))
        self.compute_mec()

    def draw_grid(self):
        for x in range(0, self.width, 50):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, 50):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.width, y), 1)

    def draw_hud(self):
        # Draw HUD Panel
        hud_surface = pygame.Surface((350, 220), pygame.SRCALPHA)
        pygame.draw.rect(hud_surface, HUD_BG, hud_surface.get_rect(), border_radius=10)
        self.screen.blit(hud_surface, (20, 20))
        
        # Draw Texts
        texts = [
            self.title_font.render("Drone Base Station Planner", True, ACCENT_COLOR),
            self.font.render(f"Algorithm: {ALGORITHMS[self.algo_idx][0]}", True, TEXT_COLOR),
            self.font.render(f"Delivery Points: {len(self.points)}", True, POINT_COLOR),
            self.font.render(f"Calc Time: {self.calc_time:.3f} ms", True, TEXT_COLOR),
            self.font.render(f"Radius (Battery): {self.radius:.1f} units", True, CENTER_COLOR),
            self.font.render("", True, TEXT_COLOR),
            self.font.render("CONTROLS:", True, ACCENT_COLOR),
            self.font.render("Left Click : Add Target", True, TEXT_COLOR),
            self.font.render("Right Click: Add Cluster", True, TEXT_COLOR),
            self.font.render("Keys 1-5   : Switch Algorithm", True, TEXT_COLOR),
            self.font.render("C : Clear | ESC : Quit", True, TEXT_COLOR)
        ]
        
        y_offset = 30
        for t in texts:
            self.screen.blit(t, (35, y_offset))
            y_offset += 25 if t.get_height() > 20 else 20

    def draw_drone_base(self, cx, cy, r):
        if r <= 0: return
        
        # Transparent radar fill
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.circle(surf, CIRCLE_COLOR, (int(cx), int(cy)), int(r))
        self.screen.blit(surf, (0, 0))
        
        # Anti-aliased circle outline
        pygame.gfxdraw.aacircle(self.screen, int(cx), int(cy), int(r), CENTER_COLOR)
        
        # Pulsing center icon (crosshair)
        pulse_r = 6 + math.sin(self.pulse) * 2
        pygame.draw.circle(self.screen, CENTER_COLOR, (int(cx), int(cy)), int(pulse_r))
        pygame.draw.line(self.screen, CENTER_COLOR, (cx - 15, cy), (cx + 15, cy), 2)
        pygame.draw.line(self.screen, CENTER_COLOR, (cx, cy - 15), (cx, cy + 15), 2)
        
        # Draw radius line sweeping like a radar
        sweep_x = cx + r * math.cos(self.pulse * 2)
        sweep_y = cy + r * math.sin(self.pulse * 2)
        pygame.draw.line(self.screen, (240, 98, 146, 150), (cx, cy), (sweep_x, sweep_y), 1)

    def draw(self):
        self.screen.fill(BG_COLOR)
        self.draw_grid()
        
        # Draw points
        for p in self.points:
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 4, POINT_COLOR)
            pygame.gfxdraw.aacircle(self.screen, int(p[0]), int(p[1]), 4, POINT_COLOR)
            
        # Draw center and circle
        if len(self.points) > 0:
            self.draw_drone_base(self.center[0], self.center[1], self.radius)
            
        self.draw_hud()
        pygame.display.flip()

    def run(self):
        while self.running:
            self.pulse += 0.05
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_c:
                        self.points = []
                        self.compute_mec()
                    elif pygame.K_1 <= event.key <= pygame.K_5:
                        idx = event.key - pygame.K_1
                        if idx != self.algo_idx:
                            self.algo_idx = idx
                            self.compute_mec()
                    elif pygame.K_KP1 <= event.key <= pygame.K_KP5:
                        idx = event.key - pygame.K_KP1
                        if idx != self.algo_idx:
                            self.algo_idx = idx
                            self.compute_mec()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left click
                        self.add_point(event.pos)
                    elif event.button == 3: # Right click
                        self.add_cluster(event.pos, count=50, spread=40)
            
            self.draw()
            self.clock.tick(60)
            
        pygame.quit()

if __name__ == "__main__":
    app = DroneDeliveryApp()
    app.run()
