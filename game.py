"""Manual drift game: top-down chase view, camera aligned with velocity.

Controls:
    LEFT / RIGHT   steer (first-order lag to full lock)
    UP / DOWN      throttle / brake-reverse (first-order lag)
    R              restart (new layout on random tracks)
    ESC            quit

Usage:
    python game.py                          # circular track, keyboard
    python game.py --track random           # random track, new layout per run
    python game.py --track random --seed 7  # reproducible first layout
    python game.py --track free             # open sandbox, no off-track/finish
    python game.py --mode grip              # HUD shows grip reward shaping
    python game.py --controller pid         # driver: keyboard, rl, pid, pidref
    python game.py --track free --controller pidref   # dial-a-drift PID
    python game.py --controller rl --mode grip --track random  # RL drives

"""

import argparse

import numpy as np
import pygame

from drift_env import DriftEnv
from controllers import make_controller

SCREEN_W, SCREEN_H = 900, 700
SCALE = 8.0                    # px per meter
CAR_L, CAR_W = 4.0, 1.8        # drawn footprint [m]
CAM_TAU = 0.3                  # camera heading smoothing [s]
CAM_Y = 0.62                   # car drawn below screen center (look-ahead)
GRID_STEP = 10.0               # world-space grid spacing [m]
GRID_RADIUS = 90.0             # grid extent around the camera [m], > screen diag


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def to_screen(pts, cam_pos, cam_ang):
    """World -> screen with cam_ang pointing up. pts: (..., 2)."""
    th = np.pi / 2 - cam_ang
    c, s = np.cos(th), np.sin(th)
    d = np.asarray(pts, dtype=float) - cam_pos
    qx = c * d[..., 0] - s * d[..., 1]
    qy = s * d[..., 0] + c * d[..., 1]
    return np.stack([SCREEN_W / 2 + qx * SCALE, SCREEN_H * CAM_Y - qy * SCALE], axis=-1)


def draw_grid(screen, cam_pos, cam_ang):
    """World-fixed grid lines, for position/scale reference on the otherwise
    featureless background; spacing is in world meters, not screen pixels."""
    cx, cy = cam_pos
    xs = np.arange(np.floor((cx - GRID_RADIUS) / GRID_STEP),
                   np.ceil((cx + GRID_RADIUS) / GRID_STEP) + 1) * GRID_STEP
    ys = np.arange(np.floor((cy - GRID_RADIUS) / GRID_STEP),
                   np.ceil((cy + GRID_RADIUS) / GRID_STEP) + 1) * GRID_STEP
    for x in xs:
        col = (65, 65, 90) if abs(x) < 1e-6 else (48, 48, 55)
        p0 = to_screen(np.array([x, cy - GRID_RADIUS]), cam_pos, cam_ang)
        p1 = to_screen(np.array([x, cy + GRID_RADIUS]), cam_pos, cam_ang)
        pygame.draw.line(screen, col, p0.tolist(), p1.tolist(), 1)
    for y in ys:
        col = (65, 65, 90) if abs(y) < 1e-6 else (48, 48, 55)
        p0 = to_screen(np.array([cx - GRID_RADIUS, y]), cam_pos, cam_ang)
        p1 = to_screen(np.array([cx + GRID_RADIUS, y]), cam_pos, cam_ang)
        pygame.draw.line(screen, col, p0.tolist(), p1.tolist(), 1)


def draw_input_panel(screen, font, delta, T, intent):
    """Bottom-right bars: actual car input (delta/0.5, T) vs raw keyboard
    intent (steer, throttle). They coincide for the keyboard controller;
    for rl/pid the intent is read but has no effect on the car, so the
    two bars diverge."""
    w, h = 160, 72
    x0, y0 = SCREEN_W - w - 10, SCREEN_H - h - 10
    pygame.draw.rect(screen, (45, 45, 50), (x0, y0, w, h))
    bar_x, bar_w, bar_h = x0 + 46, 104, 12
    cx = bar_x + bar_w / 2
    rows = [("steer", -delta / 0.5, -intent["steer"]),
            ("thr", T, intent["throttle"])]
    for i, (label, car_val, user_val) in enumerate(rows):
        by = y0 + 8 + i * 26
        screen.blit(font.render(label, True, (200, 200, 200)), (x0 + 6, by))
        pygame.draw.rect(screen, (25, 25, 28), (bar_x, by, bar_w, bar_h))
        pygame.draw.line(screen, (90, 90, 95), (cx, by), (cx, by + bar_h), 1)
        cv = float(np.clip(car_val, -1.0, 1.0))
        fill_x = cx + cv * bar_w / 2
        pygame.draw.rect(screen, (70, 130, 240),
                         (min(cx, fill_x), by, max(abs(fill_x - cx), 1), bar_h))
        uv = float(np.clip(user_val, -1.0, 1.0))
        mx = cx + uv * bar_w / 2
        pygame.draw.line(screen, (230, 200, 60), (mx, by - 2), (mx, by + bar_h + 2), 2)
    legend_y = y0 + h - 16
    screen.blit(font.render("car", True, (70, 130, 240)), (x0 + 6, legend_y))
    screen.blit(font.render("input", True, (230, 200, 60)), (x0 + 50, legend_y))


def print_summary(ep_log, env, score, driver, mode, status):
    """Console benchmark line, same stats evaluate.py reports for RL/PID runs."""
    n = len(ep_log["vx"])
    if n == 0:
        return
    vx, beta, e_y = (np.array(ep_log[k]) for k in ("vx", "beta", "e_y"))
    n4 = max(n // 4, 1)
    print(f"[{driver}/{mode}] {status}: {n} steps ({n * env.DT:.1f} s), return = {score:.1f}")
    print(f"  mean |beta| = {np.degrees(np.abs(beta).mean()):.1f} deg "
          f"(settled {np.degrees(np.abs(beta[n4:]).mean()):.1f}, max {np.degrees(np.abs(beta).max()):.1f} deg), "
          f"mean |e_y| = {np.abs(e_y).mean():.2f} m")
    print(f"  vx settled mean = {vx[n4:].mean():.2f} m/s, max = {vx.max():.2f}")


def car_polygon(x, y, psi):
    base = np.array([[CAR_L / 2, CAR_W / 2], [CAR_L / 2, -CAR_W / 2],
                     [-CAR_L / 2, -CAR_W / 2], [-CAR_L / 2, CAR_W / 2]])
    c, s = np.cos(psi), np.sin(psi)
    return base @ np.array([[c, s], [-s, c]]) + [x, y]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=["circle", "random", "free"], default="circle")
    p.add_argument("--mode", choices=["drift", "grip"], default="drift",
                   help="reward shaping shown in the HUD")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--controller", choices=["keyboard", "rl", "pid", "pidref"],
                   default="keyboard", help="who drives")
    args = p.parse_args()
    args.model = f"models/{args.mode}_{args.track}/best_model"

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("DriftEnv")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    small_font = pygame.font.SysFont("monospace", 13)
    big = pygame.font.SysFont("monospace", 36, bold=True)

    env = DriftEnv(mode=args.mode, track_type=args.track)
    controller = make_controller(args.controller, env, args)
    obs, _ = env.reset(seed=args.seed)
    controller.reset()
    delta, T, score = 0.0, 0.0, 0.0
    intent = {"steer": 0.0, "throttle": 0.0}   # raw keyboard intent, for the input panel
    terms, cum = {}, {}          # last-step reward components and their totals
    ep_log = {"vx": [], "beta": [], "e_y": []}   # for the console benchmark summary
    cam_ang = env.state[2]
    over_msg = None

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                print_summary(ep_log, env, score, controller.name, args.mode, "quit")
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    print_summary(ep_log, env, score, controller.name, args.mode, "quit")
                    running = False
                if ev.key == pygame.K_r:
                    if over_msg is None:
                        print_summary(ep_log, env, score, controller.name, args.mode, "interrupted")
                    obs, _ = env.reset()
                    controller.reset()
                    delta, T, score, over_msg = 0.0, 0.0, 0.0, None
                    terms, cum = {}, {}
                    ep_log = {"vx": [], "beta": [], "e_y": []}
                    cam_ang = env.state[2]

        # ---------------- input -> action via the active controller
        if over_msg is None:
            keys = pygame.key.get_pressed()
            intent = {"steer": float(keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]),
                      "throttle": float(keys[pygame.K_UP] - keys[pygame.K_DOWN])}
            action = controller.act(obs, intent, env.DT)
            delta, T = float(action[0]), float(action[1])
            obs, reward, term, trunc, info = env.step(np.array([delta, T]))
            score += reward
            terms = info["reward_terms"]
            for k, v in terms.items():
                cum[k] = cum.get(k, 0.0) + v
            ep_log["vx"].append(info["vx"]); ep_log["beta"].append(info["beta"])
            ep_log["e_y"].append(info["e_y"])
            if term:
                over_msg = (None if args.track == "free"
                            else "OFF TRACK  -  press R")
                print_summary(ep_log, env, score, controller.name, args.mode, "off-track")
            elif info["finished"]:
                over_msg = "TRACK FINISHED  -  press R"
                print_summary(ep_log, env, score, controller.name, args.mode, "finished")

        x, y, psi, vx, vy, r = env.state
        v = float(np.hypot(vx, vy))

        # ---------------- camera: follow velocity direction (heading when slow)
        vg = np.array([vx * np.cos(psi) - vy * np.sin(psi),
                       vx * np.sin(psi) + vy * np.cos(psi)])
        target_ang = float(np.arctan2(vg[1], vg[0])) if v > 3.0 else psi
        cam_ang += wrap(target_ang - cam_ang) * env.DT / CAM_TAU
        cam_pos = np.array([x, y])

        # ---------------- draw
        screen.fill((30, 30, 35))
        draw_grid(screen, cam_pos, cam_ang)
        tr = env.track
        if args.track != "free":     # free roam has no meaningful track boundary
            for line, col, w in ((tr.left, (220, 220, 220), 3),
                                 (tr.right, (220, 220, 220), 3),
                                 (tr.xy, (90, 90, 100), 1)):
                pts = to_screen(line, cam_pos, cam_ang)
                pygame.draw.lines(screen, col, tr.closed, pts.tolist(), w)

        pygame.draw.polygon(screen, (70, 130, 240),
                            to_screen(car_polygon(x, y, psi), cam_pos, cam_ang).tolist())
        nose = to_screen(np.array([x + CAR_L / 2 * np.cos(psi), y + CAR_L / 2 * np.sin(psi)]),
                         cam_pos, cam_ang)
        pygame.draw.circle(screen, (255, 255, 255), nose.tolist(), 3)
        tip = to_screen(np.array([x, y]) + 0.5 * vg, cam_pos, cam_ang)
        org = to_screen(np.array([x, y]), cam_pos, cam_ang)
        pygame.draw.line(screen, (240, 80, 80), org.tolist(), tip.tolist(), 2)

        beta = np.degrees(np.arctan2(vy, max(vx, 0.5)))
        hud = [f"speed {v:5.1f} m/s   slip {beta:+6.1f} deg",
               f"steer {np.degrees(delta):+5.1f} deg   throttle {T:+4.2f}",
               f"score {score:8.1f}   t {env.steps * env.DT:5.1f} s",
               f"driver: {controller.name}"]
        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, (230, 230, 230)), (10, 10 + 20 * i))

        # ---------------- live reward breakdown: per-step value and running total
        beta_lbl = "+|beta|" if args.mode == "drift" else "-beta^2"
        labels = {"progress": "+prog", "e_y": "-e_y^2", "d_delta": "-ddot^2",
                  "beta": beta_lbl, "alive": "+alive", "term": "OFFTRACK",
                  "finish": "FINISH"}
        y0 = 10 + 20 * len(hud) + 8
        screen.blit(font.render(f"reward [{args.mode}]      now     total",
                                True, (200, 200, 120)), (10, y0))
        for j, k in enumerate(labels):
            if k not in terms and k not in cum:
                continue
            now, tot = terms.get(k, 0.0), cum.get(k, 0.0)
            col = (120, 220, 120) if now >= 0 else (235, 130, 130)
            line = f"  {labels[k]:8s} {now:+8.2f} {tot:+9.1f}"
            screen.blit(font.render(line, True, col), (10, y0 + 18 * (j + 1)))

        screen.blit(font.render("arrows: drive   R: restart   ESC: quit",
                                True, (140, 140, 140)), (10, SCREEN_H - 26))
        draw_input_panel(screen, small_font, delta, T, intent)
        if over_msg:
            t = big.render(over_msg, True, (255, 200, 60))
            screen.blit(t, t.get_rect(center=(SCREEN_W / 2, SCREEN_H / 2)))

        pygame.display.flip()
        clock.tick(50)  # real-time: matches dt = 0.02 s

    pygame.quit()


if __name__ == "__main__":
    main()
