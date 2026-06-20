"""Manual drift game: top-down chase view, camera aligned with velocity.

Controls:
    LEFT / RIGHT   steer (first-order lag to full lock)
    UP / DOWN      throttle / brake-reverse (first-order lag)
    R              restart (new layout on random tracks)
    ESC            quit

Usage:
    python game.py                          # circular track
    python game.py --track random           # random track, new layout per run
    python game.py --track random --seed 7  # reproducible first layout
    python game.py --mode grip              # HUD shows grip reward shaping
    python game.py --demo                   # trained drift agent plays instead
    python game.py --demo --screenshot report/figures/game.png
"""

import argparse
import os

import numpy as np
import pygame

from drift_env import DriftEnv

SCREEN_W, SCREEN_H = 900, 700
SCALE = 8.0                    # px per meter
CAR_L, CAR_W = 4.0, 1.8        # drawn footprint [m]
TAU_STEER, TAU_THR = 0.12, 0.25   # first-order input lag time constants [s]
CAM_TAU = 0.3                  # camera heading smoothing [s]
CAM_Y = 0.62                   # car drawn below screen center (look-ahead)


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


def car_polygon(x, y, psi):
    base = np.array([[CAR_L / 2, CAR_W / 2], [CAR_L / 2, -CAR_W / 2],
                     [-CAR_L / 2, -CAR_W / 2], [-CAR_L / 2, CAR_W / 2]])
    c, s = np.cos(psi), np.sin(psi)
    return base @ np.array([[c, s], [-s, c]]) + [x, y]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=["circle", "random"], default="circle")
    p.add_argument("--mode", choices=["drift", "grip"], default="drift",
                   help="reward shaping shown in the HUD")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--demo", action="store_true", help="trained agent drives")
    p.add_argument("--model", default="models/grip_random/best_model")
    p.add_argument("--screenshot", default=None, help="save a frame and exit")
    args = p.parse_args()

    model = None
    if args.demo:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("DriftEnv")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    big = pygame.font.SysFont("monospace", 36, bold=True)

    env = DriftEnv(mode=args.mode, track_type=args.track)
    obs, _ = env.reset(seed=args.seed)
    delta, T, score = 0.0, 0.0, 0.0
    terms, cum = {}, {}          # last-step reward components and their totals
    cam_ang = env.state[2]
    over_msg = None
    frame = 0

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                if ev.key == pygame.K_r:
                    obs, _ = env.reset()
                    delta, T, score, over_msg = 0.0, 0.0, 0.0, None
                    terms, cum = {}, {}
                    cam_ang = env.state[2]

        # ---------------- input -> action (first-order lag on key targets)
        if over_msg is None:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
                delta, T = float(action[0]), float(action[1])
            else:
                keys = pygame.key.get_pressed()
                steer_t = 0.5 * (keys[pygame.K_LEFT] - keys[pygame.K_RIGHT])
                thr_t = 1.0 * (keys[pygame.K_UP] - keys[pygame.K_DOWN])
                delta += (steer_t - delta) * env.DT / TAU_STEER
                T += (thr_t - T) * env.DT / TAU_THR
            obs, reward, term, trunc, info = env.step(np.array([delta, T]))
            score += reward
            terms = info["reward_terms"]
            for k, v in terms.items():
                cum[k] = cum.get(k, 0.0) + v
            if term:
                over_msg = "OFF TRACK  -  press R"
            elif info["finished"]:
                over_msg = "TRACK FINISHED  -  press R"

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
        tr = env.track
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
               f"score {score:8.1f}   t {env.steps * env.DT:5.1f} s"]
        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, (230, 230, 230)), (10, 10 + 20 * i))

        # ---------------- live reward breakdown: per-step value and running total
        beta_lbl = "+|beta|" if args.mode == "drift" else "-beta^2"
        labels = {"vx": "vx*cos", "e_y": "-e_y^2", "d_delta": "-ddot^2",
                  "beta": beta_lbl, "term": "OFFTRACK"}
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
        if over_msg:
            t = big.render(over_msg, True, (255, 200, 60))
            screen.blit(t, t.get_rect(center=(SCREEN_W / 2, SCREEN_H / 2)))

        pygame.display.flip()
        frame += 1
        if args.screenshot and (frame >= 300 or over_msg):
            os.makedirs(os.path.dirname(args.screenshot) or ".", exist_ok=True)
            pygame.image.save(screen, args.screenshot)
            print("saved", args.screenshot)
            running = False
        clock.tick(50)  # real-time: matches dt = 0.02 s

    pygame.quit()


if __name__ == "__main__":
    main()
