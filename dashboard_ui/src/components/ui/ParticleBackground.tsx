import { useEffect, useRef } from "react";

const MAX_PARTICLES = 28;
const CONNECTION_DIST = 100;
const MOUSE_RADIUS = 150;

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
}

export default function ParticleBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: -9999, y: -9999 });
  const rafRef = useRef(0);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reducedMotion) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    let w = 0;
    let h = 0;
    let frame = 0;
    let active = true;
    let particles: Particle[] = [];

    const lowPowerMode = (navigator.hardwareConcurrency ?? 8) <= 4 || window.innerWidth < 768;

    const resize = () => {
      w = window.innerWidth;
      h = window.innerHeight;
      canvas.width = w;
      canvas.height = h;
    };

    const init = () => {
      const maxCount = lowPowerMode ? 14 : MAX_PARTICLES;
      const count = Math.min(maxCount, Math.floor(w / 60));
      particles = [];

      for (let i = 0; i < count; i += 1) {
        particles.push({
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * 0.25,
          vy: (Math.random() - 0.5) * 0.25,
          r: Math.random() * 1.2 + 0.4,
        });
      }
    };

    const tick = () => {
      if (!active) return;

      frame += 1;
      if (frame & 1) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      ctx.clearRect(0, 0, w, h);
      const mx = mouseRef.current.x;
      const my = mouseRef.current.y;
      const count = particles.length;

      for (let i = 0; i < count; i += 1) {
        const p = particles[i];
        const dmx = p.x - mx;
        const dmy = p.y - my;
        const distSq = dmx * dmx + dmy * dmy;

        if (distSq < MOUSE_RADIUS * MOUSE_RADIUS && distSq > 1) {
          const dist = Math.sqrt(distSq);
          const force = ((MOUSE_RADIUS - dist) / MOUSE_RADIUS) * 0.01;
          p.vx += (dmx / dist) * force;
          p.vy += (dmy / dist) * force;
        }

        p.vx *= 0.99;
        p.vy *= 0.99;
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < -5) p.x = w + 5;
        if (p.x > w + 5) p.x = -5;
        if (p.y < -5) p.y = h + 5;
        if (p.y > h + 5) p.y = -5;
      }

      ctx.lineWidth = 0.4;
      for (let i = 0; i < count; i += 1) {
        for (let j = i + 1; j < count; j += 1) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const distSq = dx * dx + dy * dy;

          if (distSq < CONNECTION_DIST * CONNECTION_DIST) {
            const alpha = (1 - Math.sqrt(distSq) / CONNECTION_DIST) * 0.12;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(6,182,212,${alpha})`;
            ctx.stroke();
          }
        }
      }

      ctx.fillStyle = "rgba(6,182,212,0.3)";
      for (let i = 0; i < count; i += 1) {
        ctx.beginPath();
        ctx.arc(particles[i].x, particles[i].y, particles[i].r, 0, 6.2832);
        ctx.fill();
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    const onResize = () => {
      resize();
      init();
    };

    const onMouse = (event: MouseEvent) => {
      mouseRef.current = { x: event.clientX, y: event.clientY };
    };

    const onLeave = () => {
      mouseRef.current = { x: -9999, y: -9999 };
    };

    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        if (!active) {
          active = true;
          rafRef.current = requestAnimationFrame(tick);
        }
      } else {
        active = false;
        cancelAnimationFrame(rafRef.current);
      }
    };

    resize();
    init();
    rafRef.current = requestAnimationFrame(tick);

    window.addEventListener("resize", onResize, { passive: true });
    window.addEventListener("mousemove", onMouse, { passive: true });
    window.addEventListener("mouseleave", onLeave, { passive: true });
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      active = false;
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("mousemove", onMouse);
      window.removeEventListener("mouseleave", onLeave);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none fixed inset-0 z-0"
      style={{ opacity: 0.45, willChange: "contents" }}
    />
  );
}
