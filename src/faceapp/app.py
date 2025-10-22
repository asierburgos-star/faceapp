from .config import load_config
from .db import load_db, save_db, add_or_update_user
from .camera import open_camera, limit_fps
from .preproc import preprocess_face_for_embedding
from .embedding import represent_from_preprocessed, cosine_distance
from collections import deque
import cv2
import numpy as np
import time

def run_enroll(name: str):
    cfg = load_config()
    cap = open_camera(cfg)
    if not cap or not cap.isOpened():
        print("[red]No se pudo abrir la cámara[/red]")
        return

    collected = []
    print(f"[green]Registrando a {name}...[/green]")
    for i in range(4):
        ok, frame = cap.read()
        if not ok:
            continue
        face_rgb, angle, info = preprocess_face_for_embedding(frame, cfg)
        if info["reason"] == "ok":
            emb = represent_from_preprocessed(face_rgb, cfg)
            collected.append(emb)
        cv2.imshow("Enroll", frame)
        if (cv2.waitKey(10) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not collected:
        print("[yellow]No se consiguió ninguna muestra válida[/yellow]")
        return

    mean_emb = np.mean(collected, axis=0)
    mean_emb /= (np.linalg.norm(mean_emb) + 1e-9)

    db = load_db(cfg["paths"]["db_path"])
    db = add_or_update_user(db, name, mean_emb)
    save_db(cfg["paths"]["db_path"], db)
    print(f"[green]Usuario {name} registrado.[/green]")

def run_recognize():
    cfg = load_config()
    db = load_db(cfg["paths"]["db_path"])
    if not db["users"]:
        print("[yellow]No hay usuarios en la base. Ejecuta 'faceapp enroll' primero.[/yellow]")
        return

    cap = open_camera(cfg)
    if not cap or not cap.isOpened():
        print("[red]No se pudo abrir la cámara[/red]")
        return

    recent = deque(maxlen=cfg["pipeline"]["stability_frames"])
    last_label = "Desconocido"
    best_dist = 1e9
    frame_idx = 0
    prev_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        if frame_idx % cfg["pipeline"]["process_every_n"] == 0:
            face_rgb, angle, info = preprocess_face_for_embedding(frame, cfg)
            if info["reason"] == "ok":
                emb = represent_from_preprocessed(face_rgb, cfg)
                label, best_dist = match_db(emb, db, cfg)
                recent.append(label)
                if len(recent) == recent.maxlen:
                    # voto mayoritario
                    counts = {x: recent.count(x) for x in set(recent)}
                    last_label = max(counts, key=counts.get)
            else:
                recent.append("Desconocido")

        cv2.putText(frame, f"{last_label} (dist={best_dist:.3f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0) if last_label!="Desconocido" else (0,0,255), 2)

        cv2.imshow("Reconocimiento", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        prev_t = limit_fps(prev_t, cfg["camera"]["target_fps"])
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

def match_db(emb, db, cfg):
    best_name = "Desconocido"
    best_dist = 1e9
    thr = cfg["deepface"]["cosine_threshold"]
    for u in db["users"]:
        dist = cosine_distance(emb, np.array(u["embedding"]))
        if dist < best_dist:
            best_dist = dist
            best_name = u["name"] if dist < thr else "Desconocido"
    return best_name, best_dist
