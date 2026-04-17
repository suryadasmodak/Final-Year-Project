import customtkinter as ctk
from tkinter import messagebox
import threading
import cv2
from PIL import Image
import numpy as np
import tenseal as ts
from datetime import datetime
import os
from facenet_feature import extract_feature_vector
from cancelable_transform import generate_transform_key, cancelable_transform
from encrypt_store import (
    load_context, create_ckks_context,
    save_context, encrypt_template, get_database
)
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.80
CONFIDENCE_THRESHOLD = 0.9
REAL_COUNTER_REQUIRED = 5
CAMERA_INDEX = 1
YOLO_MODEL_PATH = r"c:\Users\SURYA DAS MODAK\OneDrive\Desktop\FY project\Final-Year-Project\l_version_1_300.pt"
CONTEXT_PATH = r"c:\Users\SURYA DAS MODAK\OneDrive\Desktop\FY project\Final-Year-Project\ckks_context.pkl"
KEYS_DIR = r"c:\Users\SURYA DAS MODAK\OneDrive\Desktop\FY project\Final-Year-Project\keys"

# ─── App Setup ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class BiometricApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🔐 Homomorphic Facial Authentication System")
        self.geometry("1200x750")
        self.resizable(False, False)

        # State
        self.camera_running = False
        self.cap = None
        self.yolo_model = None
        self.context = None
        self.real_counter = 0
        self.captured = False
        self.current_embedding = None
        self.current_frame = None
        self.scanning = False
        self.status_ready = None
        self.status_error = None

        # Build UI first
        self.build_ui()

        # Load users and logs
        self.refresh_users()
        self.refresh_logs()

        # Start camera
        self.start_camera()

        # Load models in background
        self.load_models()

        # Handle close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ─── Load Models ──────────────────────────────────────────────────────────
    def load_models(self):
        self.update_status("⏳ Loading models...", "yellow")
        self.status_ready = None
        threading.Thread(
            target=self._load_models_thread, daemon=True).start()
        self.after(500, self._check_model_status)

    def _load_models_thread(self):
        try:
            if os.path.exists(CONTEXT_PATH):
                self.context = load_context(CONTEXT_PATH)
            else:
                self.context = create_ckks_context()
                save_context(self.context, CONTEXT_PATH)

            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            self.yolo_model.to('cuda')

            self.status_ready = True

        except Exception as e:
            self.status_error = str(e)
            self.status_ready = False

    def _check_model_status(self):
        if self.status_ready is True:
            self.update_status("✅ System Ready", "green")
        elif self.status_ready is False:
            self.update_status(
                f"❌ Error: {self.status_error}", "red")
        else:
            self.after(500, self._check_model_status)

    # ─── Build UI ─────────────────────────────────────────────────────────────
    def build_ui(self):
        # ── Title Bar ──
        title_frame = ctk.CTkFrame(self, height=60, fg_color="#1a1a2e")
        title_frame.pack(fill="x", padx=0, pady=0)

        ctk.CTkLabel(
            title_frame,
            text="🔐  HOMOMORPHIC FACIAL AUTHENTICATION SYSTEM",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#00d4ff"
        ).pack(side="left", padx=20, pady=15)

        self.status_label = ctk.CTkLabel(
            title_frame,
            text="⏳ Loading...",
            font=ctk.CTkFont(size=13),
            text_color="yellow"
        )
        self.status_label.pack(side="right", padx=20)

        # ── Main Layout ──
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ── Left Panel — Camera ──
        left_panel = ctk.CTkFrame(
            main_frame, fg_color="#16213e", width=480)
        left_panel.pack(side="left", fill="both", padx=(0, 5))
        left_panel.pack_propagate(False)

        ctk.CTkLabel(
            left_panel,
            text="📷  LIVE CAMERA FEED",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00d4ff"
        ).pack(pady=10)

        self.camera_label = ctk.CTkLabel(
            left_panel, text="", width=460, height=360)
        self.camera_label.pack(padx=10, pady=5)

        ctk.CTkLabel(
            left_panel,
            text="Liveness Detection Progress:",
            font=ctk.CTkFont(size=12)
        ).pack(pady=(10, 2))

        self.progress_bar = ctk.CTkProgressBar(
            left_panel, width=440, height=20)
        self.progress_bar.pack(padx=10)
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(
            left_panel,
            text="Real: 0/5",
            font=ctk.CTkFont(size=12)
        )
        self.progress_label.pack(pady=2)

        self.scan_btn = ctk.CTkButton(
            left_panel,
            text="▶  START SCAN",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45,
            fg_color="#00d4ff",
            text_color="black",
            command=self.start_scan
        )
        self.scan_btn.pack(padx=10, pady=10, fill="x")

        self.result_label = ctk.CTkLabel(
            left_panel,
            text="",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.result_label.pack(pady=5)

        # ── Middle Panel — Enrollment ──
        mid_panel = ctk.CTkFrame(
            main_frame, fg_color="#16213e", width=280)
        mid_panel.pack(side="left", fill="both", padx=5)
        mid_panel.pack_propagate(False)

        ctk.CTkLabel(
            mid_panel,
            text="📋  ENROLLMENT PANEL",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00d4ff"
        ).pack(pady=10)

        self.face_preview = ctk.CTkLabel(
            mid_panel,
            text="No face captured yet",
            width=200, height=200,
            fg_color="#0f3460",
            corner_radius=10
        )
        self.face_preview.pack(padx=10, pady=10)

        ctk.CTkLabel(
            mid_panel, text="Full Name:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=15)
        self.name_entry = ctk.CTkEntry(
            mid_panel, width=250,
            placeholder_text="Enter full name...")
        self.name_entry.pack(padx=15, pady=(2, 10))

        ctk.CTkLabel(
            mid_panel, text="User ID:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=15)
        self.userid_entry = ctk.CTkEntry(
            mid_panel, width=250,
            placeholder_text="Enter user ID...")
        self.userid_entry.pack(padx=15, pady=(2, 10))

        self.enroll_btn = ctk.CTkButton(
            mid_panel,
            text="✅  ENROLL USER",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=40,
            fg_color="#00b894",
            command=self.enroll_user,
            state="disabled"
        )
        self.enroll_btn.pack(padx=15, pady=5, fill="x")

        self.score_frame = ctk.CTkFrame(mid_panel, fg_color="#0f3460")
        self.score_frame.pack(padx=15, pady=10, fill="x")

        ctk.CTkLabel(
            self.score_frame,
            text="Similarity Score",
            font=ctk.CTkFont(size=12)
        ).pack(pady=(10, 2))

        self.score_label = ctk.CTkLabel(
            self.score_frame,
            text="--",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#00d4ff"
        )
        self.score_label.pack(pady=(0, 10))

        # ── Right Panel — Tabs ──
        right_panel = ctk.CTkFrame(main_frame, fg_color="#16213e")
        right_panel.pack(
            side="left", fill="both", expand=True, padx=(5, 0))

        self.tabview = ctk.CTkTabview(
            right_panel, fg_color="#0f3460")
        self.tabview.pack(
            fill="both", expand=True, padx=10, pady=10)

        # Tab 1 — Enrolled Users
        self.tabview.add("👥 Enrolled Users")
        users_tab = self.tabview.tab("👥 Enrolled Users")

        self.users_text = ctk.CTkTextbox(
            users_tab,
            font=ctk.CTkFont(family="Courier", size=12),
            fg_color="#1a1a2e"
        )
        self.users_text.pack(
            fill="both", expand=True, padx=5, pady=5)

        ctk.CTkButton(
            users_tab,
            text="🔄 Refresh",
            width=100,
            command=self.refresh_users
        ).pack(pady=5)

        # Tab 2 — Access Logs
        self.tabview.add("📜 Access Logs")
        logs_tab = self.tabview.tab("📜 Access Logs")

        self.logs_text = ctk.CTkTextbox(
            logs_tab,
            font=ctk.CTkFont(family="Courier", size=12),
            fg_color="#1a1a2e"
        )
        self.logs_text.pack(
            fill="both", expand=True, padx=5, pady=5)

        ctk.CTkButton(
            logs_tab,
            text="🔄 Refresh",
            width=100,
            command=self.refresh_logs
        ).pack(pady=5)

        # Tab 3 — Admin Panel
        self.tabview.add("⚙️ Admin Panel")
        admin_tab = self.tabview.tab("⚙️ Admin Panel")

        ctk.CTkLabel(
            admin_tab,
            text="Admin Controls",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)

        ctk.CTkButton(
            admin_tab,
            text="🗑️  Clear All Users",
            fg_color="#e74c3c",
            hover_color="#c0392b",
            command=self.clear_all_users
        ).pack(padx=20, pady=5, fill="x")

        ctk.CTkButton(
            admin_tab,
            text="🗑️  Clear All Logs",
            fg_color="#e67e22",
            hover_color="#d35400",
            command=self.clear_all_logs
        ).pack(padx=20, pady=5, fill="x")

        ctk.CTkButton(
            admin_tab,
            text="🔄  Refresh All",
            fg_color="#2980b9",
            command=self.refresh_all
        ).pack(padx=20, pady=5, fill="x")

        self.stats_frame = ctk.CTkFrame(admin_tab, fg_color="#0f3460")
        self.stats_frame.pack(padx=20, pady=20, fill="x")

        ctk.CTkLabel(
            self.stats_frame,
            text="📊 System Statistics",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=10)

        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="",
            font=ctk.CTkFont(size=12),
            justify="left"
        )
        self.stats_label.pack(padx=10, pady=(0, 10))
        self.update_stats()

    # ─── Camera ───────────────────────────────────────────────────────────────
    def start_camera(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.camera_running = True
        threading.Thread(
            target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.current_frame = frame.copy()
            if self.scanning and self.yolo_model:
                self._process_frame(frame)
            else:
                self._display_frame(frame)

    def _process_frame(self, frame):
        results = self.yolo_model(frame, verbose=False)
        classNames = ["fake", "real"]

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = classNames[cls]

                if label == "real" and conf > CONFIDENCE_THRESHOLD:
                    self.real_counter += 1
                else:
                    self.real_counter = 0

                if (self.real_counter >= REAL_COUNTER_REQUIRED
                        and not self.captured):
                    face = frame[y1:y2, x1:x2]
                    cv2.imwrite("captured_face.jpg", face)
                    self.captured = True
                    self.scanning = False
                    self.after(0, self._on_face_captured)

                color = (0, 255, 0) if label == "real" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label.upper()} {int(conf*100)}%",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        progress = min(
            self.real_counter / REAL_COUNTER_REQUIRED, 1.0)
        self.after(0, lambda: self.progress_bar.set(progress))
        self.after(0, lambda: self.progress_label.configure(
            text=f"Real: {self.real_counter}/{REAL_COUNTER_REQUIRED}"))
        self._display_frame(frame)

    def _display_frame(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((460, 360))
            imgtk = ctk.CTkImage(img, size=(460, 360))
            self.after(0, lambda: self.camera_label.configure(
                image=imgtk))
            self.camera_label.image = imgtk
        except Exception:
            pass

    # ─── Scan ─────────────────────────────────────────────────────────────────
    def start_scan(self):
        if self.yolo_model is None:
            messagebox.showwarning(
                "Wait", "Models still loading. Please wait!")
            return
        self.real_counter = 0
        self.captured = False
        self.scanning = True
        self.scan_btn.configure(
            state="disabled", text="⏳ Scanning...")
        self.result_label.configure(
            text="👁️ Looking for real face...",
            text_color="yellow")
        self.enroll_btn.configure(state="disabled")
        self.score_label.configure(text="--")
        self.update_status("🔍 Scanning...", "yellow")

    def _on_face_captured(self):
        self.update_status("🧠 Processing face...", "yellow")
        self.result_label.configure(
            text="📸 Face captured! Processing...",
            text_color="cyan")
        threading.Thread(
            target=self._process_captured_face,
            daemon=True).start()

    def _process_captured_face(self):
        try:
            embedding = extract_feature_vector("captured_face.jpg")
            self.current_embedding = np.array(
                embedding, dtype=np.float64)

            # Show face preview
            face_img = Image.open("captured_face.jpg")
            face_img = face_img.resize((200, 200))
            face_ctk = ctk.CTkImage(face_img, size=(200, 200))
            self.after(0, lambda: self.face_preview.configure(
                image=face_ctk, text=""))
            self.face_preview.image = face_ctk

            # Authenticate
            matched, best_match, best_score = self._authenticate()
            self.after(0, lambda: self._show_result(
                matched, best_match, best_score))

        except Exception as e:
            self.after(0, lambda: self.update_status(
                f"❌ Error: {e}", "red"))
            self.after(0, lambda: self.scan_btn.configure(
                state="normal", text="▶  START SCAN"))

    def _authenticate(self):
        db = get_database()
        collection = db["enrolled_users"]
        users = list(collection.find({}))

        if len(users) == 0:
            return False, None, -1

        best_match = None
        best_score = -1

        for user in users:
            user_id = user["user_id"]
            W = generate_transform_key(user_id, keys_dir=KEYS_DIR)
            protected_query = cancelable_transform(
                self.current_embedding, W)
            enc_query = ts.ckks_vector(
                self.context, protected_query.tolist())
            enc_stored = ts.ckks_vector_from(
                self.context, user["encrypted_template"])
            dot_product = enc_query.dot(enc_stored)
            score = float(dot_product.decrypt()[0])

            if score > best_score:
                best_score = score
                best_match = user

        matched = best_score >= SIMILARITY_THRESHOLD
        return matched, best_match, best_score

    def _show_result(self, matched, best_match, best_score):
        self.score_label.configure(text=f"{best_score:.4f}")
        self.scan_btn.configure(
            state="normal", text="▶  START SCAN")

        if matched:
            self.result_label.configure(
                text=f"✅ ACCESS GRANTED\n👤 {best_match['name']}",
                text_color="#00b894")
            self.update_status("✅ Access Granted", "green")
            self.enroll_btn.configure(state="disabled")
            self._log_access(
                best_match['name'],
                best_match['user_id'],
                best_score, "GRANTED")
        else:
            self.result_label.configure(
                text="⚠️ Unknown Face\nFill details to enroll →",
                text_color="orange")
            self.update_status("⚠️ Unknown Face", "orange")
            self.enroll_btn.configure(state="normal")
            self._log_access(
                "Unknown", "N/A", best_score, "DENIED")

        self.refresh_logs()
        self.update_stats()

    # ─── Enroll ───────────────────────────────────────────────────────────────
    def enroll_user(self):
        name = self.name_entry.get().strip()
        user_id = self.userid_entry.get().strip()

        if not name or not user_id:
            messagebox.showerror(
                "Error", "Please enter both Name and User ID!")
            return

        if self.current_embedding is None:
            messagebox.showerror(
                "Error", "No face captured! Please scan first.")
            return

        db = get_database()
        collection = db["enrolled_users"]

        if collection.find_one({"user_id": user_id}):
            messagebox.showerror(
                "Error", f"User ID '{user_id}' already exists!")
            return

        threading.Thread(
            target=self._enroll_thread,
            args=(name, user_id),
            daemon=True
        ).start()

    def _enroll_thread(self, name, user_id):
        try:
            self.after(0, lambda: self.update_status(
                "🔐 Enrolling user...", "yellow"))

            W = generate_transform_key(user_id, keys_dir=KEYS_DIR)
            protected = cancelable_transform(
                self.current_embedding, W)
            encrypted_bytes = encrypt_template(
                protected, self.context)

            db = get_database()
            collection = db["enrolled_users"]
            collection.insert_one({
                "user_id": user_id,
                "name": name,
                "encrypted_template": encrypted_bytes,
                "enrolled_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")
            })

            self.after(0, lambda: self._on_enrolled(name))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror(
                "Error", str(e)))

    def _on_enrolled(self, name):
        self.result_label.configure(
            text=f"✅ ENROLLED!\n👤 {name}",
            text_color="#00b894")
        self.update_status("✅ User Enrolled!", "green")
        self.enroll_btn.configure(state="disabled")
        self.name_entry.delete(0, "end")
        self.userid_entry.delete(0, "end")
        self.refresh_users()
        self.update_stats()
        messagebox.showinfo(
            "Success", f"✅ {name} enrolled successfully!")

    # ─── Logs ─────────────────────────────────────────────────────────────────
    def _log_access(self, name, user_id, score, status):
        db = get_database()
        logs = db["access_logs"]
        logs.insert_one({
            "name": name,
            "user_id": user_id,
            "score": round(score, 4),
            "status": status,
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S")
        })

    def refresh_logs(self):
        db = get_database()
        logs = db["access_logs"]
        all_logs = list(logs.find({}).sort(
            "timestamp", -1).limit(50))
        self.logs_text.configure(state="normal")
        self.logs_text.delete("1.0", "end")
        if not all_logs:
            self.logs_text.insert(
                "end", "No access logs yet.\n")
        for log in all_logs:
            icon = "✅" if log["status"] == "GRANTED" else "❌"
            self.logs_text.insert("end",
                f"{icon} {log['timestamp']}\n"
                f"   Name: {log['name']} | ID: {log['user_id']}\n"
                f"   Score: {log['score']} | "
                f"Status: {log['status']}\n"
                f"{'─'*40}\n"
            )
        self.logs_text.configure(state="disabled")

    # ─── Users ────────────────────────────────────────────────────────────────
    def refresh_users(self):
        db = get_database()
        collection = db["enrolled_users"]
        users = list(collection.find(
            {}, {"user_id": 1, "name": 1, "enrolled_at": 1}))
        self.users_text.configure(state="normal")
        self.users_text.delete("1.0", "end")
        if not users:
            self.users_text.insert(
                "end", "No enrolled users yet.\n")
        for i, user in enumerate(users, 1):
            self.users_text.insert("end",
                f"👤 User {i}\n"
                f"   Name: {user['name']}\n"
                f"   ID:   {user['user_id']}\n"
                f"   Enrolled: {user['enrolled_at']}\n"
                f"{'─'*40}\n"
            )
        self.users_text.configure(state="disabled")

    # ─── Admin ────────────────────────────────────────────────────────────────
    def clear_all_users(self):
        if messagebox.askyesno(
                "Confirm", "⚠️ Delete ALL enrolled users?"):
            db = get_database()
            db["enrolled_users"].delete_many({})
            self.refresh_users()
            self.update_stats()
            messagebox.showinfo("Done", "All users cleared!")

    def clear_all_logs(self):
        if messagebox.askyesno(
                "Confirm", "⚠️ Delete ALL access logs?"):
            db = get_database()
            db["access_logs"].delete_many({})
            self.refresh_logs()
            messagebox.showinfo("Done", "All logs cleared!")

    def refresh_all(self):
        self.refresh_users()
        self.refresh_logs()
        self.update_stats()

    def update_stats(self):
        db = get_database()
        total_users = db["enrolled_users"].count_documents({})
        total_logs = db["access_logs"].count_documents({})
        granted = db["access_logs"].count_documents(
            {"status": "GRANTED"})
        denied = db["access_logs"].count_documents(
            {"status": "DENIED"})
        self.stats_label.configure(
            text=f"👥 Total Users:     {total_users}\n"
                 f"📜 Total Scans:     {total_logs}\n"
                 f"✅ Access Granted:  {granted}\n"
                 f"❌ Access Denied:   {denied}"
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────
    def update_status(self, text, color):
        colors = {
            "green": "#00b894",
            "yellow": "#fdcb6e",
            "red": "#e74c3c",
            "orange": "#e67e22"
        }
        self.status_label.configure(
            text=text,
            text_color=colors.get(color, "white")
        )

    def on_close(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.destroy()


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = BiometricApp()
    app.mainloop()