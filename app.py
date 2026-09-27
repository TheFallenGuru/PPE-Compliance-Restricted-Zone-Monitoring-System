#!/usr/bin/env python
"""
app.py — Professional Industrial HMI for PPE Compliance & Restricted Zone Monitoring.

Features:
- Dark industrial control-room dashboard layout
- Native Windows file picker for arbitrary video files from any directory
- Camera index probing and selection for live webcam monitoring
- Complete 10-class PPE compliance policy configuration
- Interactive multi-point polygon ROI drawing tool with original resolution coordinate mapping
- Embedded live video canvas with completely isolated, fail-safe rendering
- Real-time worker compliance status, active track grid, and intrusion alarm banner
- Live violation event log viewer and system diagnostic console
- Preserves all existing detection, tracking, smoothing, intrusion, and logging contracts
"""

import os
import sys
import threading
import time
import subprocess
import csv
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# Import the existing detection engine
import main

REQUIRED_PACKAGES = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("ultralytics", "ultralytics"),
    ("deep_sort_realtime", "deep-sort-realtime"),
    ("shapely", "shapely"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("PIL", "pillow"),
]


def check_dependencies():
    """Verify that all required dependencies are importable."""
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def check_model_weights():
    """Verify that models/yolo8m.pt exists and matches the SH17 schema."""
    model_path = os.path.join(os.getcwd(), "models", "yolo8m.pt")
    if not os.path.exists(model_path):
        return False, f"Model file not found at: {model_path}\nPlease ensure the custom SH17 weights are in models/yolo8m.pt.\nDo NOT download stock COCO weights."
    try:
        main.validate_sh17_model(model_path)
        return True, "SH17 17-class model validated successfully."
    except Exception as e:
        return False, f"Model validation failed: {e}"


def probe_cameras(max_cams=4):
    """Detect available camera indexes."""
    available = []
    for idx in range(max_cams):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available if available else [0]


AVAILABLE_PPE_OPTIONS = [
    ("helmet", "Helmet"),
    ("safety-vest", "Safety Vest"),
    ("gloves", "Gloves"),
    ("shoes", "Shoes"),
    ("face-mask", "Face Mask"),
    ("glasses", "Glasses"),
    ("face-guard", "Face Guard"),
    ("ear-mufs", "Ear Muffs"),
    ("safety-suit", "Safety Suit"),
    ("medical-suit", "Medical Suit"),
]
DEFAULT_REQUIRED_PPE = {'helmet', 'safety-vest', 'gloves', 'shoes', 'face-mask'}


class PolygonDrawingDialog(tk.Toplevel):
    """Interactive multi-point polygon editor dialog rendered directly on the first video frame."""

    def __init__(self, parent, frame, initial_polygon=None):
        super().__init__(parent)
        self.title("Define Restricted Zone (Click to Add Polygon Points)")
        self.configure(bg="#0F172A")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.confirmed_polygon = None
        self.orig_h, self.orig_w = frame.shape[:2]

        # Calculate display scale preserving aspect ratio (max 960x540)
        max_w, max_h = 960, 540
        scale = min(max_w / self.orig_w, max_h / self.orig_h, 1.0)
        self.disp_w = max(320, int(self.orig_w * scale))
        self.disp_h = max(240, int(self.orig_h * scale))
        self.scale_x = self.orig_w / self.disp_w
        self.scale_y = self.orig_h / self.disp_h

        # Original coordinates storage
        self.points = []
        if initial_polygon:
            for pt in initial_polygon:
                self.points.append((int(pt[0]), int(pt[1])))

        self.current_mouse = None

        # Prepare base image
        resized_frame = cv2.resize(frame, (self.disp_w, self.disp_h))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        self.base_pil = Image.fromarray(rgb_frame)
        self.photo_img = ImageTk.PhotoImage(image=self.base_pil)

        self._build_ui()
        self._bind_events()
        self._redraw_canvas()

        # Center on parent
        self.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        dw = self.winfo_width()
        dh = self.winfo_height()
        x = max(20, px + (pw - dw) // 2)
        y = max(20, py + (ph - dh) // 2)
        self.geometry(f"+{x}+{y}")

    def _build_ui(self):
        # Header instructions
        hdr = tk.Frame(self, bg="#1E293B", padx=12, pady=8)
        hdr.pack(fill="x")

        tk.Label(
            hdr,
            text="RESTRICTED ZONE POLYGON EDITOR",
            font=("Segoe UI", 11, "bold"),
            bg="#1E293B",
            fg="#F8FAFC"
        ).pack(anchor="w")

        tk.Label(
            hdr,
            text="Click multiple points to outline the hazardous area. Minimum 3 points required. Coordinates are accurately mapped to full resolution.",
            font=("Segoe UI", 9),
            bg="#1E293B",
            fg="#94A3B8"
        ).pack(anchor="w")

        # Canvas viewport
        canvas_container = tk.Frame(self, bg="#0F172A", padx=10, pady=8)
        canvas_container.pack()

        self.canvas = tk.Canvas(
            canvas_container,
            width=self.disp_w,
            height=self.disp_h,
            bg="#000000",
            highlightthickness=1,
            highlightbackground="#334155",
            cursor="crosshair"
        )
        self.canvas.pack()

        # Control & action toolbar
        bar = tk.Frame(self, bg="#1E293B", padx=12, pady=10)
        bar.pack(fill="x")

        self.status_lbl = tk.Label(
            bar,
            text="Points: 0 (Add at least 3 points)",
            font=("Segoe UI", 9, "bold"),
            bg="#1E293B",
            fg="#F59E0B"
        )
        self.status_lbl.pack(side="left", padx=(0, 15))

        btn_cancel = tk.Button(
            bar, text="✖ Cancel", font=("Segoe UI", 9),
            bg="#334155", fg="#F8FAFC", activebackground="#475569", activeforeground="#FFFFFF",
            bd=0, padx=12, pady=5, cursor="hand2", command=self._on_cancel
        )
        btn_cancel.pack(side="right", padx=(5, 0))

        btn_reset = tk.Button(
            bar, text="🔄 Reset", font=("Segoe UI", 9),
            bg="#334155", fg="#F8FAFC", activebackground="#475569", activeforeground="#FFFFFF",
            bd=0, padx=12, pady=5, cursor="hand2", command=self._on_reset
        )
        btn_reset.pack(side="right", padx=5)

        btn_undo = tk.Button(
            bar, text="↩ Undo Point", font=("Segoe UI", 9),
            bg="#334155", fg="#F8FAFC", activebackground="#475569", activeforeground="#FFFFFF",
            bd=0, padx=12, pady=5, cursor="hand2", command=self._on_undo
        )
        btn_undo.pack(side="right", padx=5)

        self.btn_confirm = tk.Button(
            bar, text="✓ Confirm & Save Polygon", font=("Segoe UI", 9, "bold"),
            bg="#0284C7", fg="#FFFFFF", activebackground="#0369A1", activeforeground="#FFFFFF",
            bd=0, padx=16, pady=5, cursor="hand2", command=self._on_confirm
        )
        self.btn_confirm.pack(side="right", padx=5)

    def _bind_events(self):
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.bind("<Return>", lambda e: self._on_confirm())
        self.bind("<Escape>", lambda e: self._on_cancel())
        self.bind("<BackSpace>", lambda e: self._on_undo())
        self.bind("<u>", lambda e: self._on_undo())
        self.bind("<r>", lambda e: self._on_reset())

    def _on_canvas_click(self, event):
        clamped_x = max(0, min(self.disp_w, event.x))
        clamped_y = max(0, min(self.disp_h, event.y))
        orig_x = int(round(clamped_x * self.scale_x))
        orig_y = int(round(clamped_y * self.scale_y))
        orig_x = max(0, min(self.orig_w, orig_x))
        orig_y = max(0, min(self.orig_h, orig_y))

        self.points.append((orig_x, orig_y))
        self._redraw_canvas()

    def _on_canvas_motion(self, event):
        self.current_mouse = (event.x, event.y)
        self._redraw_canvas()

    def _on_undo(self):
        if self.points:
            self.points.pop()
            self._redraw_canvas()

    def _on_reset(self):
        self.points.clear()
        self._redraw_canvas()

    def _on_confirm(self):
        if len(self.points) < 3:
            messagebox.showwarning(
                "Incomplete Polygon",
                "A restricted area polygon requires at least 3 points.\nPlease click more points on the frame.",
                parent=self
            )
            return

        val_poly = main.validate_polygon(self.points, self.orig_w, self.orig_h)
        if val_poly and len(val_poly) >= 3:
            self.confirmed_polygon = val_poly
            self.destroy()
        else:
            messagebox.showerror(
                "Invalid Polygon",
                "The selected points do not form a valid non-degenerate polygon.\nPlease reset or adjust the vertices.",
                parent=self
            )

    def _on_cancel(self):
        self.confirmed_polygon = None
        self.destroy()

    def _redraw_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo_img, anchor="nw")

        disp_pts = [
            (int(round(pt[0] / self.scale_x)), int(round(pt[1] / self.scale_y)))
            for pt in self.points
        ]

        # Draw filled polygon if >= 3 points
        if len(disp_pts) >= 3:
            flat_pts = [coord for pt in disp_pts for coord in pt]
            self.canvas.create_polygon(
                flat_pts,
                fill="#8B5CF6",
                stipple="gray25",
                outline="#8B5CF6",
                width=2
            )

        # Draw boundary lines
        if len(disp_pts) > 1:
            for i in range(len(disp_pts) - 1):
                self.canvas.create_line(
                    disp_pts[i][0], disp_pts[i][1],
                    disp_pts[i+1][0], disp_pts[i+1][0],
                    fill="#00E5FF", width=2
                )
            if len(disp_pts) >= 3:
                self.canvas.create_line(
                    disp_pts[-1][0], disp_pts[-1][1],
                    disp_pts[0][0], disp_pts[0][1],
                    fill="#00E5FF", width=2
                )

        # Draw guideline to current mouse position
        if disp_pts and self.current_mouse:
            self.canvas.create_line(
                disp_pts[-1][0], disp_pts[-1][1],
                self.current_mouse[0], self.current_mouse[1],
                fill="#F8FAFC", dash=(4, 4), width=1
            )

        # Draw vertex markers
        for idx, (px, py) in enumerate(disp_pts):
            r = 5
            self.canvas.create_oval(px - r, py - r, px + r, py + r, fill="#EF4444", outline="#FFFFFF", width=1.5)
            self.canvas.create_text(px + 10, py - 8, text=str(idx + 1), fill="#FFFFFF", font=("Segoe UI", 9, "bold"))

        # Update status banner
        pt_count = len(self.points)
        if pt_count >= 3:
            self.status_lbl.config(
                text=f"Points: {pt_count} (Valid Polygon — Ready to Confirm)",
                fg="#10B981"
            )
            self.btn_confirm.config(state="normal", bg="#0284C7")
        else:
            self.status_lbl.config(
                text=f"Points: {pt_count} (Add {3 - pt_count} more point{'s' if 3 - pt_count > 1 else ''})",
                fg="#F59E0B"
            )


class PPELauncherApp(tk.Tk):
    """Professional Industrial Control Room HMI for PPE Monitoring & Intrusion Detection."""

    def __init__(self):
        super().__init__()
        self.title("PPE Compliance & Restricted Zone Monitoring — Industrial HMI")
        self.geometry("1200x820")
        self.minsize(1050, 700)
        self.configure(bg="#0F172A")

        # Detection thread control
        self.detection_thread = None
        self.stop_event = threading.Event()
        self.is_running = False

        # Input sources & ROI state
        self.source_mode_var = tk.StringVar(value="video")  # "video" or "webcam"
        default_video = ""
        self.video_path_var = tk.StringVar(value=default_video)
        self.cam_choice = tk.StringVar(value="Camera 0 (Default)")
        self.video_roi = None
        self.webcam_roi = None
        self.save_video_var = tk.BooleanVar(value=True)

        # 10 Selectable PPE Classes
        self.selected_ppe_vars = {
            key: tk.BooleanVar(value=(key in DEFAULT_REQUIRED_PPE))
            for key, _ in AVAILABLE_PPE_OPTIONS
        }

        # Real-time state trackers
        self.current_session_dir = ""
        self.active_workers_cache = []
        self.latest_photo_img = None
        self._processed_violation_count = 0
        self.logged_violation_keys = set()
        self._last_intrusion_log_time = 0

        # Status variable
        self.status_var = tk.StringVar(value="Status: Ready. Model & system initialized.")

        # Build UI
        self._setup_styles()
        self._build_layout()

        # Handle clean window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initial operational log messages
        self.log_system_event("SYS", "PPE Safety Monitoring System HMI initialized.")
        self.log_system_event("SYS", f"Working directory: {os.getcwd()}")
        self.log_system_event("INFO", f"Default PPE compliance policy loaded ({len(DEFAULT_REQUIRED_PPE)} items: {sorted(list(DEFAULT_REQUIRED_PPE))}).")

        # Post-init checks
        self.after(100, self.initial_checks)

    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Notebook styling
        self.style.configure(
            "TNotebook",
            background="#1E293B",
            borderwidth=0
        )
        self.style.configure(
            "TNotebook.Tab",
            background="#334155",
            foreground="#94A3B8",
            font=("Segoe UI", 9, "bold"),
            padding=[12, 5]
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", "#0F172A")],
            foreground=[("selected", "#F8FAFC")]
        )

        # Treeview styling
        self.style.configure(
            "Treeview",
            background="#1E293B",
            foreground="#F8FAFC",
            fieldbackground="#1E293B",
            font=("Segoe UI", 9),
            rowheight=24,
            borderwidth=0
        )
        self.style.configure(
            "Treeview.Heading",
            background="#334155",
            foreground="#F8FAFC",
            font=("Segoe UI", 9, "bold"),
            borderwidth=1,
            relief="flat"
        )
        self.style.map(
            "Treeview.Heading",
            background=[("active", "#475569")]
        )

        # Action Buttons
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background="#0284C7", foreground="#FFFFFF", padding=6)
        self.style.map("Primary.TButton", background=[("active", "#0369A1")])

        self.style.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), background="#EF4444", foreground="#FFFFFF", padding=6)
        self.style.map("Danger.TButton", background=[("active", "#DC2626")])

        self.style.configure("Secondary.TButton", font=("Segoe UI", 9), background="#334155", foreground="#F8FAFC", padding=4)
        self.style.map("Secondary.TButton", background=[("active", "#475569")])

        self.style.configure("Accent.TButton", font=("Segoe UI", 9, "bold"), background="#8B5CF6", foreground="#FFFFFF", padding=4)
        self.style.map("Accent.TButton", background=[("active", "#7C3AED")])

    def _build_layout(self):
        # Top Header Bar
        header = tk.Frame(self, bg="#1E293B", padx=16, pady=10)
        header.pack(fill="x")

        title_frame = tk.Frame(header, bg="#1E293B")
        title_frame.pack(side="left")

        tk.Label(
            title_frame,
            text="🛡  PPE COMPLIANCE & RESTRICTED ZONE MONITORING",
            font=("Segoe UI", 14, "bold"),
            bg="#1E293B",
            fg="#F8FAFC"
        ).pack(anchor="w")

        tk.Label(
            title_frame,
            text="Industrial Vision HMI  |  YOLOv8 SH17  •  DeepSORT  •  Polygon Intrusion",
            font=("Segoe UI", 8),
            bg="#1E293B",
            fg="#94A3B8"
        ).pack(anchor="w")

        # Top Right Performance Indicators
        top_stats = tk.Frame(header, bg="#1E293B")
        top_stats.pack(side="right")

        self.sys_status_badge = tk.Label(
            top_stats,
            text="● SYSTEM IDLE",
            font=("Segoe UI", 9, "bold"),
            bg="#334155",
            fg="#94A3B8",
            padx=10,
            pady=4
        )
        self.sys_status_badge.pack(side="left", padx=4)

        self.fps_badge = tk.Label(
            top_stats,
            text="FPS: --",
            font=("Segoe UI", 9, "bold"),
            bg="#334155",
            fg="#38BDF8",
            padx=10,
            pady=4
        )
        self.fps_badge.pack(side="left", padx=4)

        self.frame_badge = tk.Label(
            top_stats,
            text="Frame: 0",
            font=("Segoe UI", 9, "bold"),
            bg="#334155",
            fg="#F8FAFC",
            padx=10,
            pady=4
        )
        self.frame_badge.pack(side="left", padx=4)

        # Main Body (Left Controls + Right Monitoring Viewport)
        body = tk.Frame(self, bg="#0F172A", padx=12, pady=10)
        body.pack(fill="both", expand=True)

        # Left Column: Configuration & Controls (Width: 380px)
        left_col = tk.Frame(body, bg="#0F172A", width=380)
        left_col.pack(side="left", fill="y", padx=(0, 10))
        left_col.pack_propagate(False)

        # 1. Source Selection Card
        self._build_source_card(left_col)

        # 2. Required PPE Policy Card (10 Classes)
        self._build_ppe_policy_card(left_col)

        # 3. Restricted Zone Configuration Card
        self._build_roi_card(left_col)

        # 4. Action Controls Card
        self._build_controls_card(left_col)

        # Right Column: Live Viewport & Telemetry
        right_col = tk.Frame(body, bg="#0F172A")
        right_col.pack(side="right", fill="both", expand=True)

        # Live Video Canvas Container
        self._build_live_monitor_card(right_col)

        # Telemetry & Event Tabs Container
        self._build_tabs_card(right_col)

        # Bottom Global Status Bar
        status_bar_frame = tk.Frame(self, bg="#1E293B", padx=14, pady=4)
        status_bar_frame.pack(fill="x", side="bottom")

        self.status_lbl = tk.Label(
            status_bar_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 8),
            bg="#1E293B",
            fg="#94A3B8",
            anchor="w"
        )
        self.status_lbl.pack(side="left", fill="x", expand=True)

        self.session_lbl = tk.Label(
            status_bar_frame,
            text="Session: [None]",
            font=("Segoe UI", 8),
            bg="#1E293B",
            fg="#64748B",
            anchor="e"
        )
        self.session_lbl.pack(side="right")

    def _build_source_card(self, parent):
        card = tk.LabelFrame(
            parent,
            text=" 1. Input Source ",
            font=("Segoe UI", 9, "bold"),
            bg="#1E293B",
            fg="#F8FAFC",
            padx=10,
            pady=8,
            bd=1,
            relief="solid"
        )
        card.pack(fill="x", pady=(0, 8))

        mode_row = tk.Frame(card, bg="#1E293B")
        mode_row.pack(fill="x", pady=(0, 6))

        rb_video = tk.Radiobutton(
            mode_row,
            text="Video File",
            value="video",
            variable=self.source_mode_var,
            font=("Segoe UI", 9, "bold"),
            bg="#1E293B",
            fg="#F8FAFC",
            selectcolor="#0F172A",
            activebackground="#1E293B",
            activeforeground="#FFFFFF",
            command=self._on_source_mode_change
        )
        rb_video.pack(side="left", padx=(0, 15))

        rb_cam = tk.Radiobutton(
            mode_row,
            text="Live Webcam",
            value="webcam",
            variable=self.source_mode_var,
            font=("Segoe UI", 9, "bold"),
            bg="#1E293B",
            fg="#F8FAFC",
            selectcolor="#0F172A",
            activebackground="#1E293B",
            activeforeground="#FFFFFF",
            command=self._on_source_mode_change
        )
        rb_cam.pack(side="left")

        # Video File Controls Container
        self.file_controls_frame = tk.Frame(card, bg="#1E293B")
        self.file_controls_frame.pack(fill="x")

        file_row = tk.Frame(self.file_controls_frame, bg="#1E293B")
        file_row.pack(fill="x", pady=(2, 2))

        self.file_entry = tk.Entry(
            file_row,
            textvariable=self.video_path_var,
            font=("Segoe UI", 8),
            bg="#0F172A",
            fg="#F8FAFC",
            insertbackground="#FFFFFF",
            bd=1,
            relief="solid"
        )
        self.file_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        btn_browse = tk.Button(
            file_row,
            text="📂 Browse...",
            font=("Segoe UI", 8, "bold"),
            bg="#334155",
            fg="#F8FAFC",
            activebackground="#475569",
            activeforeground="#FFFFFF",
            bd=0,
            padx=8,
            pady=2,
            cursor="hand2",
            command=self.browse_video_file
        )
        btn_browse.pack(side="right")

        # Webcam Controls Container
        self.cam_controls_frame = tk.Frame(card, bg="#1E293B")

        cam_row = tk.Frame(self.cam_controls_frame, bg="#1E293B")
        cam_row.pack(fill="x", pady=(2, 2))

        available_cams = probe_cameras()
        self.cam_options = [f"Camera {i}" + (" (Default)" if i == 0 else "") for i in available_cams]
        if not self.cam_choice.get() in self.cam_options:
            self.cam_choice.set(self.cam_options[0])

        self.cam_dropdown = ttk.Combobox(
            cam_row,
            textvariable=self.cam_choice,
            values=self.cam_options,
            state="readonly",
            font=("Segoe UI", 8)
        )
        self.cam_dropdown.pack(side="left", fill="x", expand=True, padx=(0, 6))

        btn_refresh_cam = tk.Button(
            cam_row,
            text="🔄 Probe",
            font=("Segoe UI", 8),
            bg="#334155",
            fg="#F8FAFC",
            activebackground="#475569",
            activeforeground="#FFFFFF",
            bd=0,
            padx=8,
            pady=2,
            cursor="hand2",
            command=self._refresh_cameras
        )
        btn_refresh_cam.pack(side="right")

        self._on_source_mode_change()

    def _on_source_mode_change(self):
        mode = self.source_mode_var.get()
        if mode == "video":
            self.cam_controls_frame.pack_forget()
            self.file_controls_frame.pack(fill="x")
            vpath = self.video_path_var.get()
            self.log_system_event("INFO", f"Input mode set to Video File: {os.path.basename(vpath) if vpath else 'None'}")
        else:
            self.file_controls_frame.pack_forget()
            self.cam_controls_frame.pack(fill="x")
            cam = self.cam_choice.get()
            self.log_system_event("INFO", f"Input mode set to Live Webcam: {cam}")
        self._update_roi_badge()

    def _refresh_cameras(self):
        cams = probe_cameras()
        self.cam_options = [f"Camera {i}" + (" (Default)" if i == 0 else "") for i in cams]
        self.cam_dropdown['values'] = self.cam_options
        if self.cam_options:
            self.cam_choice.set(self.cam_options[0])
        self.status_var.set(f"Camera probe complete: Found {len(cams)} video device(s).")
        self.log_system_event("INFO", f"Camera probe complete: Found {len(cams)} video device(s).")

    def _build_ppe_policy_card(self, parent):
        card = tk.LabelFrame(
            parent,
            text=" 2. Required PPE Compliance Policy ",
            font=("Segoe UI", 9, "bold"),
            bg="#1E293B",
            fg="#F8FAFC",
            padx=10,
            pady=6,
            bd=1,
            relief="solid"
        )
        card.pack(fill="x", pady=(0, 8))

        grid_frame = tk.Frame(card, bg="#1E293B")
        grid_frame.pack(fill="x", pady=(2, 4))

        for idx, (key, label) in enumerate(AVAILABLE_PPE_OPTIONS):
            row = idx % 5
            col = idx // 5
            cb = tk.Checkbutton(
                grid_frame,
                text=label,
                variable=self.selected_ppe_vars[key],
                font=("Segoe UI", 8),
                bg="#1E293B",
                fg="#F8FAFC",
                selectcolor="#0F172A",
                activebackground="#1E293B",
                activeforeground="#FFFFFF",
                anchor="w",
                command=lambda k=key: self._on_ppe_checkbox_toggle(k)
            )
            cb.grid(row=row, column=col, sticky="w", padx=4, pady=1)

        btn_row = tk.Frame(card, bg="#1E293B")
        btn_row.pack(fill="x", pady=(4, 2))

        btn_all = tk.Button(
            btn_row, text="Select All", font=("Segoe UI", 8),
            bg="#334155", fg="#F8FAFC", activebackground="#475569", activeforeground="#FFFFFF",
            bd=0, padx=6, pady=2, cursor="hand2", command=self.select_all_ppe
        )
        btn_all.pack(side="left", padx=(0, 4))

        btn_none = tk.Button(
            btn_row, text="Clear All", font=("Segoe UI", 8),
            bg="#334155", fg="#F8FAFC", activebackground="#475569", activeforeground="#FFFFFF",
            bd=0, padx=6, pady=2, cursor="hand2", command=self.clear_all_ppe
        )
        btn_none.pack(side="left", padx=(0, 4))

        btn_def = tk.Button(
            btn_row, text="Reset Default", font=("Segoe UI", 8),
            bg="#334155", fg="#F8FAFC", activebackground="#475569", activeforeground="#FFFFFF",
            bd=0, padx=6, pady=2, cursor="hand2", command=self.reset_default_ppe
        )
        btn_def.pack(side="left")

    def _build_roi_card(self, parent):
        card = tk.LabelFrame(
            parent,
            text=" 3. Restricted Zone (Polygon ROI) ",
            font=("Segoe UI", 9, "bold"),
            bg="#1E293B",
            fg="#F8FAFC",
            padx=10,
            pady=8,
            bd=1,
            relief="solid"
        )
        card.pack(fill="x", pady=(0, 8))

        self.roi_status_lbl = tk.Label(
            card,
            text="Restricted Zone: Disabled (Full Scene / PPE Only)",
            font=("Segoe UI", 8, "bold"),
            bg="#1E293B",
            fg="#94A3B8",
            anchor="w"
        )
        self.roi_status_lbl.pack(fill="x", pady=(0, 6))

        btn_row = tk.Frame(card, bg="#1E293B")
        btn_row.pack(fill="x")

        btn_define_roi = tk.Button(
            btn_row,
            text="📐 Define Polygon...",
            font=("Segoe UI", 8, "bold"),
            bg="#8B5CF6",
            fg="#FFFFFF",
            activebackground="#7C3AED",
            activeforeground="#FFFFFF",
            bd=0,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self.open_polygon_editor
        )
        btn_define_roi.pack(side="left", padx=(0, 6))

        btn_clear_roi = tk.Button(
            btn_row,
            text="❌ Clear Zone",
            font=("Segoe UI", 8),
            bg="#334155",
            fg="#F8FAFC",
            activebackground="#475569",
            activeforeground="#FFFFFF",
            bd=0,
            padx=8,
            pady=4,
            cursor="hand2",
            command=self.clear_current_roi
        )
        btn_clear_roi.pack(side="left")

    def _update_roi_badge(self):
        if not hasattr(self, 'roi_status_lbl'):
            return
        active_roi = self.get_active_roi()
        if active_roi and len(active_roi) >= 3:
            self.roi_status_lbl.config(
                text=f"Restricted Zone: Active ({len(active_roi)} points confirmed)",
                fg="#10B981"
            )
        else:
            self.roi_status_lbl.config(
                text="Restricted Zone: Disabled (Full Scene / PPE Only)",
                fg="#94A3B8"
            )

    def _build_controls_card(self, parent):
        card = tk.LabelFrame(
            parent,
            text=" 4. Execution Controls ",
            font=("Segoe UI", 9, "bold"),
            bg="#1E293B",
            fg="#F8FAFC",
            padx=10,
            pady=10,
            bd=1,
            relief="solid"
        )
        card.pack(fill="x", pady=(0, 4))

        cb_save = tk.Checkbutton(
            card,
            text="Record & save annotated video output",
            variable=self.save_video_var,
            font=("Segoe UI", 8),
            bg="#1E293B",
            fg="#94A3B8",
            selectcolor="#0F172A",
            activebackground="#1E293B",
            activeforeground="#FFFFFF"
        )
        cb_save.pack(anchor="w", pady=(0, 8))

        btn_row = tk.Frame(card, bg="#1E293B")
        btn_row.pack(fill="x")

        self.btn_start = tk.Button(
            btn_row,
            text="▶  START MONITORING",
            font=("Segoe UI", 10, "bold"),
            bg="#0284C7",
            fg="#FFFFFF",
            activebackground="#0369A1",
            activeforeground="#FFFFFF",
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2",
            command=self.start_monitoring_action
        )
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_stop = tk.Button(
            btn_row,
            text="⏹  STOP",
            font=("Segoe UI", 10, "bold"),
            bg="#334155",
            fg="#64748B",
            activebackground="#DC2626",
            activeforeground="#FFFFFF",
            state="disabled",
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2",
            command=self.stop_detection
        )
        self.btn_stop.pack(side="right", fill="x", expand=True, padx=(4, 0))

    def _build_live_monitor_card(self, parent):
        frame = tk.Frame(parent, bg="#1E293B", bd=1, relief="solid")
        frame.pack(fill="both", expand=True, pady=(0, 8))

        # Top Bar of Live Viewport
        vp_hdr = tk.Frame(frame, bg="#1E293B", padx=10, pady=6)
        vp_hdr.pack(fill="x")

        tk.Label(
            vp_hdr,
            text="LIVE VISION MONITOR",
            font=("Segoe UI", 9, "bold"),
            bg="#1E293B",
            fg="#F8FAFC"
        ).pack(side="left")

        # Intrusion Banner
        self.intrusion_banner = tk.Label(
            vp_hdr,
            text="● ZONE SECURE",
            font=("Segoe UI", 8, "bold"),
            bg="#334155",
            fg="#10B981",
            padx=8,
            pady=2
        )
        self.intrusion_banner.pack(side="right", padx=(6, 0))

        # Worker Count Badges
        self.workers_badge = tk.Label(
            vp_hdr,
            text="Workers: 0",
            font=("Segoe UI", 8, "bold"),
            bg="#334155",
            fg="#F8FAFC",
            padx=6,
            pady=2
        )
        self.workers_badge.pack(side="right", padx=2)

        self.safe_badge = tk.Label(
            vp_hdr,
            text="Safe: 0",
            font=("Segoe UI", 8, "bold"),
            bg="#334155",
            fg="#10B981",
            padx=6,
            pady=2
        )
        self.safe_badge.pack(side="right", padx=2)

        self.partial_badge = tk.Label(
            vp_hdr,
            text="Partial: 0",
            font=("Segoe UI", 8, "bold"),
            bg="#334155",
            fg="#F59E0B",
            padx=6,
            pady=2
        )
        self.partial_badge.pack(side="right", padx=2)

        self.unsafe_badge = tk.Label(
            vp_hdr,
            text="Unsafe: 0",
            font=("Segoe UI", 8, "bold"),
            bg="#334155",
            fg="#EF4444",
            padx=6,
            pady=2
        )
        self.unsafe_badge.pack(side="right", padx=2)

        # Video Canvas
        self.video_canvas = tk.Canvas(
            frame,
            bg="#090D16",
            highlightthickness=0
        )
        self.video_canvas.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Initial Canvas Standby Graphic
        self.video_canvas.bind("<Configure>", lambda e: self._draw_standby_screen())
        self._draw_standby_screen()

    def _draw_standby_screen(self):
        if self.is_running and self.latest_photo_img:
            return
        self.video_canvas.delete("all")
        cw = max(200, self.video_canvas.winfo_width())
        ch = max(150, self.video_canvas.winfo_height())
        self.video_canvas.create_text(
            cw // 2, ch // 2 - 12,
            text="📹  NO ACTIVE VIDEO STREAM",
            fill="#475569",
            font=("Segoe UI", 12, "bold")
        )
        self.video_canvas.create_text(
            cw // 2, ch // 2 + 14,
            text="Configure source and click 'Start Monitoring' to begin live detection",
            fill="#334155",
            font=("Segoe UI", 9)
        )

    def _build_tabs_card(self, parent):
        tab_container = tk.Frame(parent, bg="#0F172A", height=230)
        tab_container.pack(fill="x")
        tab_container.pack_propagate(False)

        self.notebook = ttk.Notebook(tab_container)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Active Workers Grid
        tab_workers = tk.Frame(self.notebook, bg="#1E293B")
        self.notebook.add(tab_workers, text="  👥 Active Workers  ")

        cols_workers = ("id", "status", "missing", "detected", "bbox")
        self.tree_workers = ttk.Treeview(
            tab_workers,
            columns=cols_workers,
            show="headings",
            selectmode="browse"
        )
        self.tree_workers.heading("id", text="Track ID")
        self.tree_workers.heading("status", text="Compliance Status")
        self.tree_workers.heading("missing", text="Missing Required PPE")
        self.tree_workers.heading("detected", text="Detected PPE")
        self.tree_workers.heading("bbox", text="Bounding Box [L,T,R,B]")

        self.tree_workers.column("id", width=80, anchor="center")
        self.tree_workers.column("status", width=120, anchor="center")
        self.tree_workers.column("missing", width=180, anchor="w")
        self.tree_workers.column("detected", width=220, anchor="w")
        self.tree_workers.column("bbox", width=150, anchor="center")

        scroll_w = ttk.Scrollbar(tab_workers, orient="vertical", command=self.tree_workers.yview)
        self.tree_workers.configure(yscrollcommand=scroll_w.set)
        self.tree_workers.pack(side="left", fill="both", expand=True)
        scroll_w.pack(side="right", fill="y")

        # Color tags for Treeview
        self.tree_workers.tag_configure("SAFE", foreground="#10B981")
        self.tree_workers.tag_configure("PARTIAL", foreground="#F59E0B")
        self.tree_workers.tag_configure("UNSAFE", foreground="#EF4444")

        # Tab 2: PPE Violation Events Log
        tab_events = tk.Frame(self.notebook, bg="#1E293B")
        self.notebook.add(tab_events, text="  ⚠️ PPE Violation Log  ")

        cols_events = ("time", "person", "status", "missing", "event", "snapshot")
        self.tree_events = ttk.Treeview(
            tab_events,
            columns=cols_events,
            show="headings",
            selectmode="browse"
        )
        self.tree_events.heading("time", text="Time")
        self.tree_events.heading("person", text="Person")
        self.tree_events.heading("status", text="Status")
        self.tree_events.heading("missing", text="Missing PPE")
        self.tree_events.heading("event", text="Event Trigger")
        self.tree_events.heading("snapshot", text="Evidence Snapshot")

        self.tree_events.column("time", width=90, anchor="center")
        self.tree_events.column("person", width=80, anchor="center")
        self.tree_events.column("status", width=110, anchor="center")
        self.tree_events.column("missing", width=160, anchor="w")
        self.tree_events.column("event", width=180, anchor="w")
        self.tree_events.column("snapshot", width=200, anchor="w")

        scroll_e = ttk.Scrollbar(tab_events, orient="vertical", command=self.tree_events.yview)
        self.tree_events.configure(yscrollcommand=scroll_e.set)
        self.tree_events.pack(side="left", fill="both", expand=True)
        scroll_e.pack(side="right", fill="y")

        # Tab 3: System Console Log
        tab_console = tk.Frame(self.notebook, bg="#1E293B")
        self.notebook.add(tab_console, text="  💻 System Output  ")

        self.console_txt = tk.Text(
            tab_console,
            bg="#090D16",
            fg="#94A3B8",
            insertbackground="#FFFFFF",
            font=("Consolas", 8),
            bd=0,
            wrap="word",
            state="disabled"
        )
        self.console_txt.tag_configure("TIME", foreground="#64748B", font=("Consolas", 8))
        self.console_txt.tag_configure("SYS", foreground="#38BDF8", font=("Consolas", 8, "bold"))
        self.console_txt.tag_configure("OK", foreground="#10B981", font=("Consolas", 8, "bold"))
        self.console_txt.tag_configure("INFO", foreground="#E2E8F0", font=("Consolas", 8))
        self.console_txt.tag_configure("WARN", foreground="#F59E0B", font=("Consolas", 8, "bold"))
        self.console_txt.tag_configure("ALARM", foreground="#EF4444", font=("Consolas", 8, "bold"))
        self.console_txt.tag_configure("MSG", foreground="#CBD5E1", font=("Consolas", 8))

        scroll_c = ttk.Scrollbar(tab_console, orient="vertical", command=self.console_txt.yview)
        self.console_txt.configure(yscrollcommand=scroll_c.set)
        self.console_txt.pack(side="left", fill="both", expand=True)
        scroll_c.pack(side="right", fill="y")

    def log_system_event(self, level: str, message: str):
        """Append a timestamped, color-coded operational log entry to the System Output console."""
        if threading.current_thread() is not threading.main_thread():
            try:
                self.after(0, lambda: self.log_system_event(level, message))
            except Exception:
                pass
            return

        if not hasattr(self, 'console_txt') or self.console_txt is None:
            return

        try:
            timestamp_str = time.strftime("[%H:%M:%S] ")
            level_tag = level.upper()
            prefix = f"[{level_tag:<5}] "

            self.console_txt.configure(state="normal")
            self.console_txt.insert("end", timestamp_str, "TIME")
            tag = level_tag if level_tag in ("SYS", "OK", "INFO", "WARN", "ALARM") else "INFO"
            self.console_txt.insert("end", prefix, tag)
            self.console_txt.insert("end", f"{message}\n", "MSG")
            self.console_txt.configure(state="disabled")
            self.console_txt.see("end")
        except Exception:
            pass

    def _on_ppe_checkbox_toggle(self, ppe_key):
        """Handle individual PPE checkbox toggles and log policy changes."""
        active = self.get_selected_required_ppe()
        state = "REQUIRED" if self.selected_ppe_vars[ppe_key].get() else "OPTIONAL"
        self.log_system_event("INFO", f"PPE Policy updated: '{ppe_key}' set to {state} ({len(active)} active classes).")

    def on_close(self):
        """Cleanly handle window close event: stop active detection and exit."""
        if self.is_running:
            self.stop_event.set()
        self.destroy()

    # --- System & Dependency Checks ---
    def initial_checks(self):
        missing = check_dependencies()
        if missing:
            self.log_system_event("WARN", f"Missing dependencies detected: {', '.join(missing)}")
            ans = messagebox.askyesno(
                "Missing Dependencies",
                f"The following packages are missing:\n\n{', '.join(missing)}\n\nInstall them automatically?",
                parent=self
            )
            if ans:
                self.status_var.set("Installing missing dependencies...")
                self.log_system_event("INFO", "Installing missing dependencies via pip...")
                self.update()
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                    self.log_system_event("OK", "Dependencies installed successfully.")
                    messagebox.showinfo("Success", "Dependencies installed successfully!", parent=self)
                except Exception as e:
                    self.log_system_event("ALARM", f"Failed to install dependencies: {e}")
                    messagebox.showerror("Error", f"Failed to install dependencies: {e}", parent=self)
                    return
        else:
            self.log_system_event("OK", "Core runtime dependencies verified.")

        ok, msg = check_model_weights()
        if not ok:
            messagebox.showerror("Model Error", msg, parent=self)
            self.status_var.set("Status: Error — Model missing or invalid.")
            self.log_system_event("ALARM", f"Model verification error: {msg}")
        else:
            self.status_var.set("Status: Ready. Model verified (17 SH17 classes).")
            self.log_system_event("OK", "AI Vision Model verified: models/yolo8m.pt (17 SH17 classes confirmed).")

    # --- Source & Video Browser Methods ---
    def browse_video_file(self):
        """Native Windows file picker supporting arbitrary video files from anywhere on the filesystem."""
        selected = filedialog.askopenfilename(
            parent=self,
            title="Select Video File for PPE Monitoring",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.flv"),
                ("MP4 Videos (*.mp4)", "*.mp4"),
                ("AVI Videos (*.avi)", "*.avi"),
                ("QuickTime Videos (*.mov)", "*.mov"),
                ("Matroska Videos (*.mkv)", "*.mkv"),
                ("All Files (*.*)", "*.*")
            ]
        )
        if selected:
            norm_path = os.path.normpath(selected)
            self.video_path_var.set(norm_path)
            self.clear_video_roi()
            self.status_var.set(f"Loaded video: {os.path.basename(norm_path)}")
            self.log_system_event("INFO", f"Input video selected: {os.path.basename(norm_path)} ({norm_path})")

    def get_selected_source(self):
        mode = self.source_mode_var.get()
        if mode == "webcam":
            cam_str = self.cam_choice.get()
            try:
                return int(cam_str.split()[1])
            except Exception:
                return 0
        else:
            return self.video_path_var.get().strip()

    # --- PPE Checklist Public Methods (Maintains 100% test compatibility) ---
    def get_selected_required_ppe(self):
        """Return set of selected required PPE class names."""
        return {key for key, var in self.selected_ppe_vars.items() if var.get()}

    def select_all_ppe(self):
        for var in self.selected_ppe_vars.values():
            var.set(True)
        self.log_system_event("INFO", "PPE Policy: Selected ALL 10 required PPE classes.")

    def clear_all_ppe(self):
        for var in self.selected_ppe_vars.values():
            var.set(False)
        self.log_system_event("WARN", "PPE Policy: CLEARED all required PPE classes (0 active).")

    def reset_default_ppe(self):
        for key, var in self.selected_ppe_vars.items():
            var.set(key in DEFAULT_REQUIRED_PPE)
        self.log_system_event("INFO", f"PPE Policy: Reset to default 5 classes: {sorted(list(DEFAULT_REQUIRED_PPE))}")

    # --- Polygon ROI Public Methods (Maintains 100% test compatibility) ---
    def get_active_roi(self):
        mode = self.source_mode_var.get()
        return self.webcam_roi if mode == "webcam" else self.video_roi

    def clear_webcam_roi(self):
        self.webcam_roi = None
        self._update_roi_badge()
        self.status_var.set("Webcam restricted zone cleared.")
        self.log_system_event("WARN", "Restricted zone (Webcam ROI) cleared — running in Full Scene mode.")

    def clear_video_roi(self):
        self.video_roi = None
        self._update_roi_badge()
        self.status_var.set("Video restricted zone cleared.")
        self.log_system_event("WARN", "Restricted zone (Video ROI) cleared — running in Full Scene mode.")

    def clear_current_roi(self):
        mode = self.source_mode_var.get()
        if mode == "webcam":
            self.clear_webcam_roi()
        else:
            self.clear_video_roi()

    def open_polygon_editor(self):
        """Extract first frame from selected source and launch interactive polygon dialog."""
        source = self.get_selected_source()
        if not source and source != 0:
            messagebox.showwarning("No Source", "Please select a valid video file or camera first.", parent=self)
            return

        if isinstance(source, str) and not os.path.exists(source):
            messagebox.showerror("File Not Found", f"Cannot find video file:\n{source}", parent=self)
            return

        self.status_var.set("Extracting reference frame for polygon editor...")
        self.update()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showerror("Source Error", f"Unable to open video source:\n{source}", parent=self)
            return

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            messagebox.showerror("Frame Error", f"Failed to read first frame from:\n{source}", parent=self)
            return

        current_roi = self.get_active_roi()
        dialog = PolygonDrawingDialog(self, frame, initial_polygon=current_roi)
        self.wait_window(dialog)

        if dialog.confirmed_polygon is not None:
            if self.source_mode_var.get() == "webcam":
                self.webcam_roi = dialog.confirmed_polygon
            else:
                self.video_roi = dialog.confirmed_polygon
            self._update_roi_badge()
            self.status_var.set(f"Polygon confirmed ({len(dialog.confirmed_polygon)} vertices).")
            self.log_system_event("OK", f"Restricted zone configured: {len(dialog.confirmed_polygon)} vertices confirmed.")
        else:
            self.status_var.set("Polygon editing cancelled.")
            self.log_system_event("INFO", "Polygon editing cancelled by operator.")

    # --- Screen Switching Stubs (Maintains 100% test compatibility with test_app_gui.py) ---
    def show_webcam_screen(self):
        self.source_mode_var.set("webcam")
        self._on_source_mode_change()

    def show_video_screen(self):
        self.source_mode_var.set("video")
        self._on_source_mode_change()

    def show_main_screen(self):
        self.source_mode_var.set("video")
        self._on_source_mode_change()

    # --- Execution & Monitoring Engine ---
    def start_monitoring_action(self):
        if self.is_running:
            return

        selected_ppe = self.get_selected_required_ppe()
        if not selected_ppe:
            messagebox.showwarning("No PPE Selected", "Please select at least one required PPE item in the checklist.", parent=self)
            return

        source = self.get_selected_source()
        if not source and source != 0:
            messagebox.showwarning("No Source", "Please select a video file or camera device.", parent=self)
            return

        if isinstance(source, str) and not os.path.exists(source):
            messagebox.showerror("File Not Found", f"Cannot find video file:\n{source}", parent=self)
            return

        active_roi = self.get_active_roi()
        self.start_detection_engine(
            source=source,
            roi=active_roi,
            no_roi=(active_roi is None),
            no_save_video=not self.save_video_var.get(),
            required_ppe=selected_ppe
        )

    def start_webcam_detection(self):
        self.source_mode_var.set("webcam")
        self.start_monitoring_action()

    def start_video_detection(self):
        self.source_mode_var.set("video")
        self.start_monitoring_action()

    def start_detection_engine(self, source, roi=None, no_roi=False, no_save_video=False, required_ppe=None):
        if self.is_running:
            return

        self.is_running = True
        self.stop_event.clear()
        self._processed_violation_count = 0
        self.logged_violation_keys.clear()

        # Update button states
        self.btn_start.config(state="disabled", bg="#334155")
        self.btn_stop.config(state="normal", bg="#EF4444", fg="#FFFFFF")
        self.sys_status_badge.config(text="● MONITORING ACTIVE", bg="#065F46", fg="#10B981")

        # Clear active workers table
        for item in self.tree_workers.get_children():
            self.tree_workers.delete(item)

        source_desc = f"Webcam {source}" if isinstance(source, int) else os.path.basename(str(source))
        self.status_var.set(f"Status: Monitoring {source_desc}...")

        self.log_system_event("SYS", f"Starting monitoring engine on source: {source_desc}")
        active_ppe_list = sorted(list(required_ppe)) if required_ppe else []
        self.log_system_event("INFO", f"Active PPE policy ({len(active_ppe_list)} items): {active_ppe_list}")
        if roi:
            self.log_system_event("INFO", f"Restricted zone active ({len(roi)} vertices configured).")
        else:
            self.log_system_event("INFO", "Restricted zone disabled (full scene monitoring).")

        def worker():
            try:
                results = main.run_detection(
                    source=source,
                    headless=False,
                    roi=roi,
                    no_roi=no_roi,
                    no_save_video=no_save_video,
                    stop_event=self.stop_event,
                    required_ppe=required_ppe,
                    frame_callback=self._on_frame_telemetry,
                    show_window=False
                )
                self.after(0, lambda: self.on_detection_complete(results))
            except Exception as e:
                self.after(0, lambda err=e: self.on_detection_error(err))

        self.detection_thread = threading.Thread(target=worker, daemon=True)
        self.detection_thread.start()

    def stop_detection(self):
        if self.is_running:
            self.status_var.set("Status: Stopping monitoring session...")
            self.stop_event.set()
            self.log_system_event("WARN", "Operator requested monitoring stop. Halting detection loop...")

    # --- Telemetry & Real-Time UI Dispatch (Conservative & Safe) ---
    def _on_frame_telemetry(self, telemetry):
        """Thread-safe telemetry callback invoked by main.py."""
        # Schedule UI update on main Tkinter event loop
        try:
            self.after(0, lambda: self._process_telemetry_safe(telemetry))
        except Exception:
            pass

    def _process_telemetry_safe(self, telemetry):
        """Safely process and update the HMI view without risking the detection loop."""
        try:
            # 1. Update session path
            sdir = telemetry.get('session_dir', '')
            if sdir and sdir != self.current_session_dir:
                self.current_session_dir = sdir
                self.session_lbl.config(text=f"Session: {os.path.basename(sdir)}")
                self.log_system_event("SYS", f"Session folder active: {os.path.basename(sdir)}")

            # 2. Performance counters
            fps_val = telemetry.get('fps', 0.0)
            fc_val = telemetry.get('frame_count', 0)
            self.fps_badge.config(text=f"FPS: {fps_val:.1f}")
            self.frame_badge.config(text=f"Frame: {fc_val}")

            # 3. Worker counts & intrusion status
            counts = telemetry.get('worker_counts', {})
            tot = counts.get('total', 0)
            safe = counts.get('safe', 0)
            part = counts.get('partial', 0)
            uns = counts.get('unsafe', 0)

            self.workers_badge.config(text=f"Workers: {tot}")
            self.safe_badge.config(text=f"Safe: {safe}")
            self.partial_badge.config(text=f"Partial: {part}")
            self.unsafe_badge.config(text=f"Unsafe: {uns}")

            # Intrusion state
            int_state = telemetry.get('intrusion_state', {})
            is_int = int_state.get('is_intrusion', False)
            px_count = int_state.get('motion_pixels', 0)
            has_poly = bool(int_state.get('polygon'))

            if is_int:
                self.intrusion_banner.config(
                    text=f"▲ ALARM: INTRUSION ({px_count} px)",
                    bg="#991B1B",
                    fg="#FFFFFF"
                )
                now = time.time()
                if now - getattr(self, '_last_intrusion_log_time', 0) >= 4.0:
                    self._last_intrusion_log_time = now
                    self.log_system_event("ALARM", f"INTRUSION DETECTED! Motion: {px_count} pixels in restricted zone.")
            elif has_poly:
                self.intrusion_banner.config(
                    text="● ZONE SECURE",
                    bg="#065F46",
                    fg="#10B981"
                )
            else:
                self.intrusion_banner.config(
                    text="○ ZONE DISABLED",
                    bg="#334155",
                    fg="#94A3B8"
                )

            # 4. Active Workers Table
            active_workers = telemetry.get('active_workers', [])
            for row in self.tree_workers.get_children():
                self.tree_workers.delete(row)

            for w in active_workers:
                tid = w.get('track_id', '?')
                st = w.get('status', 'UNKNOWN')
                miss = ", ".join(w.get('missing_ppe', [])) if w.get('missing_ppe') else "None"
                det = ", ".join(w.get('detected_ppe', [])) if w.get('detected_ppe') else "None"
                bbox = str(w.get('bbox', []))

                tag = "SAFE" if st == "SAFE" else ("PARTIAL" if st == "PARTIAL PPE" else "UNSAFE")
                self.tree_workers.insert(
                    "", "end",
                    values=(tid, st, miss, det, bbox),
                    tags=(tag,)
                )

            # 5. Check for new rows in violation_log.csv
            self._tail_violation_log()

            # 6. Embedded live video canvas display (Fail-safe)
            bgr_frame = telemetry.get('frame')
            if bgr_frame is not None and self.video_canvas is not None:
                cw = self.video_canvas.winfo_width()
                ch = self.video_canvas.winfo_height()
                if cw > 50 and ch > 50:
                    fh, fw = bgr_frame.shape[:2]
                    scale = min(cw / fw, ch / fh)
                    nw = max(1, int(fw * scale))
                    nh = max(1, int(fh * scale))
                    resized = cv2.resize(bgr_frame, (nw, nh))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    self.latest_photo_img = ImageTk.PhotoImage(image=pil_img)
                    self.video_canvas.delete("all")
                    ox = (cw - nw) // 2
                    oy = (ch - nh) // 2
                    self.video_canvas.create_image(ox, oy, image=self.latest_photo_img, anchor="nw")

        except Exception:
            # Conservative isolated fallback: Never crash main detection loop
            pass

    def _tail_violation_log(self):
        """Read and append any newly logged violation rows to the UI table."""
        if not self.current_session_dir:
            return
        log_csv = os.path.join(self.current_session_dir, "violation_log.csv")
        if not os.path.exists(log_csv):
            return

        try:
            with open(log_csv, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    date_val = row.get("Date", "")
                    time_val = row.get("Time", "")
                    person_val = row.get("Person", "")
                    status_val = row.get("Status", "")
                    missing_val = row.get("Missing PPE", "")
                    event_val = row.get("Event", "")
                    snap_val = row.get("Snapshot", "")

                    event_key = (date_val, time_val, person_val, missing_val, event_val)
                    if event_key not in self.logged_violation_keys:
                        self.logged_violation_keys.add(event_key)
                        self.tree_events.insert(
                            "", 0,  # Insert at top (newest first)
                            values=(time_val, person_val, status_val, missing_val, event_val, os.path.basename(snap_val))
                        )
                        self.log_system_event("WARN", f"PPE Event: Person {person_val} [{event_val}] ({status_val}: missing {missing_val})")
        except Exception:
            pass

    # --- Lifecycle Completion Handlers ---
    def on_detection_complete(self, results):
        self.is_running = False
        self.btn_start.config(state="normal", bg="#0284C7")
        self.btn_stop.config(state="disabled", bg="#334155", fg="#64748B")
        self.sys_status_badge.config(text="● SESSION COMPLETE", bg="#334155", fg="#F8FAFC")

        frames = results.get('frames_processed', 0) if isinstance(results, dict) else 0
        violations = results.get('violations_logged', 0) if isinstance(results, dict) else 0
        intrusions = results.get('intrusions_logged', 0) if isinstance(results, dict) else 0
        session_dir = results.get('session_dir', '') if isinstance(results, dict) else ''

        self.status_var.set(f"Status: Complete. Processed {frames} frames ({violations} PPE events, {intrusions} intrusions).")
        self._tail_violation_log()
        self.log_system_event("OK", f"Monitoring complete: {frames} frames processed, {violations} PPE events, {intrusions} intrusions.")

        messagebox.showinfo(
            "Monitoring Session Complete",
            f"Monitoring session completed successfully!\n\n"
            f"Session Directory:\n{session_dir}\n\n"
            f"Frames Processed: {frames}\n"
            f"PPE Violation Events Logged: {violations}\n"
            f"Restricted Zone Intrusions: {intrusions}",
            parent=self
        )

    def on_detection_error(self, err):
        self.is_running = False
        self.btn_start.config(state="normal", bg="#0284C7")
        self.btn_stop.config(state="disabled", bg="#334155", fg="#64748B")
        self.sys_status_badge.config(text="● ERROR OCCURRED", bg="#991B1B", fg="#FFFFFF")
        self.status_var.set(f"Status: Detection error: {err}")
        self.log_system_event("ALARM", f"Detection error: {err}")
        messagebox.showerror("Detection Error", f"An unexpected error occurred during detection:\n\n{err}", parent=self)


if __name__ == "__main__":
    app = PPELauncherApp()
    app.mainloop()
