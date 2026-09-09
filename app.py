from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_from_directory
)
 
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)
from rag.chunking import chunk_text_with_timestamps
from rag.vector_store import create_vector_store
from rag.search import (semantic_search, global_semantic_search)
from rag.qa import answer_question
from werkzeug.utils import secure_filename
from rag.vector_store import delete_video_vectors
from config import Config
from models import db, User, Video

from datetime import datetime
from whisper_service import transcribe_audio
from ollama_service import generate_summary, parse_summary


import os
import uuid
import subprocess


app = Flask(__name__)

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------

app.config.from_object(Config)


# ---------------------------------------
# DATABASE FOLDER
# ---------------------------------------

BASE_DIR = os.path.abspath(
    os.path.dirname(__file__)
)

DATABASE_FOLDER = os.path.join(
    BASE_DIR,
    "database"
)

os.makedirs(
    DATABASE_FOLDER,
    exist_ok=True
)


# ---------------------------------------
# UPLOAD FOLDERS
# ---------------------------------------

VIDEO_FOLDER = os.path.join(
    BASE_DIR,
    "uploads",
    "videos",
    
)

AUDIO_FOLDER = os.path.join(
    BASE_DIR,
    "uploads",
    "audio"
)

TRANSCRIPT_FOLDER = os.path.join(
    BASE_DIR,
    "uploads",
    "transcripts"
)

SUMMARY_FOLDER = os.path.join(
    BASE_DIR,
    "uploads",
    "summaries"
)


os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)


app.config["VIDEO_FOLDER"] = VIDEO_FOLDER
app.config["AUDIO_FOLDER"] = AUDIO_FOLDER
app.config["TRANSCRIPT_FOLDER"] = TRANSCRIPT_FOLDER
app.config["SUMMARY_FOLDER"] = SUMMARY_FOLDER


# ---------------------------------------
# ALLOWED VIDEO EXTENSIONS
# ---------------------------------------

ALLOWED_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm"
}


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ---------------------------------------
# DATABASE
# ---------------------------------------

db.init_app(app)

with app.app_context():
    db.create_all()


# ---------------------------------------
# HOME
# ---------------------------------------

@app.route("/")
def home():

    if "user_id" in session:
        return redirect(
            url_for("dashboard")
        )

    return redirect(
        url_for("login")
    )


# =======================================
# AUTHENTICATION
# =======================================

# ---------------------------------------
# REGISTER
# ---------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if not name or not email or not password:

            flash(
                "All fields are required.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:

            flash(
                "Email already registered.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password)
        )

        db.session.add(user)
        db.session.commit()

        flash(
            "Registration successful. Please login.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# ---------------------------------------
# LOGIN
# ---------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        user = User.query.filter_by(
            email=email
        ).first()

        if user and check_password_hash(
            user.password_hash,
            password
        ):

            session.clear()

            session["user_id"] = user.id
            session["user_name"] = user.name

            return redirect(
                url_for("dashboard")
            )

        flash(
            "Invalid email or password.",
            "error"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "login.html"
    )
# ---------------------------------------
# ASK
# ---------------------------------------


@app.route("/ask/<int:video_id>", methods=["POST"])
def ask_question(video_id):

        if "user_id" not in session:
            return redirect(url_for("login"))

        user_id = session["user_id"]

        video = Video.query.filter_by(
            id=video_id,
            user_id=user_id
        ).first_or_404()

        question = request.form.get(
            "question",
            ""
        ).strip()

        if not question:

            flash(
                "Please enter a question.",
                "warning"
            )

            return redirect(
                url_for(
                    "video_details",
                    video_id=video_id
                )
            )

        try:

            # -----------------------------
            # Semantic Search
            # -----------------------------

            results = semantic_search(
                question,
                video_id,
                top_k=5
            )

            # -----------------------------
            # Build Context
            # -----------------------------

            if results:

                context = "\n\n".join(
                    result["chunk"]
                    for result in results
                )

                # -----------------------------
                # Ollama Answer
                # -----------------------------

                answer = answer_question(
                    question,
                    context
                )

            else:

                answer = (
                    "I could not find the answer "
                    "in this video."
                )

            # -----------------------------
            # Read Transcript
            # -----------------------------

            transcript = ""

            if video.transcript_filename:

                transcript_path = os.path.join(
                    TRANSCRIPT_FOLDER,
                    video.transcript_filename
                )

                if os.path.exists(transcript_path):

                    with open(
                        transcript_path,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        transcript = f.read()

            # -----------------------------
            # Read Summary
            # -----------------------------

            summary = ""

            if video.summary_filename:

                summary_path = os.path.join(
                    SUMMARY_FOLDER,
                    video.summary_filename
                )

                if os.path.exists(summary_path):

                    with open(
                        summary_path,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        summary = f.read()

            # -----------------------------
            # Render Page
            # -----------------------------

            return render_template(
                "video_details.html",
                video=video,
                transcript=transcript,
                summary=summary,
                question=question,
                answer=answer,
                results=results
            )

        except Exception as e:

            print("RAG Error:", e)

            flash(
                f"AI search failed: {str(e)}",
                "danger"
            )

            return redirect(
                url_for(
                    "video_details",
                    video_id=video_id
                )
            )


            


# ---------------------------------------
# LOGOUT
# ---------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("login")
    )


# =======================================
# DASHBOARD
# =======================================

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    # Get user's videos
    videos = Video.query.filter_by(
        user_id=user_id
    ).order_by(
        Video.uploaded_at.desc()
    ).all()

    # Dashboard statistics
    total_videos = len(videos)

    ready_videos = sum(
        1 for video in videos
        if video.status == "Ready for Search"
    )

    processing_videos = sum(
        1 for video in videos
        if video.status == "Processing"
    )

    summarized_videos = sum(
        1 for video in videos
        if video.summary_filename
    )

    # Latest 5 videos
    recent_videos = videos[:5]

    return render_template(
        "dashboard.html",
        user=User.query.get(user_id),
        videos=videos,
        recent_videos=recent_videos,
        total_videos=total_videos,
        ready_videos=ready_videos,
        processing_videos=processing_videos,
        summarized_videos=summarized_videos
    )

# =======================================
# VIDEO UPLOAD
# =======================================


@app.route("/upload-video", methods=["GET", "POST"])
def upload_video():

    # Check if user is logged in
    if "user_id" not in session:
        return redirect(url_for("login"))

    # Get logged-in user's ID
    user_id = session["user_id"]

    # If page is opened normally
    if request.method == "GET":
        return render_template("upload.html")

    # Get uploaded file
    file = request.files.get("video")

    # Check file
    if not file or file.filename == "":
        flash("Please select a video file.", "danger")
        return redirect(url_for("upload_video"))

    # Check extension
    if not allowed_file(file.filename):
        flash(
            "Invalid video format. Allowed: MP4, AVI, MOV, MKV, WEBM",
            "danger"
        )
        return redirect(url_for("upload_video"))

    # Secure original filename
    original_filename = secure_filename(file.filename)

    # Create unique filename
    unique_filename = (
        f"{user_id}_{uuid.uuid4().hex}_{original_filename}"
    )

    # Original video path
    video_path = os.path.join(
        VIDEO_FOLDER,
        unique_filename
    )

    # Save original uploaded video
    file.save(video_path)

    # Create audio filename
    audio_filename = (
        f"{os.path.splitext(unique_filename)[0]}.wav"
    )

    # Audio path
    audio_path = os.path.join(
        AUDIO_FOLDER,
        audio_filename
    )

    transcript_path = None

    try:

        # ==========================================
        # STEP 1: EXTRACT AUDIO FOR WHISPER
        # ==========================================

        print("Extracting audio using FFmpeg...")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,

                # IMPORTANT:
                # Extract audio only.
                # This does NOT modify video_path.
                "-vn",

                # Convert audio to WAV
                "-acodec",
                "pcm_s16le",

                # Whisper-friendly format
                "-ar",
                "16000",
                "-ac",
                "1",

                audio_path
            ],
            check=True
        )

        print("Audio extraction completed.")

        # ==========================================
        # STEP 2: TRANSCRIBE AUDIO USING WHISPER
        # ==========================================

        print("Transcribing audio using Whisper...")

        # Call Whisper ONLY ONCE
        segments = transcribe_audio(audio_path)

        print("Whisper transcription completed.")

        # ==========================================
# STEP 3: CREATE SENTENCE-BASED TRANSCRIPT
# ==========================================

        transcript_lines = []

        current_text = []
        current_start = None
        current_end = None

        for segment in segments:

            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()

            if not text:
                continue

            if current_start is None:
                current_start = start

            current_text.append(text)
            current_end = end

            combined_text = " ".join(current_text)

            word_count = len(combined_text.split())

            ends_sentence = combined_text.endswith(
                (".", "?", "!", "।")
            )

            if ends_sentence or word_count >= 25:

                transcript_lines.append(
                    f"[{current_start:.2f} - {current_end:.2f}] {combined_text}"
                )

                current_text = []
                current_start = None
                current_end = None


        # Add remaining text
        if current_text:

            combined_text = " ".join(current_text)

            transcript_lines.append(
                f"[{current_start:.2f} - {current_end:.2f}] {combined_text}"
            )


        transcript_text = "\n".join(transcript_lines)

        print("Sentence-based transcript created.")

        # ==========================================
        # STEP 4: SAVE TRANSCRIPT
        # ==========================================

        transcript_filename = (
            f"{user_id}_{uuid.uuid4().hex}.txt"
        )

        transcript_path = os.path.join(
            TRANSCRIPT_FOLDER,
            transcript_filename
        )

        with open(
            transcript_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(transcript_text)

        print("Timestamped transcript saved.")

        # ==========================================
        # STEP 5: SAVE VIDEO INFORMATION
        # ==========================================

        video = Video(
            filename=unique_filename,
            original_filename=original_filename,
            audio_filename=audio_filename,
            transcript_filename=transcript_filename,
            summary_filename=None,
            status="Transcribed",
            user_id=user_id
        )

        db.session.add(video)
        db.session.commit()

        print(f"Video database ID: {video.id}")

        # ==========================================
        # STEP 6: RAG CHUNKING
        # ==========================================

        chunks = chunk_text_with_timestamps(segments)

        print("\n===== TIMESTAMPED CHUNKS =====\n")

        for i, chunk in enumerate(chunks, start=1):

            print(f"Chunk {i}")
            print(
                f"Start: {chunk['start_time']:.2f}s"
            )
            print(
                f"End:   {chunk['end_time']:.2f}s"
            )
            print(
                f"Text:  {chunk['text']}"
            )
            print("-" * 60)

        print(f"Created {len(chunks)} chunks.")

        # ==========================================
        # STEP 7: CREATE EMBEDDINGS + CHROMADB
        # ==========================================

        print(
            "Creating embeddings and storing in ChromaDB..."
        )

        create_vector_store(
            chunks,
            video.id
        )

        print("ChromaDB storage completed.")

        # ==========================================
        # STEP 8: UPDATE STATUS
        # ==========================================

        video.status = "Ready for search"

        db.session.commit()

        flash(
            "Video uploaded and indexed successfully!",
            "success"
        )

        return redirect(url_for("videos"))

    except Exception as e:

        # ==========================================
        # ERROR HANDLING
        # ==========================================

        print("Processing Error:", e)

        db.session.rollback()

        # Delete original video
        if os.path.exists(video_path):
            os.remove(video_path)

        # Delete extracted audio
        if os.path.exists(audio_path):
            os.remove(audio_path)

        # Delete transcript
        if transcript_path and os.path.exists(transcript_path):
            os.remove(transcript_path)

        flash(
            f"Video processing failed: {str(e)}",
            "danger"
        )

        return redirect(url_for("upload_video"))



# =======================================
# SUMMARY VIDEO
# ========

@app.route("/summarize/<int:video_id>")
def summarize_video(video_id):

    # Check login
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    # Find video
    video = Video.query.filter_by(
        id=video_id,
        user_id=user_id
    ).first()

    if not video:
        flash("Video not found.", "danger")
        return redirect(url_for("videos"))

    # Check transcript
    if not video.transcript_filename:
        flash("Transcript not available.", "danger")
        return redirect(url_for("videos"))

    # Transcript path
    transcript_path = os.path.join(
        TRANSCRIPT_FOLDER,
        video.transcript_filename
    )

    # Check transcript file
    if not os.path.exists(transcript_path):
        flash("Transcript file not found.", "danger")
        return redirect(url_for("videos"))

    try:

        # ==========================================
        # STEP 1: READ TRANSCRIPT
        # ==========================================

        with open(
            transcript_path,
            "r",
            encoding="utf-8"
        ) as f:

            transcript = f.read()

        # ==========================================
        # STEP 2: SEND TRANSCRIPT TO OLLAMA
        # ==========================================

        print("Generating AI summary...")

        ai_result = generate_summary(transcript)

        print("AI summary generated successfully.")

        # ==========================================
        # STEP 3: CREATE SUMMARY FILE
        # ==========================================

        summary_filename = (
            f"{video.id}_{uuid.uuid4().hex}.txt"
        )

        summary_path = os.path.join(
            SUMMARY_FOLDER,
            summary_filename
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(ai_result)

        # ==========================================
        # STEP 4: UPDATE DATABASE
        # ==========================================

        video.summary_filename = summary_filename
        video.status = "Summarized"

        db.session.commit()

        flash(
            "AI summary generated successfully!",
            "success"
        )

        return redirect(
            url_for(
                "video_details",
                video_id=video.id
            )
        )

    except Exception as e:

        print("Summary Error:", e)

        flash(
            f"Summary generation failed: {str(e)}",
            "danger"
        )

        return redirect(url_for("videos"))



# =======================================
# VIDEO DETAILS
# =======================================
@app.route("/video-details/<int:video_id>")
def video_details(video_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    video = Video.query.filter_by(
        id=video_id,
        user_id=user_id
    ).first()

    if not video:
        flash("Video not found.", "danger")
        return redirect(url_for("videos"))

    # ============================================
    # Read Transcript
    # ============================================

    transcript = ""
    transcript_segments = []

    if video.transcript_filename:

        transcript_path = os.path.join(
            TRANSCRIPT_FOLDER,
            video.transcript_filename
        )

        if os.path.exists(transcript_path):

            with open(
                transcript_path,
                "r",
                encoding="utf-8"
            ) as f:

                transcript = f.read()

    # ============================================
    # Read Summary
    # ============================================

    summary = ""

    summary_data = {
        "summary": "",
        "key_points": [],
        "keywords": []
    }

    if video.summary_filename:

        summary_path = os.path.join(
            SUMMARY_FOLDER,
            video.summary_filename
        )

        print("Summary file:", summary_path)

        if os.path.exists(summary_path):

            with open(
                summary_path,
                "r",
                encoding="utf-8"
            ) as f:

                summary = f.read()

            print("Raw summary:")
            print(summary)

            # Parse structured summary
            summary_data = parse_summary(summary)

            print("Parsed summary:")
            print(summary_data)

        else:

            print("Summary file does not exist.")

    # ============================================
    # Render Page
    # ============================================

    return render_template(
        "video_details.html",
        video=video,
        transcript=transcript,
        summary=summary,
        summary_data=summary_data
    )
# 
# =======================================
# VIDEO LIBRARY
# =======================================

@app.route("/videos")
def videos():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    user = User.query.get(
        session["user_id"]
    )


    user_videos = Video.query.filter_by(
        user_id=user.id
    ).order_by(
        Video.uploaded_at.desc()
    ).all()


    return render_template(
        "videos.html",
        user=user,
        videos=user_videos
    )


# =======================================
# VIDEO PLAYER
# =======================================

@app.route("/video/<filename>")
def uploaded_video(filename):

    if "user_id" not in session:
        return redirect(url_for("login"))

    video = Video.query.filter_by(
        filename=filename,
        user_id=session["user_id"]
    ).first_or_404()

    return send_from_directory(
        VIDEO_FOLDER,
        filename,
        conditional=True
    )


# =======================================
# SEARCH
# =======================================

@app.route("/search", methods=["GET", "POST"])
def intelligent_search():

    if "user_id" not in session:
        return redirect(url_for("login"))

    videos = Video.query.filter_by(
        user_id=session["user_id"]
    ).all()

    results = []
    answer = ""
    question = ""

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()

        if not question:

            flash(
                "Please enter a question.",
                "warning"
            )

            return redirect(
                url_for("intelligent_search")
            )

        video_ids = [
            video.id
            for video in videos
        ]

        try:

            results = global_semantic_search(
                question,
                video_ids,
                top_k=5
            )

            if results:

                context_parts = []

                for result in results:

                    source_video = Video.query.get(result["video_id"])

                    if source_video:

                        context_parts.append(
                            f"""
                VIDEO: {source_video.original_filename}
                TIMESTAMP: {result["start_time"]:.2f} - {result["end_time"]:.2f} seconds

                CONTENT:
                {result["chunk"]}
                """
                        )

                context = "\n\n".join(context_parts)

                answer = answer_question(
                    question,
                    context
                )

            else:

                answer = (
                    "I could not find the answer "
                    "in your videos."
                )

        except Exception as e:

            print("Multi-video search error:", e)

            flash(
                f"Search failed: {str(e)}",
                "danger"
            )

    return render_template(
        "search.html",
        videos=videos,
        results=results,
        answer=answer,
        question=question
    )

# DELETE VIDEO

@app.route("/delete-video/<int:video_id>", methods=["POST"])
def delete_video(video_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    video = Video.query.filter_by(
        id=video_id,
        user_id=session["user_id"]
    ).first_or_404()

    try:

        # -----------------------------
        # Delete ChromaDB vectors
        # -----------------------------

        delete_video_vectors(video.id)

        # -----------------------------
        # Delete video file
        # -----------------------------

        if video.filename:

            video_path = os.path.join(
                VIDEO_FOLDER,
                video.filename
            )

            if os.path.exists(video_path):
                os.remove(video_path)

        # -----------------------------
        # Delete audio
        # -----------------------------

        if video.audio_filename:

            audio_path = os.path.join(
                AUDIO_FOLDER,
                video.audio_filename
            )

            if os.path.exists(audio_path):
                os.remove(audio_path)

        # -----------------------------
        # Delete transcript
        # -----------------------------

        if video.transcript_filename:

            transcript_path = os.path.join(
                TRANSCRIPT_FOLDER,
                video.transcript_filename
            )

            if os.path.exists(transcript_path):
                os.remove(transcript_path)

        # -----------------------------
        # Delete summary
        # -----------------------------

        if video.summary_filename:

            summary_path = os.path.join(
                SUMMARY_FOLDER,
                video.summary_filename
            )

            if os.path.exists(summary_path):
                os.remove(summary_path)

        # -----------------------------
        # Delete database record
        # -----------------------------

        db.session.delete(video)
        db.session.commit()

        flash(
            "Video deleted successfully.",
            "success"
        )

    except Exception as e:

        db.session.rollback()

        print("Delete Video Error:", e)

        flash(
            f"Failed to delete video: {str(e)}",
            "danger"
        )

    return redirect(url_for("videos"))


# =======================================
# SETTINGS
# =======================================

@app.route("/settings", methods=["GET", "POST"])
def settings():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = User.query.get_or_404(session["user_id"])

    if request.method == "POST":

        action = request.form.get("action")

        # =========================
        # UPDATE PROFILE
        # =========================
        if action == "profile":

            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()

            if not name or not email:
                flash("Name and email are required.", "danger")
                return redirect(url_for("settings"))

            existing_user = User.query.filter(
                User.email == email,
                User.id != user.id
            ).first()

            if existing_user:
                flash("This email is already registered.", "danger")
                return redirect(url_for("settings"))

            user.name = name
            user.email = email

            db.session.commit()

            flash("Profile updated successfully.", "success")

            return redirect(url_for("settings"))


        # =========================
        # CHANGE PASSWORD
        # =========================
        elif action == "password":

            current_password = request.form.get("current_password", "")
            new_password = request.form.get("new_password", "")
            confirm_password = request.form.get("confirm_password", "")

            if not current_password or not new_password or not confirm_password:
                flash("Please fill in all password fields.", "danger")
                return redirect(url_for("settings"))

            if not check_password_hash(
                user.password_hash,
                current_password
            ):
                flash("Current password is incorrect.", "danger")
                return redirect(url_for("settings"))

            if len(new_password) < 8:
                flash(
                    "New password must be at least 8 characters long.",
                    "danger"
                )
                return redirect(url_for("settings"))

            if new_password != confirm_password:
                flash("New passwords do not match.", "danger")
                return redirect(url_for("settings"))

            if current_password == new_password:
                flash(
                    "New password must be different from your current password.",
                    "danger"
                )
                return redirect(url_for("settings"))

            user.password_hash = generate_password_hash(new_password)

            db.session.commit()

            flash("Password changed successfully.", "success")

            return redirect(url_for("settings"))

    return render_template(
        "settings.html",
        user=user
    )

# =======================================
# RUN
# =======================================

if __name__ == "__main__":

    app.run(
        debug=True
    )