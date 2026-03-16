
# -------------------- IMPORTS --------------------
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from django.contrib.auth import authenticate, login as auth_login, logout
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import os
import json
import base64
import numpy as np
import cv2
import threading
import time
import logging

from PIL import Image
from django.core.files.base import ContentFile

from .models import Student, Exam, CheatingEvent, CheatingImage, CheatingAudio

# AI modules
from .ml_models.object_detection import detectObject
from .ml_models.audio_detection import audio_detection
from .ml_models.gaze_tracking import gaze_tracking

import face_recognition

logger = logging.getLogger(__name__)

warning = None
stop_event = threading.Event()


# -------------------- HOME --------------------
def home(request):
    return render(request, "home.html")


# -------------------- VIDEO STREAM --------------------
def gen_frames():
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


def video_feed(request):
    return StreamingHttpResponse(
        gen_frames(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )


# -------------------- REGISTRATION --------------------
def registration(request):

    if request.method == "POST":

        name = request.POST["name"]
        address = request.POST["address"]
        email = request.POST["email"]
        password = request.POST["password"]
        captured_photo = request.POST.get("photo_data")

        img_data = base64.b64decode(captured_photo.split(",")[1])
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        face_locations = face_recognition.face_locations(image)

        if not face_locations:
            messages.error(request, "No face detected.")
            return redirect("registration")

        encoding = face_recognition.face_encodings(image, face_locations)[0]

        user = User.objects.create(
            username=email,
            email=email,
            password=make_password(password),
            first_name=name
        )

        Student.objects.create(
            user=user,
            name=name,
            address=address,
            email=email,
            photo=ContentFile(img_data, name=f"{name}.jpg"),
            face_encoding=encoding.tolist()
        )

        messages.success(request, "Registration successful")
        return redirect("login")

    return render(request, "registration.html")


# -------------------- LOGIN --------------------
@csrf_exempt
def login(request):

    if request.method == "POST":

        email = request.POST.get("email")
        password = request.POST.get("password")
        photo_data = request.POST.get("captured_photo")

        photo_data = base64.b64decode(photo_data.split(",")[1])

        nparr = np.frombuffer(photo_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        face_locations = face_recognition.face_locations(image)

        if not face_locations:
            return JsonResponse({"success": False, "error": "Face not detected"})

        captured_encoding = face_recognition.face_encodings(image, face_locations)[0]

        user = authenticate(request, username=email, password=password)

        if not user:
            return JsonResponse({"success": False, "error": "Invalid credentials"})

        student = user.student
        stored_encoding = np.array(student.face_encoding)

        match = face_recognition.compare_faces([stored_encoding], captured_encoding)[0]

        if not match:
            return JsonResponse({"success": False, "error": "Face mismatch"})

        auth_login(request, user)

        return JsonResponse({
            "success": True,
            "redirect_url": "/dashboard/",
            "student_name": student.name
        })

    return render(request, "login.html")


# -------------------- LOGOUT --------------------
def logout_view(request):
    logout(request)
    return redirect("home")


# -------------------- DASHBOARD --------------------
@login_required
def dashboard(request):

    return render(request, "dashboard.html", {
        "user_name": request.user.first_name
    })


# -------------------- EXAM PAGE --------------------
@login_required
def exam(request):

    try:

        file_path = os.path.join(
            settings.BASE_DIR,
            "proctoring",
            "dummy_data",
            "ai.json"
        )

        with open(file_path) as f:
            data = json.load(f)

        questions = data["questions"]

    except:
        return HttpResponse("Questions file error")

    stop_event.clear()

    threading.Thread(
        target=background_processing,
        args=(request,),
        daemon=True
    ).start()

    threading.Thread(
        target=process_audio,
        args=(request,),
        daemon=True
    ).start()

    return render(request, "exam.html", {
        "questions": questions
    })


# -------------------- SUBMIT EXAM --------------------
@login_required
def submit_exam(request):

    if request.method == "POST":

        stop_event.set()

        try:

            file_path = os.path.join(
                settings.BASE_DIR,
                "proctoring",
                "dummy_data",
                "ai.json"
            )

            with open(file_path) as f:
                data = json.load(f)

        except:
            return HttpResponse("Question file error")

        questions = data["questions"]

        total_questions = len(questions)
        correct_answers = 0

        for q in questions:

            qid = q["id"]

            user_answer = request.POST.get(f"answer_{qid}")

            if user_answer == q["correct_answer"]:
                correct_answers += 1

        percentage = (correct_answers / total_questions) * 100

        Exam.objects.create(
            student=request.user.student,
            exam_name="AI Exam",
            total_questions=total_questions,
            correct_answers=correct_answers,
            percentage_score=round(percentage, 2),
            status="completed"
        )

        messages.success(request, "Exam submitted successfully")

        return redirect("result")

    return HttpResponse("Invalid request")


# -------------------- RESULT --------------------
@login_required
def result(request):

    exam = Exam.objects.filter(
        student=request.user.student,
        status="completed"
    ).latest("timestamp")

    return render(request, "result.html", {
        "user_name": request.user.student.name,
        "score": exam.correct_answers,
        "total_questions": exam.total_questions,
        "percentage": exam.percentage_score
    })


# -------------------- AUDIO DETECTION --------------------
def process_audio(request):

    global warning

    while not stop_event.is_set():

        audio = audio_detection()

        if audio["audio_detected"]:
            warning = "ALERT: Suspicious audio detected!"

        time.sleep(2)


# -------------------- VIDEO DETECTION --------------------
def background_processing(request):

    cap = cv2.VideoCapture(0)

    while not stop_event.is_set():

        ret, frame = cap.read()

        if not ret:
            break

        labels, processed_frame, person_count, detected_objects = detectObject(frame)

        if person_count > 1:
            global warning
            warning = "ALERT: Multiple persons detected!"

        gaze = gaze_tracking(frame)

        if gaze["gaze"] != "center":
            warning = "ALERT: Not looking at screen"

        time.sleep(0.5)

    cap.release()


# -------------------- WARNING API --------------------
@csrf_exempt
def get_warning(request):
    return JsonResponse({"warning": warning})


# -------------------- ADD QUESTION PAGE --------------------
def add_question(request):
    return render(request, "add_question.html")


# -------------------- SIMPLE PAGES --------------------
def about(request):
    return render(request, "about.html")


def contact(request):
    return render(request, "contact.html")
# -------------------- EXAM SUBMISSION SUCCESS PAGE --------------------
def exam_submission_success(request):
    return render(request, 'exam_submission_success.html')
# -------------------- ADMIN DASHBOARD --------------------
from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Count
from .models import Student

@staff_member_required(login_url='/admin/login/')
def admin_dashboard(request):

    students = Student.objects.annotate(
        exam_count=Count('exams'),
        cheating_event_count=Count('cheating_events')
    )

    context = {
        "students": students
    }

    return render(request, "admin_dashboard.html", context)