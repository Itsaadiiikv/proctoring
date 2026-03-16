
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [

    # Home Pages
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),

    # Authentication
    path('registration/', views.registration, name='registration'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Video Feed
    path('video_feed/', views.video_feed, name='video_feed'),

    # Dashboard
    path('dashboard/', views.dashboard, name='dashboard'),

    # Exam System
    path('exam/', views.exam, name='exam'),
    path('submit_exam/', views.submit_exam, name='submit_exam'),
    path('result/', views.result, name='result'),

    # Warning API
    path('get_warning/', views.get_warning, name='get_warning'),

    # Admin
    path('admin_dashboard/', views.admin_dashboard, name='admin_dashboard'),

    # Add Question Page
    path('admin_dashboard/add_question/', views.add_question, name='add_question'),

]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

