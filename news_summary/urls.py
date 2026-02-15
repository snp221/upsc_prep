from django.contrib import admin
from django.urls import include, path
from .views import (
    current_affairs,
    home,
    affairs_by_date,
    upsc_chat,
    answer_eval,
    ca_generate_digest,
    todo_list,
    todo_toggle,
)

urlpatterns = [
    path('home/', home, name='home'),
    path('current-affairs/', current_affairs, name='current_affairs'),
    path("current-affairs/<str:date>/", affairs_by_date, name="ca_detail"),
    path("current-affairs/<str:date>/generate-digest/", ca_generate_digest, name="ca_generate_digest"),
    path("upsc-chat/", upsc_chat, name="upsc_chat"),
    path("answer-eval/", answer_eval, name="answer_eval"),
    path("todos/", todo_list, name="todo_list"),
    path("todos/<int:pk>/toggle/", todo_toggle, name="todo_toggle"),

]