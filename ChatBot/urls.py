from django.urls import path
from . import views 

urlpatterns = [
    path("",views.index,name="index"),
    path('contact/', views.contact, name='contact'),
    path('registration/', views.registration, name='registration'),
    path('login/', views.login, name='login'),
    path("reset",views.reset,name="reset"),
]