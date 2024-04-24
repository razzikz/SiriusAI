from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from skills import views
from django.views.i18n import JavaScriptCatalog

urlpatterns = [
    path('login/', views.login)
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
