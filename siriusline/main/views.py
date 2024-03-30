from django.shortcuts import render


def index(request):
    return render(request, 'main/main.html')


def queue(request):
    return render(request, 'main/queue.html')


def faq(request):
    return render(request, 'main/faq.html')


def settings(request):
    return render(request, 'main/settings.html')