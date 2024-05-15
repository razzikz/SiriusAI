from django.shortcuts import render


def index(request):
    return render(request, 'main_root/main.html')


def queue(request):
    return render(request, 'main_root/queue.html')


def faq(request):
    return render(request, 'main_root/faq.html')


def settings(request):
    return render(request, 'main_root/settings.html')


def test(request):
    return render(request, 'main_root/test.html')