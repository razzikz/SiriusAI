# Generated by Django 5.0.6 on 2024-05-14 18:16

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0002_alter_counter_count'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Counter',
        ),
    ]