# Generated by Django 5.0.6 on 2024-05-14 18:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='counter',
            name='count',
            field=models.FloatField(),
        ),
    ]
