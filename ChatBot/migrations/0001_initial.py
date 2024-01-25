# Generated by Django 3.2.11 on 2023-05-26 12:12

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Message',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.CharField(default='Hi', max_length=255)),
                ('is_bot', models.BooleanField(default=False)),
            ],
        ),
    ]
