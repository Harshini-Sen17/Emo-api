# Generated by Django 3.2 on 2021-05-01 16:19

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Songs',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=50)),
                ('description', models.TextField()),
                ('singer', models.CharField(max_length=20)),
                ('mood', models.CharField(choices=[('1', 'Angry'), ('2', 'Happy'), ('3', 'Neutral'), ('4', 'Sad'), ('3', 'Suprise')], max_length=30)),
                ('link', models.URLField()),
            ],
        ),
    ]
