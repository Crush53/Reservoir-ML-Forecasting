# Generated by Django 4.2.6 on 2023-12-06 02:29

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ResMeta',
            fields=[
                ('stn_id', models.CharField(max_length=15, primary_key=True, serialize=False)),
                ('stn_name', models.CharField(max_length=255)),
                ('lat', models.FloatField()),
                ('lon', models.FloatField()),
            ],
            options={
                'db_table': 'res_meta',
            },
        ),
        migrations.CreateModel(
            name='WeatherData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('datetime', models.DateTimeField()),
                ('average_temperature', models.FloatField()),
                ('precipitation', models.FloatField()),
                ('stn_id', models.ForeignKey(db_column='stn_id', on_delete=django.db.models.deletion.CASCADE, to='res_view.resmeta')),
            ],
            options={
                'db_table': 'weather_data',
            },
        ),
        migrations.CreateModel(
            name='ResData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('datetime', models.DateTimeField()),
                ('storage_value', models.FloatField()),
                ('stn_id', models.ForeignKey(db_column='stn_id', on_delete=django.db.models.deletion.CASCADE, to='res_view.resmeta')),
            ],
            options={
                'db_table': 'res_data',
            },
        ),
    ]
