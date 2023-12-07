from django.db import models

# Create your models here.
class ResMeta(models.Model):
    stn_id = models.CharField(max_length=15, primary_key=True)
    stn_name = models.CharField(max_length=255)
    lat = models.FloatField()  # Renamed from latitude to lat
    lon = models.FloatField()  # Renamed from longitude to lon

    def __str__(self):
        return self.stn_name

    class Meta:
        db_table = 'res_meta'  # Specify the table name

class ResData(models.Model):
    stn_id = models.ForeignKey(ResMeta, on_delete=models.CASCADE, db_column='stn_id')
    datetime = models.DateTimeField()
    storage_value = models.FloatField()

    def __str__(self):
        return f"{self.stn_id.stn_name} - {self.datetime.strftime('%Y-%m-%d')}: {self.storage_value}"

    class Meta:
        db_table = 'res_data'  # Specify the table name explicitly
        
class WeatherData(models.Model):
    stn_id = models.ForeignKey(ResMeta, on_delete=models.CASCADE, db_column='stn_id')
    datetime = models.DateTimeField()
    average_temperature = models.FloatField()
    precipitation = models.FloatField()

    def __str__(self):
        return f"{self.stn_id.stn_name} - {self.datetime.strftime('%Y-%m-%d')}"

    class Meta:
        db_table = 'weather_data'  # Specify the table name explicitly