from django.db import migrations, models
import django.contrib.gis.db.models.fields


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0005_remove_dataextractionlog_api_source_and_more'),
    ]

    operations = [
        # 1. Drop DataExtractionLog
        migrations.DeleteModel(
            name='DataExtractionLog',
        ),

        # 2. Update SoilCache
        migrations.AlterModelOptions(
            name='soilcache',
            options={'verbose_name': 'Soil Cache', 'verbose_name_plural': 'Soil Caches'},
        ),
        migrations.AlterUniqueTogether(
            name='soilcache',
            unique_together=set(),
        ),
        migrations.RemoveField(
            model_name='soilcache',
            name='latitude',
        ),
        migrations.RemoveField(
            model_name='soilcache',
            name='longitude',
        ),
        migrations.AddField(
            model_name='soilcache',
            name='location',
            field=django.contrib.gis.db.models.fields.PointField(
                help_text='Geographic point (lon, lat) of soil cache',
                srid=4326,
            ),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='soilcache',
            name='bulk_density',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='soilcache',
            name='organic_carbon_pct',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='soilcache',
            name='ph',
            field=models.FloatField(blank=True, null=True),
        ),

        # 3. Update WeatherCache
        migrations.AlterModelOptions(
            name='weathercache',
            options={'verbose_name': 'Weather Cache', 'verbose_name_plural': 'Weather Caches'},
        ),
        migrations.RemoveIndex(
            model_name='weathercache',
            name='data_weathe_latitud_69b729_idx',
        ),
        migrations.AlterUniqueTogether(
            name='weathercache',
            unique_together=set(),
        ),
        migrations.RemoveField(
            model_name='weathercache',
            name='latitude',
        ),
        migrations.RemoveField(
            model_name='weathercache',
            name='longitude',
        ),
        migrations.AddField(
            model_name='weathercache',
            name='location',
            field=django.contrib.gis.db.models.fields.PointField(
                help_text='Geographic point (lon, lat) of weather cache',
                srid=4326,
            ),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='weathercache',
            name='humidity',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='weathercache',
            name='precipitation',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='weathercache',
            name='radiation',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='weathercache',
            name='temp_max',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='weathercache',
            name='temp_min',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterUniqueTogether(
            name='weathercache',
            unique_together={('location', 'date')},
        ),
        migrations.AddIndex(
            model_name='weathercache',
            index=models.Index(fields=['location', 'date'], name='data_weathe_locatio_fa3351_idx'),
        ),
    ]
