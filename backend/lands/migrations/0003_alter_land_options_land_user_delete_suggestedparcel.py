from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('lands', '0002_alter_land_id_alter_suggestedparcel_id'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='land',
            options={'ordering': ['-created_at'], 'verbose_name': 'Land Field', 'verbose_name_plural': 'Land Fields'},
        ),
        migrations.AddField(
            model_name='land',
            name='user',
            field=models.ForeignKey(
                default=1,
                help_text='Farmer who owns this land parcel/field',
                on_delete=django.db.models.deletion.CASCADE,
                related_name='lands',
                to=settings.AUTH_USER_MODEL,
            ),
            preserve_default=False,
        ),
        migrations.DeleteModel(
            name='SuggestedParcel',
        ),
    ]
