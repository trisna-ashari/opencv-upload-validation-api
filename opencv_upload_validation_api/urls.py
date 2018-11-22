from django.conf.urls import url
from django.contrib import admin
import validation.views
 
urlpatterns = [
        url(r'^admin/', admin.site.urls),
        url(r'^validation/validate/$', validation.views.validate),
]