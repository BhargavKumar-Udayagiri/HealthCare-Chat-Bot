from django.db import models

class registration(models.Model):
    First_Name=models.CharField(max_length=100)
    Last_Name=models.CharField(max_length=100)
    Email=models.EmailField(primary_key="True",max_length=100)
    Password=models.CharField(max_length=100)
    Phone_Number=models.CharField(max_length=10)
    Address=models.CharField(max_length=100)

class Message(models.Model):
    text = models.CharField(max_length=255, default="Hi")
    is_bot = models.BooleanField(default=False)
    is_asking = models.BooleanField(default=False)
    
    def __str__(self):
        return self.text


class Prompt(models.Model):

    is_propmted = models.BooleanField(default=False)

    def __str__(self):
        return str(self.is_propmted)