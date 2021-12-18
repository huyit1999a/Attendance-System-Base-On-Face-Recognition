from django.contrib.auth.models import AbstractUser
from django.db import models
from django.db.models.signals import post_save, post_delete, pre_delete
from django.dispatch import receiver
import os
from employee_management_system import settings

# Overriding the Default Django Auth User and adding One More Field (user_type)


class CustomUser(AbstractUser):
    user_type_data = ((1, "HOD"), (2, "Employee"))
    user_type = models.CharField(
        default=1, choices=user_type_data, max_length=10)


class AdminHOD(models.Model):
    id = models.AutoField(primary_key=True)
    admin = models.OneToOneField(CustomUser, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    objects = models.Manager()


class Departments(models.Model):
    id = models.AutoField(primary_key=True)
    department_name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    objects = models.Manager()

    def __str__(self):
        return self.department_name


class Employee(models.Model):
    id = models.AutoField(primary_key=True)
    admin = models.OneToOneField(CustomUser, on_delete=models.CASCADE)
    gender = models.CharField(max_length=50)
    profile_pic = models.FileField(null=True, blank=True)
    address = models.TextField()
    name = models.CharField(max_length=255)
    department_id = models.ForeignKey(
        Departments, on_delete=models.DO_NOTHING, default=1
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    objects = models.Manager()


class Attendance(models.Model):
    # Subject Attendance
    id = models.AutoField(primary_key=True)
    employee_id = models.ForeignKey(Employee, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    department_id = models.CharField(max_length=50)
    attendance_date = models.DateField(auto_now_add=True)
    attendance_time_in = models.CharField(max_length=20)
    attendance_time_out = models.CharField(max_length=20)
    present= models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    objects = models.Manager()


# Creating Django Signals

# It's like trigger in database. It will run only when Data is Added in CustomUser model


@receiver(post_save, sender=CustomUser)
# Now Creating a Function which will automatically insert data in HOD, Staff or Employee
def create_user_profile(sender, instance, created, **kwargs):
    # if Created is true (Means Data Inserted)
    if created:
        # Check the user_type and insert the data in respective tables
        if instance.user_type == 1:
            AdminHOD.objects.create(admin=instance)
        if instance.user_type == 2:
            Employee.objects.create(
                admin=instance,
                department_id=Departments.objects.get(id=1),
                address="",
                name="",
                profile_pic="",
                gender="",
            )


@receiver(post_save, sender=CustomUser)
def save_user_profile(sender, instance, **kwargs):
    if instance.user_type == 1:
        instance.adminhod.save()
    if instance.user_type == 2:
        instance.employee.save()
