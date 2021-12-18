from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import messages
from django.core.files.storage import FileSystemStorage  # To upload Profile Picture
from django.urls import reverse
from datetime import datetime  # To Parse input DateTime into Python Date Time Object
from django.core.paginator import Paginator

from employee_management_app.models import (
    CustomUser,
    Employee,
    Attendance,
)
from django.views.decorators.cache import cache_control
from django.contrib.auth.decorators import login_required
from .decorators import allowed_users
import xlwt
# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
def employee_home(request):
    current_month = datetime.now().month
    employee = Employee.objects.get(admin=request.user.id)
    total_attendance = Attendance.objects.filter(
        employee_id=employee.id, created_at__month=current_month).count()
    context = {
        "employee": employee,
        "total_attendance": total_attendance
    }
   
    return render(request, "employee_template/employee_home_template.html", context)

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
def employee_view_attendance(request):
    employee = Employee.objects.get(admin=request.user.id)
    if request.method == "POST":
        year = request.POST.get('year')
        month = request.POST.get('month')
        display = True

        data = Attendance.objects.filter(employee_id=employee.id,created_at__year=year,created_at__month=month,present=True).order_by('-created_at')
        
        if not data:
            messages.warning(request, "Không có dữ liệu để hiển thị")
            display = False

        context = {
            "data": data,
            'employee_id': employee.id,
            'month': month,
            'year': year,
            'display': display
        }

        return render(request, "employee_template/employee_view_attendance.html", context)
    else:
        current_month = datetime.now().month
        current_year = datetime.now().year
        data = Attendance.objects.filter(employee_id=employee.id,created_at__month=current_month)
        display = True
        context = {
            "data": data,
            "employee": employee,
            "display": display,
            'month': current_month,
            'year': current_year
        }
        return render(request, "employee_template/employee_view_attendance.html", context)

@login_required(login_url="login")
def employee_export_excel(request, month, year):
    employee = Employee.objects.get(admin=request.user.id)
    response = HttpResponse(content_type='application/ms-excel')
    response['Content-Disposition'] = 'attachment; filename=Diemdanhcuatoi' + \
        str(datetime.now().date())+'.xls'

    wb = xlwt.Workbook(encoding='utf-8')
    ws = wb.add_sheet('Diemdanh')
    row_num = 0
    font_style = xlwt.XFStyle()
    font_style.font.bold = True

    colums = ['ID', 'HO TEN','PHONG BAN', 'NGAY', 'THOI GIAN VAO', 'THOI GIAN RA']

    for col_num in range(len(colums)):
        ws.write(row_num, col_num, colums[col_num], font_style)

    font_style = xlwt.XFStyle()

    data = Attendance.objects.filter(employee_id=employee.id,created_at__year=year,created_at__month=month,present=True).values_list(
        'employee_id', 'name','department_id', 'attendance_date', 'attendance_time_in','attendance_time_out')
    
    rows = data

    for row in rows:
        row_num += 1

        for col_num in range(len(row)):
            ws.write(row_num, col_num, str(row[col_num]), font_style)
    wb.save(response)

    return response
# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
def employee_profile(request):
    user = CustomUser.objects.get(id=request.user.id)
    employee = Employee.objects.get(admin=user)

    context = {"user": user, "employee": employee}
    return render(request, "employee_template/employee_profile.html", context)

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
def employee_profile_update(request):
    if request.method != "POST":
        messages.error(request, "Invalid Method!")
        return redirect("employee_profile")
    else:

        password = request.POST.get("password")
        address = request.POST.get("address")

        try:
            customuser = CustomUser.objects.get(id=request.user.id)
            if password != None and password != "":
                customuser.set_password(password)
            customuser.save()

            employee = Employee.objects.get(admin=customuser.id)
            employee.address = address

            employee.save()

            messages.success(request, "Cập nhật hồ sơ thành công")
            return redirect("employee_profile")
        except:
            messages.error(request, "Cập nhật hồ sơ thất bại")
            return redirect("employee_profile")
