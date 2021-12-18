from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.contrib import messages
from django.core.files.storage import FileSystemStorage  # To upload Profile Picture
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.core import serializers
from django.db.models import Count
import json
import cv2
import os
import xlwt
import dlib
from math import hypot
from datetime import datetime
from django.core.paginator import Paginator
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
import math
import random
from sklearn import neighbors
from scipy.spatial import distance as dist
from employee_management_app.extract_embedding import *
from calendar import monthrange

from employee_management_app.models import (
    CustomUser,
    Departments,
    Employee,
    Attendance,
)
from .forms import AddEmployeeForm, EditEmployeeForm
from employee_management_app.define_constants import *
from django.views.decorators.cache import cache_control
from django.contrib.auth.decorators import login_required
from .decorators import allowed_users, admin_only


# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def admin_home(request):
    all_employee_count = Employee.objects.all().count()
    all_department_count = Departments.objects.all().count()

    context = {
        "all_employee_count": all_employee_count,
        "all_department_count": all_department_count
    }
    return render(request, "hod_template/home_content.html", context)

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def take_attendance(request):
    return render(request, "hod_template/take_attendance_template.html")

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def add_department(request):
    return render(request, "hod_template/add_department_template.html")

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def add_department_save(request):
    if request.method != "POST":
        messages.error(request, "Invalid Method!")
        return redirect("add_department")
    else:
        department = request.POST.get("department")
        try:
            department_model = Departments(department_name=department)
            department_model.save()
            messages.success(request, "Thêm phòng ban thành công!")
            return redirect("manage_department")
        except:
            messages.error(request, "Thêm phòng ban thất bại!")
            return redirect("add_department")

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def manage_department(request):
    departments = Departments.objects.all()
    context = {"departments": departments}
    return render(request, "hod_template/manage_department_template.html", context)

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def edit_department(request, department_id):
    department = Departments.objects.get(id=department_id)
    context = {"department": department, "id": department_id}
    return render(request, "hod_template/edit_department_template.html", context)

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def edit_department_save(request):
    if request.method != "POST":
        HttpResponse("Invalid Method")
    else:
        department_id = request.POST.get("department_id")
        department_name = request.POST.get("department_name")

        try:
            department = Departments.objects.get(id=department_id)
            department.department_name = department_name
            department.save()

            messages.success(request, "Cập nhật tên phòng ban thành công")
            return redirect("manage_department")

        except:
            messages.error(request, "Cập nhật thất bại")
            return redirect("/edit_department/" + department_id)

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def delete_department(request, department_id):
    department = Departments.objects.get(id=department_id)
    try:
        department.delete()
        messages.success(request, "Xóa phòng ban thành công.")
        return redirect("manage_department")
    except:
        messages.error(request, "Xóa thất bại.")
        return redirect("manage_department")

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def add_employee(request):
    form = AddEmployeeForm()
    context = {"form": form}
    return render(request, "hod_template/add_employee_template.html", context)

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def add_employee_save(request):
    if request.method != "POST":
        messages.error(request, "Invalid Method")
        return redirect("add_employee")
    else:
        form = AddEmployeeForm(request.POST, request.FILES)

        if form.is_valid():
            first_name = form.cleaned_data["name"]
            last_name = form.cleaned_data["name"]
            username = form.cleaned_data["email"]
            name = form.cleaned_data["name"]
            email = form.cleaned_data["email"]
            password = form.cleaned_data["password"]
            address = form.cleaned_data["address"]
            department_id = form.cleaned_data["department_id"]
            gender = form.cleaned_data["gender"]

            # Getting Profile Pic first
            # First Check whether the file is selected or not
            # Upload only if file is selected
            if len(request.FILES) != 0:
                profile_pic = request.FILES["profile_pic"]
                # fs = FileSystemStorage()
                # filename = fs.save(profile_pic.name, profile_pic)
                # profile_pic_url = fs.url(filename)
            else:
                profile_pic_url = None

            try:
                user = CustomUser.objects.create_user(
                    username=username,
                    password=password,
                    email=email,
                    first_name=first_name,
                    last_name=last_name,
                    user_type=2,
                )
                user.employee.address = address
                department_obj = Departments.objects.get(id=department_id)
                user.employee.department_id = department_obj

                user.employee.name = name
                user.employee.gender = gender
                user.employee.profile_pic = profile_pic

                user.save()
                messages.success(request, "Thêm nhân viên thành công!")
                return redirect("manage_employee")
            except:
                messages.error(request, "Thêm nhân viên thất bại!")
                print(form.errors)
                return redirect("add_employee")
        else:
            return redirect("add_employee")

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def manage_employee(request):
    employees = Employee.objects.all()
    context = {"employees": employees}
    return render(request, "hod_template/manage_employee_template.html", context)

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def edit_employee(request, employee_id):
    # Adding employee ID into Session Variable
    request.session["employee_id"] = employee_id
    employee = Employee.objects.get(admin=employee_id)
    form = EditEmployeeForm()

    # Filling the form with Data from Database
    form.fields["email"].initial = employee.admin.email
    # form.fields["password"].initial = employee.admin.password
    form.fields["name"].initial = employee.name
    form.fields["address"].initial = employee.address
    form.fields["department_id"].initial = employee.department_id.id
    form.fields["gender"].initial = employee.gender

    context = {"id": employee_id, "form": form}
    return render(request, "hod_template/edit_employee_template.html", context)

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def edit_employee_save(request):
    if request.method != "POST":
        return HttpResponse("Invalid Method!")
    else:
        employee_id = request.session.get("employee_id")
        if employee_id == None:
            return redirect("/manage_employee")

        form = EditEmployeeForm(request.POST, request.FILES)
        print(request.POST)
        if form.is_valid():
            first_name = form.cleaned_data["name"]
            last_name = form.cleaned_data["name"]
            name = form.cleaned_data["name"]
            address = form.cleaned_data["address"]
            department_id = form.cleaned_data["department_id"]
            gender = form.cleaned_data["gender"]

            # Getting Profile Pic first
            # First Check whether the file is selected or not
            # Upload only if file is selected
            if len(request.FILES) != 0:
                profile_pic = request.FILES["profile_pic"]
                # fs = FileSystemStorage()
                # filename = fs.save(profile_pic.name, profile_pic)
                # profile_pic_url = fs.url(filename)
            else:
                profile_pic = None

            try:
                # First Update into Custom User Model
                user = CustomUser.objects.get(id=employee_id)
                user.first_name = first_name
                user.last_name = last_name
                user.save()

                # Then Update employees Table
                employee_model = Employee.objects.get(admin=employee_id)
                employee_model.address = address

                department_obj = Departments.objects.get(id=department_id)
                employee_model.department_id = department_obj
                employee_model.name = name
                employee_model.gender = gender
                
                if profile_pic != None:
                    os.remove(employee_model.profile_pic.path)
                    employee_model.profile_pic = profile_pic
                    
                    
                employee_model.save()
                # Delete employee_id SESSION after the data is updated
                del request.session["employee_id"]

                messages.success(request, "Cập nhật nhân viên thành công")
                return redirect("/manage_employee/")
            except:
                return redirect("/edit_employee/" + employee_id)
        else:
            messages.error(request, "Cập nhật nhân viên thất bại")
            print(form.errors)
            return redirect("/edit_employee/" + employee_id)

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def delete_employee(request, employee_id):
    employee = Employee.objects.get(admin=employee_id)
    user = CustomUser.objects.get(id=employee_id)

    try:
        if len(employee.profile_pic) > 0:
            os.remove(employee.profile_pic.path)

        employee.delete()
        user.delete()
        messages.success(request, "Xóa nhân viên thành công")
        return redirect("manage_employee")
    except:
        messages.error(request, "Xóa nhân viên thất bại")
        print(employee.profile_pic.path)
        return redirect("manage_employee")


@csrf_exempt
def check_email_exist(request):
    email = request.POST.get("email")
    user_obj = CustomUser.objects.filter(email=email).exists()
    if user_obj:
        return HttpResponse(True)
    else:
        return HttpResponse(False)


@csrf_exempt
def check_username_exist(request):
    username = request.POST.get("username")
    user_obj = CustomUser.objects.filter(username=username).exists()
    if user_obj:
        return HttpResponse(True)
    else:
        return HttpResponse(False)

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def admin_profile(request):
    user = CustomUser.objects.get(id=request.user.id)

    context = {"user": user}
    return render(request, "hod_template/admin_profile.html", context)

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def admin_profile_update(request):
    if request.method != "POST":
        messages.error(request, "Invalid Method!")
        return redirect("admin_profile")
    else:
        first_name = request.POST.get("first_name")
        last_name = request.POST.get("last_name")
        password = request.POST.get("password")

        try:
            customuser = CustomUser.objects.get(id=request.user.id)
            customuser.first_name = first_name
            customuser.last_name = last_name
            if password != None and password != "":
                customuser.set_password(password)
            customuser.save()
            messages.success(request, "Profile Updated Successfully")
            return redirect("admin_profile")
        except:
            messages.error(request, "Failed to Update Profile")
            return redirect("admin_profile")

def employee_profile(request):
    pass

def get_EAR_ratio(eye_points):
    # euclidean distance between two vertical eye landmarks
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])

    # euclidean distance between horizontal eye landmarks
    C = dist.euclidean(eye_points[0], eye_points[3])

    # Eye Aspect Ratio
    return (A + B) / (2.0 * C)

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def train(request):
    all_face_encodings = {}

    # Load a sample picture and learn how to recognize it.
    images = []
    encodings = []
    names = []
    files = []
    employee_ids = []
    employees = Employee.objects.all()

    for employee in employees:
        images.append(employee.name + "_image")
        encodings.append(employee.name + "_face_encoding")
        files.append(employee.profile_pic)
        names.append(employee.name)  # + "\nID: " + str(employee.id)
        employee_ids.append(employee.id)
    
    # for i in range(0, len(images)):
    #     images[i] = face_recognition.load_image_file(files[i])
    #     boxes = face_recognition.face_locations(images[i],model="hog")
    #     encodings[i] = face_recognition.face_encodings(images[i],boxes,num_jitters=1)[0]
    #     print(f"Data saved for {i+1} images...")

    #     # Create arrays of known face encodings and their names
    # all_face_encodings = {
    #     "known_face_encodings": encodings,
    #     "known_face_names": names,
    #     "face_id": employee_ids,
    # }
    # print(all_face_encodings)
    
    # f = open('face_recognition_data/embeddings.pickle', 'wb')
    # f.write(pickle.dumps(all_face_encodings))
    # f.close()
    # messages.success(request, 'Training dữ liệu thành công')

    embeddings_model_file = "face_recognition_data/embeddings.pickle"
    staff_details = Extract_Embeddings.get_staff_details(embeddings_model_file)
    staff_id = Extract_Embeddings.get_staff_id(embeddings_model_file)

    if not os.path.exists(embeddings_model_file):
        for i in range(0, len(images)):
            images[i] = face_recognition.load_image_file(files[i])
            # boxes = face_recognition.face_locations(images[i],model="hog")
            encodings[i] = face_recognition.face_encodings(images[i],num_jitters=1)[0]
            print(f"Data saved for {i+1} images...")

         # Create arrays of known face encodings and their names
        all_face_encodings = {
            "known_face_encodings": encodings,
            "known_face_names": names,
            "face_id": employee_ids,
        }
        print(all_face_encodings)
        
        f = open('face_recognition_data/embeddings.pickle', 'wb')
        f.write(pickle.dumps(all_face_encodings))
        f.close()
        messages.success(request, 'Training dữ liệu thành công')    
    else:
        [old_data,unique_names,faces_id] = Extract_Embeddings.check_pretrained_file(embeddings_model_file)
        remaining_names = Extract_Embeddings.get_remaining_names(staff_details,unique_names)
        face_id = Extract_Embeddings.get_remaining_faceid(staff_id,faces_id)
        print(face_id)
        # data = Extract_Embeddings.get_remaining_face(staff_details,remaining_names)
        data = Extract_Embeddings.get_remaining_face(staff_details,face_id)

        if data != None:
            [encodings,names,employee_ids] = data
            new_data = {"known_face_encodings":encodings,"known_face_names":names,"face_id":employee_ids}
            combined_data = {"known_face_encodings":[],"known_face_names":[],"face_id":[]}
            combined_data["known_face_encodings"] = old_data["known_face_encodings"] + new_data["known_face_encodings"]
            combined_data["known_face_names"] = old_data["known_face_names"] + new_data["known_face_names"]
            combined_data["face_id"] = old_data["face_id"] + new_data["face_id"]
            
            f = open('face_recognition_data/embeddings.pickle', 'wb')
            f.write(pickle.dumps(combined_data))
            f.close()
            messages.success(request, 'Training dữ liệu thành công')
        else:
             # Load face encodings
            with open('face_recognition_data/embeddings.pickle', 'rb') as f:
                all_face_encodings = pickle.load(f)
            print(all_face_encodings)
            messages.success(request, f'Không tìm thấy dữ liệu mới... dữ liệu đã được huấn luyện rồi')
  
    return redirect('/take_attendance/')

def detect_with_webcam(request):
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    images = []
    encodings = []
    names = []
    files = []
    employeeIds = []

    employees = Employee.objects.all()

    for employee in employees:
        # images.append(employee.name + "_image")
        # encodings.append(employee.name + "_face_encoding")
        # files.append(employee.profile_pic)
        # names.append(employee.name)  # + "\nID: " + str(employee.id)
        employeeIds.append(employee.id)
       
    employee_ids = employeeIds
    
    # Load face encodings
    with open('face_recognition_data/embeddings.pickle', 'rb') as f:
        all_face_encodings = pickle.load(f)
    
    the_values = all_face_encodings.values()
    known_face_names = list(the_values)[1]
    known_face_encodings = np.array(list(the_values)[0])

    eye_blink_counter = 0
    eye_blink_total = 0
    random_blink_number = random.randint(N_MIN_EYE_MIN,N_MAX_EYE_BLINK)
    current_name = None
    font = cv2.FONT_HERSHEY_DUPLEX
    
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Loop through each face in this frame of video
        for index, (loc, face_encoding, landmark) in enumerate(zip(face_locations, face_encodings, face_landmarks)):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                temp_name = current_name
                employee_id = employee_ids[best_match_index]
                current_name = known_face_names[best_match_index]
            else:
                current_name = "Unknown"
            
            if not current_name == "Unknown": 
                # Eye blink detection
                left_eye_points = np.array(landmark['left_eye'], dtype=np.int32)
                right_eye_points = np.array(landmark['right_eye'], dtype=np.int32)

                EAR_avg = (get_EAR_ratio(left_eye_points) + get_EAR_ratio(right_eye_points) ) / 2
                # Check if EAR ratio is less than threshold
                if EAR_avg < EAR_ratio_threshold:
                    eye_blink_counter += 1
                else:
                    # Check if counter is greater than min_frames_eyes_closed threshold
                    if eye_blink_counter >= MIN_FRAMES_EYES_CLOSED:
                        eye_blink_total += 1

                    # Reset eye blink counter
                    eye_blink_counter = 0
                
                # If temp_name doesn't matches , reset eye_blink_total and set new random_blink_number
                if temp_name != current_name:
                    eye_blink_total = 0
                    random_blink_number = random.randint(N_MIN_EYE_MIN,N_MAX_EYE_BLINK)
                # Set messages and face box color
                blink_message = f"Blink {random_blink_number} times, blinks:{eye_blink_total}"
                
                today = datetime.now()
                day = today.date()
                time = today.strftime("%H:%M:%S")
                
                try:
                    attendance_exist = Attendance.objects.get(
                        employee_id=employee_id, attendance_date=day)
                except:
                    attendance_exist = None
                
                if attendance_exist:
                    attendence_message = "Diem danh thanh cong"
                    face_box_color = SUCCESS_FACE_BOX_COLOR
                    text_color = TEXT_IN_SUCCESS_COLOR
                    eye_blink_total = 0
                    eye_blink_counter = 0
                else:
                    face_box_color = DEFAULT_FACE_BOX_COLOR
                    text_color = TEXT_IN_FRAME_COLOR
                    attendence_message =""
                    if random_blink_number == eye_blink_total:
                        if matches[best_match_index]:
                            face_box_color = SUCCESS_FACE_BOX_COLOR # Set face box color to green for one frame

                            employee_obj = Employee.objects.get(id=employee_id)
                            
                            employee_attendance = Attendance.objects.create(
                                employee_id=employee_obj,
                                name=employee_obj.name,
                                department_id=employee_obj.department_id.department_name,
                                attendance_date=day,
                                attendance_time_in=time,
                                attendance_time_out="",
                            )
                            employee_attendance.save()
                            
                            # Reset random_blink_number, and eye blink constants
                            random_blink_number = random.randint(N_MIN_EYE_MIN,N_MAX_EYE_BLINK)
                            eye_blink_total = 0
                            eye_blink_counter = 0

                # Draw Eye points and display blink_message and attendence_message
                cv2.polylines(frame, [left_eye_points], True, EYE_COLOR , 1)
                cv2.polylines(frame, [right_eye_points], True, EYE_COLOR , 1)
                cv2.putText(frame,blink_message,(10,50),cv2.FONT_HERSHEY_PLAIN,1.5,TEXT_IN_FRAME_COLOR,2)
                cv2.putText(frame,attendence_message,(20,450),cv2.FONT_HERSHEY_PLAIN,2,SUCCESS_FACE_BOX_COLOR,2)
            else:
                # Set face_box_color for unknown face
                face_box_color = UNKNOWN_FACE_BOX_COLOR

            # Draw Reactangle around faces with their names
            cv2.rectangle(frame,(loc[3],loc[0]),(loc[1],loc[2]),face_box_color,2) # top-right, bottom-left
            cv2.putText(frame,current_name,(loc[3],loc[0]-5),cv2.FONT_HERSHEY_PLAIN,2,text_color,2)
            
        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    return redirect("/manage_attendance")

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def manage_attendance(request):
    if request.method == "POST":
        year = request.POST.get('year')
        month = request.POST.get('month')
        display = True
        
        data = Attendance.objects.filter(created_at__year=year,created_at__month=month,present=True)
        count_attendance = data.values('employee_id__id','name','department_id').annotate(dcount=Count('employee_id__id'))
        totalDays = monthrange(2021, 11)[1]
        
        if not count_attendance:
            messages.warning(request, "Không có dữ liệu để hiển thị")
            display = False

        context = {
            "data":data,
            "count_attendance":count_attendance,
            "display": display,
            "totalDays": totalDays,
            "month": month,
            "year": year
        }

        return render(request, "hod_template/manage_attendance_template.html", context)
    else:
        return render(request, "hod_template/manage_attendance_template.html")



# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def view_employee_attendance(request, employee_id, month, year):
    data = Attendance.objects.filter(employee_id=employee_id,created_at__year=year,created_at__month=month,present=True).order_by('-created_at')
    
    context = {
        "data": data,
        'employee_id': employee_id,
        'month': month,
        'year': year
    }
    return render(request, "hod_template/view_employee_attendance.html", context)

# @cache_control(no_cache=True, must_revalidate=True, no_store=True)
@login_required(login_url="login")
@admin_only
def export_excel(request, employee_id, month, year):
    response = HttpResponse(content_type='application/ms-excel')
    response['Content-Disposition'] = 'attachment; filename=Diemdanh' + \
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

    data = Attendance.objects.filter(employee_id=employee_id,created_at__year=year,created_at__month=month,present=True).values_list(
        'employee_id', 'name','department_id', 'attendance_date', 'attendance_time_in','attendance_time_out')
    
    rows = data

    for row in rows:
        row_num += 1

        for col_num in range(len(row)):
            ws.write(row_num, col_num, str(row[col_num]), font_style)
    wb.save(response)

    return response

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def mark_your_attendance(request):
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    images = []
    encodings = []
    names = []
    files = []
    employeeIds = []

    employees = Employee.objects.all()

    for employee in employees:
        # images.append(employee.name + "_image")
        # encodings.append(employee.name + "_face_encoding")
        # files.append(employee.profile_pic)
        names.append(employee.name)
        employeeIds.append(employee.id)
       
    employee_ids = employeeIds
    
    # Load face encodings
    with open('face_recognition_data/embeddings.pickle', 'rb') as f:
        all_face_encodings = pickle.load(f)
    
    the_values = all_face_encodings.values()
    # known_face_names = list(the_values)[1]
    known_face_names = names
    known_face_encodings = np.array(list(the_values)[0])

    eye_blink_counter = 0
    eye_blink_total = 0
    random_blink_number = random.randint(N_MIN_EYE_MIN,N_MAX_EYE_BLINK)
    current_name = None
    font = cv2.FONT_HERSHEY_DUPLEX
    process_this_frame = True
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
        if process_this_frame:
            # Find all the faces and face enqcodings in the frame of video
            face_locations = face_recognition.face_locations(rgb_frame)
            face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Loop through each face in this frame of video
            for index, (loc, face_encoding, landmark) in enumerate(zip(face_locations, face_encodings, face_landmarks)):
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                

                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    temp_name = current_name
                    employee_id = employee_ids[best_match_index]
                    current_name = known_face_names[best_match_index]
                else:
                    current_name = "Unknown"
                
                if not current_name == "Unknown": 
                    # Eye blink detection
                    left_eye_points = np.array(landmark['left_eye'], dtype=np.int32)
                    right_eye_points = np.array(landmark['right_eye'], dtype=np.int32)

                    EAR_avg = (get_EAR_ratio(left_eye_points) + get_EAR_ratio(right_eye_points) ) / 2
                    # Check if EAR ratio is less than threshold
                    
                    if EAR_avg < EAR_ratio_threshold:
                        eye_blink_counter += 1
                    else:
                        # Check if counter is greater than min_frames_eyes_closed threshold
                        if eye_blink_counter >= MIN_FRAMES_EYES_CLOSED:
                            eye_blink_total += 1

                        # Reset eye blink counter
                        eye_blink_counter = 0
                    
                    # If temp_name doesn't matches , reset eye_blink_total and set new random_blink_number
                    if temp_name != current_name:
                        eye_blink_total = 0
                        random_blink_number = random.randint(N_MIN_EYE_MIN,N_MAX_EYE_BLINK)
                    # Set messages and face box color
                    blink_message = f"Blink {random_blink_number} times, blinks:{eye_blink_total}"
                    
                    today = datetime.now()
                    day = today.date()
                    time = today.strftime("%H:%M:%S")
                    
                    try:
                        attendance_exist = Attendance.objects.get(
                            employee_id=employee_id, attendance_date=day)
                    except:
                        attendance_exist = None
                    
                    if attendance_exist:
                        if attendance_exist.attendance_time_in:
                            attendence_message = "Diem danh thanh cong"
                            face_box_color = SUCCESS_FACE_BOX_COLOR
                            text_color = TEXT_IN_SUCCESS_COLOR
                            eye_blink_total = 0
                            eye_blink_counter = 0
                    else:
                        face_box_color = DEFAULT_FACE_BOX_COLOR
                        text_color = TEXT_IN_FRAME_COLOR
                        attendence_message =""
                        if random_blink_number == eye_blink_total:
                            if matches[best_match_index]:
                                face_box_color = SUCCESS_FACE_BOX_COLOR # Set face box color to green for one frame

                                employee_obj = Employee.objects.get(id=employee_id)
                                
                                employee_attendance = Attendance.objects.create(
                                    employee_id=employee_obj,
                                    name=employee_obj.name,
                                    department_id=employee_obj.department_id.department_name,
                                    attendance_date=day,
                                    attendance_time_in=time,
                                    attendance_time_out="",
                                )
                                employee_attendance.save()
                                
                                # Reset random_blink_number, and eye blink constants
                                random_blink_number = random.randint(N_MIN_EYE_MIN,N_MAX_EYE_BLINK)
                                eye_blink_total = 0
                                eye_blink_counter = 0

                    # Draw Eye points and display blink_message and attendence_message
                    # cv2.polylines(frame, [left_eye_points], True, EYE_COLOR , 1)
                    # cv2.polylines(frame, [right_eye_points], True, EYE_COLOR , 1)
                    cv2.putText(frame,blink_message,(10,50),cv2.FONT_HERSHEY_PLAIN,1.5,TEXT_IN_FRAME_COLOR,2)
                    cv2.putText(frame,"ID: " +str(employee_id),(loc[3],loc[0]-30),cv2.FONT_HERSHEY_PLAIN,2,text_color,2)
                    cv2.putText(frame,attendence_message,(20,450),cv2.FONT_HERSHEY_PLAIN,2,SUCCESS_FACE_BOX_COLOR,2)
                else:
                    # Set face_box_color for unknown face
                    face_box_color = UNKNOWN_FACE_BOX_COLOR
                    text_color = TEXT_IN_FRAME_COLOR

                
                # Draw Reactangle around faces with their names
                cv2.rectangle(frame,(loc[3],loc[0]),(loc[1],loc[2]),face_box_color,2) # top-right, bottom-left
                cv2.putText(frame,current_name,(loc[3],loc[0]-5),cv2.FONT_HERSHEY_PLAIN,2,text_color,2)
            
        # Display the resulting image
        cv2.imshow('Mark Attendance - In - Press q to exit', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    return redirect("home")


def mark_your_attendance_out(request):
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    images = []
    encodings = []
    names = []
    files = []
    employeeIds = []

    employees = Employee.objects.all()

    for employee in employees:
        # images.append(employee.name + "_image")
        # encodings.append(employee.name + "_face_encoding")
        # files.append(employee.profile_pic)
        names.append(employee.name)
        employeeIds.append(employee.id)
       
    employee_ids = employeeIds
    
    # Load face encodings
    with open('face_recognition_data/embeddings.pickle', 'rb') as f:
        all_face_encodings = pickle.load(f)
    
    the_values = all_face_encodings.values()
    # known_face_names = list(the_values)[1]
    known_face_names = names
    known_face_encodings = np.array(list(the_values)[0])

    eye_blink_counter = 0
    eye_blink_total = 0
    random_blink_number = random.randint(N_MIN_EYE_MIN,N_MAX_EYE_BLINK)
    current_name = None
    font = cv2.FONT_HERSHEY_DUPLEX
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Loop through each face in this frame of video
        for index, (loc, face_encoding, landmark) in enumerate(zip(face_locations, face_encodings, face_landmarks)):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            

            if matches[best_match_index]:
                temp_name = current_name
                employee_id = employee_ids[best_match_index]
                current_name = known_face_names[best_match_index]
            else:
                current_name = "Unknown"
            
            if not current_name == "Unknown": 
                # Eye blink detection
                left_eye_points = np.array(landmark['left_eye'], dtype=np.int32)
                right_eye_points = np.array(landmark['right_eye'], dtype=np.int32)

                EAR_avg = (get_EAR_ratio(left_eye_points) + get_EAR_ratio(right_eye_points) ) / 2
                # Check if EAR ratio is less than threshold
                # print(EAR_avg)
                if EAR_avg < EAR_ratio_threshold:
                    eye_blink_counter += 1
                else:
                    # Check if counter is greater than min_frames_eyes_closed threshold
                    if eye_blink_counter >= MIN_FRAMES_EYES_CLOSED:
                        eye_blink_total += 1

                    # Reset eye blink counter
                    eye_blink_counter = 0
                
                # If temp_name doesn't matches , reset eye_blink_total and set new random_blink_number
                if temp_name != current_name:
                    eye_blink_total = 0
                    random_blink_number = random.randint(N_MIN_EYE_MIN,N_MAX_EYE_BLINK)
                # Set messages and face box color
                blink_message = f"Blink {random_blink_number} times, blinks:{eye_blink_total}"
                
                today = datetime.now()
                day = today.date()
                time = today.strftime("%H:%M:%S")
                
                try:
                    attendance_exist = Attendance.objects.get(
                        employee_id=employee_id, attendance_date=day)
                except:
                    attendance_exist = None

               
                if attendance_exist:          
                    if attendance_exist.attendance_time_in: 
                        if attendance_exist.attendance_time_out:
                            attendence_message = "Diem danh thanh cong"
                            face_box_color = SUCCESS_FACE_BOX_COLOR
                            text_color = TEXT_IN_SUCCESS_COLOR
                            eye_blink_total = 0
                            eye_blink_counter = 0
                        else:
                            face_box_color = DEFAULT_FACE_BOX_COLOR
                            text_color = TEXT_IN_FRAME_COLOR
                            attendence_message =""
                            if random_blink_number == eye_blink_total:
                                if matches[best_match_index]:
                                    face_box_color = SUCCESS_FACE_BOX_COLOR # Set face box color to green for one frame

                                    attendance_out = Attendance.objects.get(employee_id=employee_id, attendance_date=day)
                                    attendance_out.attendance_time_out = time
                                    attendance_out.present = True
                                    attendance_out.save()
                                    
                                    # Reset random_blink_number, and eye blink constants
                                    random_blink_number = random.randint(N_MIN_EYE_MIN,N_MAX_EYE_BLINK)
                                    eye_blink_total = 0
                                    eye_blink_counter = 0

                        # Draw Eye points and display blink_message and attendence_message
                        # cv2.polylines(frame, [left_eye_points], True, EYE_COLOR , 1)
                        # cv2.polylines(frame, [right_eye_points], True, EYE_COLOR , 1)
                        cv2.putText(frame,blink_message,(10,50),cv2.FONT_HERSHEY_PLAIN,1.5,TEXT_IN_FRAME_COLOR,2)
                        cv2.putText(frame,"ID: " +str(employee_id),(loc[3],loc[0]-30),cv2.FONT_HERSHEY_PLAIN,2,text_color,2)
                        cv2.putText(frame,attendence_message,(20,450),cv2.FONT_HERSHEY_PLAIN,2,SUCCESS_FACE_BOX_COLOR,2)
                else:
                    attendence_message = "Chua diem danh vao"
                    face_box_color = DEFAULT_FACE_BOX_COLOR
                    text_color = TEXT_IN_FRAME_COLOR
                    eye_blink_total = 0
                    eye_blink_counter = 0
                    # Draw Eye points and display blink_message and attendence_message
                    # cv2.polylines(frame, [left_eye_points], True, EYE_COLOR , 1)
                    # cv2.polylines(frame, [right_eye_points], True, EYE_COLOR , 1)
                    cv2.putText(frame,attendence_message,(20,450),cv2.FONT_HERSHEY_PLAIN,2,text_color,2)
            else:
                # Set face_box_color for unknown face
                face_box_color = UNKNOWN_FACE_BOX_COLOR
                text_color = TEXT_IN_FRAME_COLOR

            # Draw Reactangle around faces with their names
            cv2.rectangle(frame,(loc[3],loc[0]),(loc[1],loc[2]),face_box_color,2) # top-right, bottom-left
            cv2.putText(frame,current_name,(loc[3],loc[0]-5),cv2.FONT_HERSHEY_PLAIN,2,text_color,2)
            
        # Display the resulting image
        cv2.imshow('Mark Attendance - Out - Press q to exit', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
    return redirect("home")
    