import cv2
import os
import numpy as np
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from employee_management_app.models import (
    CustomUser,
    Departments,
    Employee,
    Attendance,
)

rootdir = os.getcwd()

class Extract_Embeddings():      
    def check_pretrained_file(embeddings_model):
        data = pickle.loads(open(embeddings_model, "rb").read())
        names = np.array(data["known_face_names"])
        faces_id = np.array(data["face_id"])
        unique_names = np.unique(names).tolist()
        return [data,unique_names,faces_id]

    def get_staff_details(embeddings_model):
        names = []
        faces_id = []
        employees = Employee.objects.all()

        for employee in employees:
            names.append(employee.name) 
            faces_id.append(employee.id)
        
        names = [] 
        faces_id = []
        staff_details = {}
        employees = Employee.objects.all()

        for employee in employees:
            names.append(employee.name) 
            faces_id.append(employee.id)
        
        
        for index, item in enumerate(names):
            staff_details[item] = faces_id[index]
        
        return staff_details

    def get_staff_id(embeddings_model):
        names = []
        faces_id = []
        employees = Employee.objects.all()

        for employee in employees:
            names.append(employee.name) 
            faces_id.append(employee.id)
        
        names = [] 
        faces_id = []
        staff_details = {}
        employees = Employee.objects.all()

        for employee in employees:
            names.append(employee.name) 
            faces_id.append(employee.id)
        
        
        for index, item in enumerate(names):
            staff_details[faces_id[index]] = item
        
        return staff_details

    def get_remaining_faceid(dictionaries,face_id):
        face_id = np.setdiff1d(list(dictionaries.keys()),face_id).tolist()
        return face_id

    def get_remaining_names(dictionaries,unique_names):
        remaining_names = np.setdiff1d(list(dictionaries.keys()),unique_names).tolist()
        return remaining_names

    def get_all_face_pixels(dictionaries):
        image_ids = []
        image_paths = []
        image_arrays = []
        names = []
        face_ids = []
        for category in list(dictionaries.keys()):
            path = os.path.join(dataset_dir,category + "_" + dictionaries[category])
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path,img))
                image_paths.append(os.path.join(path,img))
                image_ids.append(img)
                image_arrays.append(img_array)
                names.append(category)
                face_ids.append(dictionaries[category])
        return [image_ids,image_paths,image_arrays,names,face_ids]

    def get_remaining_face(dictionaries,remaining_names):
        dictionaries = dictionaries
        remaining_names = remaining_names
        # Load a sample picture and learn how to recognize it.
        images = []
        encodings = []
        names = []
        files = []
        employee_ids = []
        
        if len(remaining_names) != 0:
            employees = Employee.objects.filter(name__in=remaining_names)
            
            for employee in employees:
                images.append(employee.name + "_image")
                encodings.append(employee.name + "_face_encoding")
                files.append(employee.profile_pic)
                names.append(employee.name)  # + "\nID: " + str(employee.id)
                employee_ids.append(employee.id)
            for i in range(0, len(images)):
                images[i] = face_recognition.load_image_file(files[i])
                boxes = face_recognition.face_locations(images[i],model="hog")
                encodings[i] = face_recognition.face_encodings(images[i],boxes,num_jitters=1)[0]
                print(f"Data saved for {i+1} images...")
            
            # Create arrays of known face encodings and their name
            
            return [encodings,names,employee_ids]
        else:
            return None

    def get_remaining_face(dictionaries,face_id):
        dictionaries = dictionaries
        face_id = face_id
        # Load a sample picture and learn how to recognize it.
        images = []
        encodings = []
        names = []
        files = []
        employee_ids = []
        
        if len(face_id) != 0:
            employees = Employee.objects.filter(id__in=face_id)
            
            for employee in employees:
                images.append(employee.name + "_image")
                encodings.append(employee.name + "_face_encoding")
                files.append(employee.profile_pic)
                names.append(employee.name)  # + "\nID: " + str(employee.id)
                employee_ids.append(employee.id)
            for i in range(0, len(images)):
                images[i] = face_recognition.load_image_file(files[i])
                boxes = face_recognition.face_locations(images[i],model="hog")
                encodings[i] = face_recognition.face_encodings(images[i],boxes,num_jitters=1)[0]
                print(f"Data saved for {i+1} images...")
            
            # Create arrays of known face encodings and their name
            
            return [encodings,names,employee_ids]
        else:
            return None





				
