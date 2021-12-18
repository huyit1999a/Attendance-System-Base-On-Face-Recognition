from django import forms
from django.forms import Form
from employee_management_app.models import Departments
from django.core.validators import FileExtensionValidator

class DateInput(forms.DateInput):
    input_type = "date"


class AddEmployeeForm(forms.Form):
    email = forms.EmailField(
        label="Địa chỉ email",
        max_length=50,
        widget=forms.EmailInput(attrs={"class": "form-control"}),
    )
    password = forms.CharField(
        label="Mật khẩu",
        max_length=50,
        widget=forms.PasswordInput(attrs={"class": "form-control"}),
    )
    name = forms.CharField(
        label="Họ và tên",
        max_length=50,
        widget=forms.TextInput(attrs={"class": "form-control"}),
    )

    address = forms.CharField(
        label="Địa chỉ",
        max_length=50,
        widget=forms.TextInput(attrs={"class": "form-control"}),
    )

    gender_list = (("Nam", "Nam"), ("Nữ", "Nữ"))

    profile_pic = forms.FileField(
        label="Ảnh đại diện",
        required=True,
        widget=forms.FileInput(attrs={"class": "form-control"}),
        validators=[FileExtensionValidator(['jpg','png','jpeg'])]
    )

    gender = forms.ChoiceField(
        label="Giới tính",
        choices=gender_list,
        widget=forms.Select(attrs={"class": "form-control"}),
    )

    def clean_photo(self):
        image_file = self.cleaned_data.get('profile_pic')
        if not image_file.name.endswith(".jpg" , ".jpeg" , ".png"):
            raise forms.ValidationError("File ảnh phải là đuôi .jpg, .jpeg, .png")
        return image_file

    def __init__(self, *args, **kwargs):
        departments = Departments.objects.all()
        department_list = []
        for department in departments:
            single_department = (department.id, department.department_name)
            department_list.append(single_department)

        super(AddEmployeeForm, self).__init__(*args, **kwargs)
        self.fields["department_id"] = forms.ChoiceField(
            label="Phòng ban",
            choices=department_list,
            widget=forms.Select(attrs={"class": "form-control"}),
        )


class EditEmployeeForm(forms.Form):
    email = forms.EmailField(
        label="Địa chỉ email",
        max_length=50,
        widget=forms.EmailInput(
            attrs={"class": "form-control", "readonly": "readonly"}
        ),
    )
    name = forms.CharField(
        label="Họ và tên",
        max_length=50,
        widget=forms.TextInput(attrs={"class": "form-control"}),
    )

    address = forms.CharField(
        label="Địa chỉ",
        max_length=50,
        widget=forms.TextInput(attrs={"class": "form-control"}),
    )

    gender_list = (("Nam", "Nam"), ("Nữ", "Nữ"))

    profile_pic = forms.FileField(
        label="Ảnh đại diện",
        required=False,
        widget=forms.FileInput(attrs={"class": "form-control"}),
        validators=[FileExtensionValidator(['jpg','png','jpeg'])]
    )

    gender = forms.ChoiceField(
        label="Giới tính",
        choices=gender_list,
        widget=forms.Select(attrs={"class": "form-control"}),
    )

    def clean_photo(self):
        image_file = self.cleaned_data.get('profile_pic')
        if not image_file.name.endswith(".jpg" , ".jpeg" , ".png"):
            raise forms.ValidationError("File ảnh phải là đuôi .jpg, .jpeg, .png")
        return image_file
        

    def __init__(self, *args, **kwargs):
        departments = Departments.objects.all()
        department_list = []
        for department in departments:
            single_department = (department.id, department.department_name)
            department_list.append(single_department)

        super(EditEmployeeForm, self).__init__(*args, **kwargs)
        self.fields["department_id"] = forms.ChoiceField(
            label="Phòng ban",
            choices=department_list,
            widget=forms.Select(attrs={"class": "form-control"}),
        )
