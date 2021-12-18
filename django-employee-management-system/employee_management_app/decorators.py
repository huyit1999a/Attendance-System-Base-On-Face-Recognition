from django.http import HttpResponse
from django.shortcuts import redirect
from employee_management_app.EmailBackEnd import EmailBackEnd

def unauthenticated_user(view_func):
    def wrapper_func(request, *args, **kwargs):
      
        user = request.user

        # Check whether the user is logged in or not
        if user.is_authenticated:
            if user.user_type == "1":
                return redirect("admin_home")

            elif user.user_type == "2":
                return redirect("employee_home")
            
        else:
            return view_func(request, *args, **kwargs)
    
    return wrapper_func

def allowed_users(allowed_roles=[]):
    def decorator(view_func):
        def wrapper_func(request, *args, **kwargs):
            group = None
            if request.user.groups.exists():
                group = request.user.groups.all()[0].name
            
            if group in allowed_roles:
                return view_func(request, *args, **kwargs)
            else:
                return HttpResponse('You are not authorized to view this page')
        return wrapper_func
    return decorator

def admin_only(view_func):
    def wrapper_function(request, *args, **kwargs):
        user = request.user
        if user.is_authenticated:
            if user.user_type == "1":
                return view_func(request, *args, **kwargs)

            if user.user_type == "2":
                return redirect("employee_home")
              
    return wrapper_function
  