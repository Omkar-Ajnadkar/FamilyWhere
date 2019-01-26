from wtforms import Form, StringField, IntegerField, DateField, PasswordField, SubmitField, BooleanField, SelectField, TextAreaField, RadioField, FieldList, FormField
from wtforms import validators, ValidationError

class AddForm(Form):
    childname = StringField('Child Name', validators=[validators.DataRequired()])  
    parentname = StringField('Parent Name', validators=[validators.DataRequired()])
    email = StringField('Email', validators=[validators.DataRequired()])
    phone = StringField('Phone', validators=[validators.DataRequired()])
    address = StringField('Address', validators=[validators.DataRequired()])
  