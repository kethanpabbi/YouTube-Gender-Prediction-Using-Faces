import cgi
form = cgi.FieldStorage()
searchterm =  form.getvalue('searchbox')
print(searchterm)