import subprocess
import sys
#from concurrent.futures import ThreadPoolExecutor

result = subprocess.run(["node", "test.js", "test.txt"], capture_output=True, text=True)
result.stdout = result.stdout.replace("\n", "")
result.stdout = result.stdout.replace("[", "")
result.stdout = result.stdout.replace("]", "")
result.stdout = result.stdout.replace(" ", "")
result.stdout = result.stdout.replace("'", "")
a = result.stdout.split(',')
print(a)
print(type(a))
print(len(a))
#def a(b):
#    result = subprocess.run(["node", "test.js"] + list(b), capture_output=True, text=True)
#    return result.stdout

#c=[]

#with ThreadPoolExecutor() as executor:
#    futures = [executor.submit(a, list3) for list3 in list2]
#    for future in futures:
#        c.append(future.result())

#print(c)